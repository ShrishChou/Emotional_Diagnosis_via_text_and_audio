"""Latency and memory benchmark -> evals/system/latency.json.

50 MELD test clips (fixed random sample), after 3 warm-up runs. Each clip goes
through the full path: Whisper ASR -> text + audio encoders -> heads -> JSON
state -> full LLM reply. Reports p50 / p95 per stage, plus `state_no_asr`
(state latency when a transcript is supplied), peak process RSS and, on MPS /
CUDA, peak GPU memory.
"""
import os

# Inference is local only: refuse any Hugging Face network access. Weights must
# already be cached (training/01_check_environment.py downloads them).
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")  # hide model load reports
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import resource
import sys
import threading
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import psutil
import torch

from config import CFG, write_json
from src.pipeline import Pipeline
from src.responder import STYLES

N_CLIPS = 50
N_WARMUP = 3


class PeakMemory:
    """Samples process RSS and GPU memory every 20 ms in a background thread."""

    def __init__(self):
        self.proc = psutil.Process()
        self.peak_rss = 0
        self.peak_gpu = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def gpu_bytes(self) -> int:
        if CFG.device == "mps":
            return torch.mps.driver_allocated_memory()
        if CFG.device == "cuda":
            return torch.cuda.memory_allocated()
        return 0

    def _run(self):
        while not self._stop.is_set():
            self.peak_rss = max(self.peak_rss, self.proc.memory_info().rss)
            self.peak_gpu = max(self.peak_gpu, self.gpu_bytes())
            time.sleep(0.02)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def run_one(pipe: Pipeline, row) -> dict:
    state = pipe.analyze(CFG.root / row.wav_path, text=None, context=row.context,
                         utterance_id=f"dia{row.dialogue_id}_utt{row.utterance_id}")
    n_chunks = sum(1 for ev in pipe.respond_spoken(state, row.context) if ev["type"] == "token")
    lat = dict(state["latency_ms"])
    lat["state_no_asr"] = lat["state_total"] - lat["asr"]
    # End of the person's speech -> the robot's first audio is ready to play.
    lat["speech_end_to_audio"] = lat["state_total"] + lat["tts_first_audio"]
    lat["speech_end_to_audio_no_asr"] = lat["state_no_asr"] + lat["tts_first_audio"]
    lat["reply_chunks"] = n_chunks
    return lat


def main():
    df = pd.read_csv(CFG.manifest_dir / "test.csv", keep_default_na=False)
    clips = df.sample(n=N_CLIPS + N_WARMUP, random_state=0)

    with PeakMemory() as mem:
        t0 = time.perf_counter()
        pipe = Pipeline(load_asr=True, load_llm=True, load_tts=True)
        pipe.warmup()
        cold = time.perf_counter() - t0
        rss_after_load = psutil.Process().memory_info().rss

        for row in clips.iloc[:N_WARMUP].itertuples():
            run_one(pipe, row)
        runs = []
        for i, row in enumerate(clips.iloc[N_WARMUP:].itertuples()):
            runs.append({"clip": f"dia{row.dialogue_id}_utt{row.utterance_id}",
                         "duration_s": row.duration_s, **run_one(pipe, row)})
            print(f"{i + 1}/{N_CLIPS} {runs[-1]['clip']}: state {runs[-1]['state_total']:.0f} ms "
                  f"(asr {runs[-1]['asr']:.0f}), first token {runs[-1]['llm_first_token']:.0f} ms")

    stages = ["audio_load", "asr", "text_enc", "audio_enc", "heads", "state_no_asr",
              "state_total", "llm_first_token", "llm_total", "tts_first_audio", "tts_total",
              "speech_end_to_audio_no_asr", "speech_end_to_audio"]
    summary = {s: {"p50": round(float(np.percentile([r[s] for r in runs], 50)), 1),
                   "p95": round(float(np.percentile([r[s] for r in runs], 95)), 1)} for s in stages}
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    ru_bytes = ru if sys.platform == "darwin" else ru * 1024

    out = {
        "device": CFG.device,
        "n_clips": N_CLIPS,
        "n_warmup": N_WARMUP,
        "clip_duration_s": {"p50": float(np.median(clips.iloc[N_WARMUP:].duration_s)),
                            "max": float(clips.iloc[N_WARMUP:].duration_s.max())},
        "cold_load_s": {"total": round(cold, 2), **pipe.cold_load_s},
        "latency_ms": summary,
        "memory": {
            "peak_rss_gb_sampled": round(mem.peak_rss / 1e9, 2),
            "peak_rss_gb_getrusage": round(ru_bytes / 1e9, 2),
            "rss_after_load_gb": round(rss_after_load / 1e9, 2),
            "peak_gpu_gb": round(mem.peak_gpu / 1e9, 2) if CFG.device != "cpu" else None,
            "gpu_memory_source": {"mps": "torch.mps.driver_allocated_memory (sampled every 20 ms)",
                                  "cuda": "torch.cuda.memory_allocated (sampled every 20 ms)"}.get(CFG.device),
        },
        "notes": "Timing starts when the finished utterance is handed to the pipeline (end of speech). "
                 "state_total includes Whisper ASR; state_no_asr is the same run minus the ASR stage, "
                 "i.e. the latency when a transcript is supplied. Replies are also spoken (Kokoro-82M "
                 "on the CPU, one sentence at a time): tts_first_audio runs from the start of the reply "
                 "to the first sentence's audio being ready, and speech_end_to_audio adds the state "
                 "latency. llm_total includes time the token loop waited on synthesis.",
        "llm": {"model": CFG.llm_model, "reply_style": CFG.reply_style,
                **{k: v for k, v in vars(STYLES[CFG.reply_style]).items() if k != "name"}},
        "targets_ms": {"state_gpu": 300, "state_cpu": 1000, "first_token_gpu": 1000, "first_token_cpu": 3000},
        "runs": runs,
    }
    write_json(CFG.out("system", "latency.json"), out)
    for s in stages:
        print(f"{s:16s} p50 {summary[s]['p50']:8.1f} ms   p95 {summary[s]['p95']:8.1f} ms")
    print(out["memory"])


if __name__ == "__main__":
    main()
