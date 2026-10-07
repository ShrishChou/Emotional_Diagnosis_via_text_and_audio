"""Reply rules and speed across LLMs and prompt styles
-> evals/replies/reply_compliance.{json,md}, evals/replies/reply_eval/<config>.csv

What this measures, and what it does not: the inputs are MELD lines, spoken by sitcom
characters to each other, not to the character. So this checks that replies follow the
rules (length, no emotion words) and how fast they come, not whether they are good replies
in a real conversation; that needs real people talking to the character.

  50 MELD test clips (stratified by gold class, gold transcripts). Each configuration
  (LLM x reply style, see src/responder.py) writes one emotion-conditioned reply per
  clip, and each reply is checked automatically (src/reply_checks.py):
    within_2_sentences  at most two sentences
    ends_cleanly        not cut off mid-sentence (e.g. by the token limit)
    emotion_word        uses a word from the banned emotion lexicon
    announces_feeling   tells the listener how they feel ("I see you're upset",
                        "you seem really happy"), not enforced by decoding
  plus first-token and total generation latency.

  Compliance bar, fixed before any run:
    within_2_sentences >= 95%, ends_cleanly >= 90%, emotion_word <= 5%,
    announces_feeling <= 10%, first-token p95 < 1000 ms.
  Decision rule: the smallest model that meets the bar.
"""
import os

# Inference is local only: refuse any Hugging Face network access. Weights must
# already be cached (training/01_check_environment.py downloads them).
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")  # hide model load reports
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import argparse
import gc
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

from config import CFG, write_json
from src.pipeline import Pipeline
from src.reply_checks import announces_feeling, count_sentences, ends_cleanly, has_emotion_word
from src.responder import Responder

N_CLIPS = 50
N_WARMUP = 2

CONFIGS = [  # (name, model, style), smallest model first
    ("qwen2.5-1.5b_baseline", "Qwen/Qwen2.5-1.5B-Instruct", "baseline"),
    ("qwen2.5-1.5b_behavior", "Qwen/Qwen2.5-1.5B-Instruct", "behavior"),
    ("qwen2.5-3b_behavior", "Qwen/Qwen2.5-3B-Instruct", "behavior"),
    ("qwen3-4b-2507_behavior", "Qwen/Qwen3-4B-Instruct-2507", "behavior"),  # non-thinking variant
]
BAR = {"within_2_sentences": (">=", 0.95), "ends_cleanly": (">=", 0.90),
       "emotion_word": ("<=", 0.05), "announces_feeling": ("<=", 0.10),
       "first_token_p95_ms": ("<", 1000.0)}


def pick_clips(df: pd.DataFrame) -> pd.DataFrame:
    """8 per gold class, shuffled, first 50 (keeps rare emotions in the sample)."""
    return df.groupby("emotion").sample(n=8, random_state=0).sample(frac=1, random_state=0).head(N_CLIPS)


def cell(s: str) -> str:
    return s.replace("|", "/").replace("\n", " ").strip()


def generate(responder: Responder, state: dict, context: str, use_emotion: bool, seed: int, style: str):
    t0 = time.perf_counter()
    first, parts = None, []
    for chunk in responder.stream(state, context, use_emotion, seed, style):
        first = first if first is not None else (time.perf_counter() - t0) * 1000
        parts.append(chunk)
    return "".join(parts).strip(), first or 0.0, (time.perf_counter() - t0) * 1000


def free_gpu() -> None:
    gc.collect()
    if CFG.device == "mps":
        torch.mps.empty_cache()
    elif CFG.device == "cuda":
        torch.cuda.empty_cache()


def passes(m: dict) -> dict:
    ops = {">=": np.greater_equal, "<=": np.less_equal, "<": np.less}
    return {k: bool(ops[op](m[k], v)) for k, (op, v) in BAR.items()}


def run_config(name, model, style, clips, states, base_params) -> dict:
    responder = Responder(model)
    llm_params = sum(p.numel() for p in responder.model.parameters())
    for row in clips.head(N_WARMUP).itertuples():
        generate(responder, states[row.clip], row.context, True, 0, style)
    rows = []
    for i, row in enumerate(clips.itertuples()):
        reply, first, total = generate(responder, states[row.clip], row.context, True, i, style)
        rows.append({"clip": row.clip, "gold": row.emotion, "predicted": states[row.clip]["emotion"],
                     "transcript": row.text, "reply": reply, "first_token_ms": round(first, 1),
                     "total_ms": round(total, 1), "sentences": count_sentences(reply),
                     "ends_cleanly": ends_cleanly(reply), "emotion_word": has_emotion_word(reply),
                     "announces_feeling": announces_feeling(reply)})
    del responder
    free_gpu()
    out = pd.DataFrame(rows)
    out.to_csv(CFG.out("replies", "reply_eval", f"{name}.csv"), index=False)
    m = {
        "model": model, "style": style, "n": len(out),
        "llm_params": llm_params, "pipeline_total_params": base_params + llm_params,
        "within_2_sentences": float((out.sentences <= 2).mean()),
        "ends_cleanly": float(out.ends_cleanly.mean()),
        "emotion_word": float(out.emotion_word.mean()),
        "announces_feeling": float(out.announces_feeling.mean()),
        "empty": float((out.reply == "").mean()),
        "mean_words": float(out.reply.str.split().str.len().mean()),
        "first_token_p50_ms": float(np.percentile(out.first_token_ms, 50)),
        "first_token_p95_ms": float(np.percentile(out.first_token_ms, 95)),
        "total_p50_ms": float(np.percentile(out.total_ms, 50)),
        "total_p95_ms": float(np.percentile(out.total_ms, 95)),
    }
    m["passes"] = passes(m)
    m["meets_bar"] = all(m["passes"].values())
    print(f"{name}: " + ", ".join(f"{k} {v:.2f}" for k, v in m.items() if isinstance(v, float)), flush=True)
    return m


def write_compliance(results: dict) -> None:
    lines = ["| Config | LLM params | Pipeline total | ≤ 2 sentences | Ends cleanly | Emotion word | "
             "Announces feeling | First token p50 / p95 (ms) | Total p50 (ms) | Meets bar |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for name, m in results.items():
        lines.append(f"| {name} | {m['llm_params'] / 1e9:.2f}B | {m['pipeline_total_params'] / 1e9:.2f}B | "
                     f"{m['within_2_sentences']:.0%} | {m['ends_cleanly']:.0%} | {m['emotion_word']:.0%} | "
                     f"{m['announces_feeling']:.0%} | {m['first_token_p50_ms']:.0f} / {m['first_token_p95_ms']:.0f} | "
                     f"{m['total_p50_ms']:.0f} | {'yes' if m['meets_bar'] else 'no'} |")
    bar = ", ".join(f"{k} {op} {v}" for k, (op, v) in BAR.items())
    lines += ["", f"{N_CLIPS} MELD test clips, gold transcripts, emotion-conditioned replies, "
              f"device {CFG.device}. Bar (fixed before running): {bar}. "
              "In the behavior style the sentence limit and the emotion-word list are enforced "
              "in decoding, so those two columns are compliant by construction there; "
              "'announces feeling' is not enforced. The inputs are sitcom lines addressed to other "
              "characters, so this measures rule-following and speed, not reply quality."]
    CFG.out("replies", "reply_compliance.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", help="subset of config names (default: all)")
    args = ap.parse_args()

    df = pd.read_csv(CFG.manifest_dir / "test.csv", keep_default_na=False)
    clips = pick_clips(df).copy()
    clips["clip"] = [f"dia{d}_utt{u}" for d, u in zip(clips.dialogue_id, clips.utterance_id)]

    pipe = Pipeline(load_asr=False, load_llm=False)
    states = {row.clip: pipe.analyze(CFG.root / row.wav_path, text=row.text, context=row.context,
                                     utterance_id=row.clip, transcript_source="gold")
              for row in clips.itertuples()}
    del pipe
    free_gpu()

    params = json.loads(CFG.out("system", "params.json").read_text())
    base_params = params["total"] - params["components"]["reply_llm"]["params"]
    path = CFG.out("replies", "reply_compliance.json")
    results = json.loads(path.read_text())["configs"] if path.exists() else {}
    for name, model, style in CONFIGS:
        if args.configs and name not in args.configs:
            continue
        results[name] = run_config(name, model, style, clips, states, base_params)
    results = {c[0]: results[c[0]] for c in CONFIGS if c[0] in results}
    passing = [k for k in results if results[k]["meets_bar"] and results[k]["style"] == "behavior"]
    write_json(path, {"n_clips": N_CLIPS, "bar": {k: list(v) for k, v in BAR.items()},
                      "decision_rule": "smallest model that meets the bar",
                      "smallest_passing": passing[0] if passing else None, "configs": results})
    write_compliance(results)
    print(CFG.out("replies", "reply_compliance.md").read_text())


if __name__ == "__main__":
    main()
