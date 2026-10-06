"""Latency after idle gaps -> results/idle_latency.{json,md}

05_bench_latency.py runs clips back to back. A conversation never does: the robot sits
idle while the person talks. This measures state and first-token latency after idle
gaps of 0 to 20 s, in two conditions:

  cold  the turn arrives after the gap, nothing else happens
  ping  Pipeline.ping() runs 1 s before the turn (what the face page does when the
        person starts holding the talk button; speech gives at least that much time)

5 test clips (gold transcripts) x 3 repeats per gap and condition, after warm-up.
"""
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from config import CFG, write_json
from src.pipeline import Pipeline

GAPS_S = [0, 2, 5, 10, 20]
REPEATS = 3
PING_LEAD_S = 1.0


def turn(pipe: Pipeline, row) -> dict:
    state = pipe.analyze(CFG.root / row.wav_path, text=row.text, context=row.context,
                         transcript_source="gold")
    for _ in pipe.respond(state, row.context):
        pass
    lat = state["latency_ms"]
    return {"state_ms": lat["state_total"], "audio_enc_ms": lat["audio_enc"],
            "first_token_ms": lat["llm_first_token"]}


def main():
    df = pd.read_csv(CFG.manifest_dir / "test.csv", keep_default_na=False)
    clips = df.sample(n=5, random_state=1)
    pipe = Pipeline(load_asr=False, load_llm=True)
    pipe.warmup()
    for row in clips.itertuples():  # warm every clip's shapes once
        turn(pipe, row)

    runs = []
    for gap in GAPS_S:
        for cond in ("cold", "ping"):
            for rep in range(REPEATS):
                row = clips.iloc[(rep + gap) % len(clips)]
                if cond == "ping" and gap >= PING_LEAD_S:
                    time.sleep(gap - PING_LEAD_S)
                    pipe.ping()
                    time.sleep(PING_LEAD_S)
                else:
                    time.sleep(gap)
                r = {"gap_s": gap, "condition": cond, **turn(pipe, row)}
                runs.append(r)
                print(f"gap {gap:>2}s {cond:4s}: state {r['state_ms']:6.0f} ms "
                      f"(audio enc {r['audio_enc_ms']:5.0f}), first token {r['first_token_ms']:6.0f} ms", flush=True)

    out = pd.DataFrame(runs)
    summary = (out.groupby(["gap_s", "condition"])[["state_ms", "audio_enc_ms", "first_token_ms"]]
                  .median().round(0).reset_index())
    write_json(CFG.results_dir / "idle_latency.json", {
        "device": CFG.device, "repeats": REPEATS, "ping_lead_s": PING_LEAD_S,
        "median_by_gap": summary.to_dict(orient="records"), "runs": runs})
    lines = ["| Idle gap (s) | Condition | State (ms) | Audio encoder (ms) | First token (ms) |",
             "| ---: | --- | ---: | ---: | ---: |"]
    for r in summary.itertuples():
        lines.append(f"| {r.gap_s} | {r.condition} | {r.state_ms:.0f} | {r.audio_enc_ms:.0f} | {r.first_token_ms:.0f} |")
    lines += ["", f"Median of {REPEATS} turns per cell, gold transcripts, {CFG.device}. `ping`: "
              f"Pipeline.ping() {PING_LEAD_S:.0f} s before the turn (at gap 0 there is no ping)."]
    (CFG.results_dir / "idle_latency.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
