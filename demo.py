"""One utterance in -> JSON emotion state + streamed emotion-aware reply out.

  python demo.py --meld test dia0_utt3            # MELD clip, gold transcript
  python demo.py --meld test dia0_utt3 --asr      # MELD clip, Whisper transcript
  python demo.py --wav clip.wav                   # transcript from Whisper
  python demo.py --wav clip.wav --text "I'm fine" # user-supplied transcript
"""
import os

# Inference is local only: refuse any Hugging Face network access. Weights must
# already be cached (training/01_check_environment.py downloads them).
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")  # hide model load reports
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import argparse
import json
import re
import sys
import time
import warnings

import pandas as pd

from config import CFG

warnings.filterwarnings("ignore")


def meld_lookup(split: str, clip: str) -> dict:
    m = re.fullmatch(r"dia(\d+)_utt(\d+)", clip)
    assert m, "clip id must look like dia12_utt3"
    df = pd.read_csv(CFG.manifest_dir / f"{split}.csv", keep_default_na=False)
    row = df[(df.dialogue_id == int(m[1])) & (df.utterance_id == int(m[2]))]
    assert len(row) == 1, f"{split}/{clip} not in the manifest"
    return row.iloc[0].to_dict()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--meld", nargs=2, metavar=("SPLIT", "CLIP"), help="e.g. test dia0_utt3")
    src.add_argument("--wav", help="path to a wav/flac file")
    ap.add_argument("--text", help="transcript (skips ASR)")
    ap.add_argument("--context", default="", help="previous lines, 'Speaker: text' separated by newlines")
    ap.add_argument("--asr", action="store_true", help="with --meld: ignore the gold transcript, use Whisper")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-warmup", action="store_true", help="skip the untimed warm-up pass")
    args = ap.parse_args()

    if args.meld:
        row = meld_lookup(*args.meld)
        wav, context, uid = row["wav_path"], row["context"], args.meld[1]
        text, source = (None, None) if args.asr else (row["text"], "gold")
        gold = row["emotion"]
    else:
        wav, context, uid = args.wav, args.context.replace("\\n", "\n"), None
        text, source, gold = args.text, ("user" if args.text else None), None

    from src.pipeline import Pipeline
    print(f"== loading models on {CFG.device} ==")
    t0 = time.perf_counter()
    pipe = Pipeline(load_asr=text is None or args.asr)
    print(f"cold load: {time.perf_counter() - t0:.1f} s  {json.dumps(pipe.cold_load_s)}")
    if not args.no_warmup:
        print(f"warm-up pass (not timed below): {pipe.warmup():.1f} s")

    print("\n== input ==")
    print(f"audio: {wav}")
    if context:
        print("context:\n  " + context.replace("\n", "\n  "))
    if gold:
        print(f"gold label (MELD annotation, for reference only): {gold}")

    state = pipe.analyze(wav, text=text, context=context, utterance_id=uid, transcript_source=source)
    print("\n== JSON state (emitted before the reply) ==")
    print(json.dumps(state, indent=2))

    print("\n== reply (streamed) ==")
    for chunk in pipe.respond(state, context, seed=args.seed):
        sys.stdout.write(chunk)
        sys.stdout.flush()
    print("\n\n== latency (ms) ==")
    print(json.dumps(state["latency_ms"], indent=2))


if __name__ == "__main__":
    main()
