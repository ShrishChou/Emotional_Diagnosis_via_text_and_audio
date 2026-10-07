"""Train the fused head on Whisper transcripts, so training matches live use.
-> evals/asr/asr_matched.{json,md}

The deployed heads were trained on gold MELD transcripts, but in the live app they
read Whisper's. This trains the fused head on:

  gold      gold transcripts (the deployed setup)
  asr       Whisper transcripts only
  gold+asr  both copies of every train utterance (2x data, same audio)

3 seeds each, early stopping on the matching dev set (gold dev for `gold`, Whisper dev
for the others), then tests on both gold and Whisper test transcripts.

Needs the Whisper transcripts and features of every split:
  python evaluation/02_asr_transcripts.py --splits train dev test   (~30 min, cached)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from config import CFG, write_json
from src.training import load_split, metrics, predict_logits, summarize, train_head

TEXT_KEY = json.loads((CFG.models_dir / "meta.json").read_text())["text_key"]


def whisper_version(d: dict, split: str) -> dict:
    """The same split with the text feature computed from Whisper transcripts."""
    path = CFG.cache_dir / f"{split}_{TEXT_KEY}_asr.npy"
    if not path.exists():
        sys.exit(f"missing {path.name}: run python evaluation/02_asr_transcripts.py --splits train dev test")
    return {**d, TEXT_KEY: torch.from_numpy(np.load(path))}


def stack(a: dict, b: dict) -> dict:
    return {k: (torch.cat([a[k], b[k]]) if isinstance(a[k], torch.Tensor) else a[k]) for k in a}


def fmt(s: dict) -> str:
    return f"{s['mean']:.3f} ± {s['std']:.3f}"


def main():
    torch.set_num_threads(8)
    gold = {s: load_split(s) for s in ("train", "dev", "test")}
    whisper = {s: whisper_version(d, s) for s, d in gold.items()}
    setups = {"gold": (gold["train"], gold["dev"]),
              "asr": (whisper["train"], whisper["dev"]),
              "gold+asr": (stack(gold["train"], whisper["train"]), whisper["dev"])}

    results = {}
    for name, (train, dev) in setups.items():
        dev_f1, on_gold, on_whisper = [], [], []
        for seed in CFG.seeds:
            head, f1, _ = train_head(train, dev, TEXT_KEY, True, seed)
            dev_f1.append(f1)
            on_gold.append(metrics(gold["test"]["y"], predict_logits(head, gold["test"], TEXT_KEY, True).argmax(1)))
            on_whisper.append(metrics(whisper["test"]["y"],
                                      predict_logits(head, whisper["test"], TEXT_KEY, True).argmax(1)))
        results[name] = {"dev_weighted_f1": {"mean": float(np.mean(dev_f1)), "std": float(np.std(dev_f1))},
                         "test_gold_transcripts": summarize(on_gold), "test_whisper_transcripts": summarize(on_whisper)}
        r = results[name]
        print(f"{name:9s} test weighted F1: gold {fmt(r['test_gold_transcripts']['weighted_f1'])}, "
              f"Whisper {fmt(r['test_whisper_transcripts']['weighted_f1'])}", flush=True)

    write_json(CFG.out("asr", "asr_matched.json"),
               {"text_key": TEXT_KEY, "seeds": list(CFG.seeds), "train_on": results})
    L = ["# Training on Whisper transcripts", "",
         "The fused head trained on gold, Whisper, or both transcripts of MELD train; mean ± std over 3 seeds. "
         "From `evaluation/08_asr_matched_training.py`.", "",
         "| Fused head trained on | Dev weighted F1 (selection set) | Test weighted F1, gold transcripts | "
         "Test weighted F1, Whisper transcripts | Test macro F1, Whisper |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, r in results.items():
        L.append(f"| {name} | {fmt(r['dev_weighted_f1'])} | {fmt(r['test_gold_transcripts']['weighted_f1'])} | "
                 f"{fmt(r['test_whisper_transcripts']['weighted_f1'])} | "
                 f"{fmt(r['test_whisper_transcripts']['macro_f1'])} |")
    L += ["", "`gold` selects on gold dev; `asr` and `gold+asr` select on Whisper dev, so their dev column is "
          "not comparable with `gold`'s."]
    CFG.out("asr", "asr_matched.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
