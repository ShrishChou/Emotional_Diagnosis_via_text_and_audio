"""Train the heads on Whisper transcripts, so training matches deployment.
-> results/asr_matched.{json,md}

The deployed heads were trained on gold MELD transcripts but, without a supplied
transcript, see Whisper output at run time, which differs even when the words are
right (punctuation, casing, fillers). This script transcribes train and dev with
whisper-small (cached in cache/asr_{split}.csv), encodes them like 02 does
(text_ctx, context from the previous two ASR lines), and trains the fused head on:

  gold      gold transcripts (the current model)
  asr       Whisper transcripts only
  gold+asr  both copies of every train utterance (2x data, same audio)

Each 3 seeds, early stopping on the matching dev set (gold dev for `gold`, ASR dev
for the others), then test on both gold and Whisper test transcripts.
"""
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import importlib.util
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

from config import CFG, write_json
from src.encoders import ASR, TextEncoder


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


te = load_script("03_train_eval")
asr_eval = load_script("07_asr_eval")
TEXT_KEY = "text_ctx"


def transcripts(split: str) -> pd.DataFrame:
    """Whisper transcripts aligned with cache/{split}_ids.csv (cached)."""
    path = (CFG.results_dir / "asr_transcripts_test.csv" if split == "test"
            else CFG.cache_dir / f"asr_{split}.csv")
    ids = pd.read_csv(CFG.cache_dir / f"{split}_ids.csv")
    man = pd.read_csv(CFG.manifest_dir / f"{split}.csv", keep_default_na=False)
    df = ids.merge(man, on=["dialogue_id", "utterance_id"], suffixes=("", "_m"))
    if not path.exists():
        asr = ASR()
        asr_text = asr_eval.transcribe_all(asr, [CFG.root / p for p in df["wav_path"]])
        del asr
        df.assign(asr_text=asr_text)[["dialogue_id", "utterance_id", "text", "asr_text"]].to_csv(path, index=False)
    saved = pd.read_csv(path, keep_default_na=False)
    df = df.merge(saved[["dialogue_id", "utterance_id", "asr_text"]], on=["dialogue_id", "utterance_id"])
    assert len(df) == len(ids)
    return df


def asr_features(split: str, enc: TextEncoder) -> torch.Tensor:
    path = CFG.cache_dir / f"{split}_{TEXT_KEY}_asr.npy"
    if not path.exists():
        df = transcripts(split)
        ctx = asr_eval.asr_context(df)
        texts = df["asr_text"].tolist()
        vec = torch.cat([enc.encode(texts[i:i + 64], ctx[i:i + 64]) for i in range(0, len(texts), 64)])
        np.save(path, vec.numpy())
    return torch.from_numpy(np.load(path))


def with_text(d: dict, vec: torch.Tensor) -> dict:
    return {**d, TEXT_KEY: vec}


def concat(a: dict, b: dict) -> dict:
    return {k: (torch.cat([a[k], b[k]]) if isinstance(a[k], torch.Tensor) else a[k]) for k in a}


def main():
    gold = {s: te.load_split(s) for s in ("train", "dev", "test")}
    enc = TextEncoder()
    asr_vec = {s: asr_features(s, enc) for s in ("train", "dev", "test")}
    del enc
    asr = {s: with_text(gold[s], asr_vec[s]) for s in gold}

    setups = {"gold": (gold["train"], gold["dev"]),
              "asr": (asr["train"], asr["dev"]),
              "gold+asr": (concat(gold["train"], asr["train"]), asr["dev"])}
    torch.set_num_threads(8)
    results = {}
    for name, (train, dev) in setups.items():
        runs = {"gold_test": [], "asr_test": [], "dev": []}
        for seed in CFG.seeds:
            head, dev_f1, _ = te.train_head(train, dev, TEXT_KEY, True, seed)
            runs["dev"].append(dev_f1)
            for test_name, test in (("gold_test", gold["test"]), ("asr_test", asr["test"])):
                pred = te.predict_logits(head, test, TEXT_KEY, True).argmax(1)
                runs[test_name].append(te.metrics(test["y"], pred))
        results[name] = {"dev_weighted_f1": {"mean": float(np.mean(runs["dev"])), "std": float(np.std(runs["dev"]))},
                         "test_gold_transcripts": te.summarize(runs["gold_test"]),
                         "test_whisper_transcripts": te.summarize(runs["asr_test"])}
        r = results[name]
        print(f"{name}: test wF1 gold {te.fmt(r['test_gold_transcripts']['weighted_f1'])}, "
              f"whisper {te.fmt(r['test_whisper_transcripts']['weighted_f1'])}", flush=True)

    write_json(CFG.results_dir / "asr_matched.json", {"text_key": TEXT_KEY, "seeds": list(CFG.seeds),
                                                      "train_on": results})
    lines = ["| Fused head trained on | Dev weighted F1 (selection set) | Test wF1, gold transcripts | "
             "Test wF1, Whisper transcripts | Test macro F1, Whisper |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, r in results.items():
        lines.append(f"| {name} | {te.fmt(r['dev_weighted_f1'])} | {te.fmt(r['test_gold_transcripts']['weighted_f1'])} | "
                     f"{te.fmt(r['test_whisper_transcripts']['weighted_f1'])} | "
                     f"{te.fmt(r['test_whisper_transcripts']['macro_f1'])} |")
    lines += ["", "Mean ± std over 3 seeds. `gold` selects on gold dev; `asr` and `gold+asr` on Whisper dev."]
    (CFG.results_dir / "asr_matched.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
