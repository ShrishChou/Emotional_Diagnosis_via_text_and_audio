"""Whisper transcripts instead of gold: what do they cost? -> evals/asr/asr_eval.json

The results table uses MELD's gold transcripts, but a live robot hears Whisper's.
This transcribes MELD clips with whisper-small, encodes the transcripts with RoBERTa
exactly like the gold ones (context = the previous two *Whisper* lines, speaker names
from MELD since there is no diarization), and scores the deployed heads on test.

  python evaluation/02_asr_transcripts.py                        # test only (~6 min)
  python evaluation/02_asr_transcripts.py --splits train dev test  # also what 08 trains on (~30 min)

Transcripts and features are cached, so reruns are fast:
  evals/asr/asr_transcripts_test.csv, cache/asr_{train,dev}.csv, cache/{split}_<text feature>_asr.npy
Also reports the corpus word error rate (lowercased, punctuation removed).
"""
import os

os.environ["HF_HUB_OFFLINE"] = "1"   # inference is local only; weights are already cached
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm

from config import CFG, write_json
from src.audio import read_raw
from src.data import dialogue_context, edit_distance, load_manifest, words
from src.encoders import ASR, TextEncoder
from src.fusion import load_deployed

ASR_BATCH = 16


def transcript_path(split: str) -> Path:
    # Test transcripts are part of the evidence (committed); train/dev ones are a cache.
    return CFG.out("asr", "asr_transcripts_test.csv") if split == "test" else CFG.cache_dir / f"asr_{split}.csv"


def transcripts(split: str, asr: ASR | None) -> pd.DataFrame:
    """The split's rows (in cache/{split}_ids.csv order) with an `asr_text` column."""
    ids = pd.read_csv(CFG.cache_dir / f"{split}_ids.csv")
    df = ids.merge(load_manifest(split), on=["dialogue_id", "utterance_id"], how="left", suffixes=("", "_m"))
    path = transcript_path(split)
    if not path.exists():
        asr = asr or ASR()
        paths = [CFG.root / p for p in df["wav_path"]]
        text = []
        for i in tqdm(range(0, len(paths), ASR_BATCH), desc=f"whisper {split}"):
            text += asr.transcribe_batch([read_raw(p)[0] for p in paths[i:i + ASR_BATCH]])
        df.assign(asr_text=text)[["dialogue_id", "utterance_id", "text", "asr_text"]].to_csv(path, index=False)
    saved = pd.read_csv(path, keep_default_na=False)[["dialogue_id", "utterance_id", "asr_text"]]
    df = df.merge(saved, on=["dialogue_id", "utterance_id"])
    assert len(df) == len(ids), f"{path} does not cover every {split} row"
    return df


def asr_features(split: str, text_key: str, enc: TextEncoder | None = None) -> torch.Tensor:
    """RoBERTa features of the Whisper transcripts, built like the gold ones (cached)."""
    path = CFG.cache_dir / f"{split}_{text_key}_asr.npy"
    if not path.exists():
        df = transcripts(split, None)
        enc = enc or TextEncoder()
        texts = df["asr_text"].tolist()
        ctx = dialogue_context(df, "asr_text") if text_key == "text_ctx" else None
        vec = torch.cat([enc.encode(texts[i:i + 64], ctx[i:i + 64] if ctx else None)
                         for i in range(0, len(texts), 64)])
        np.save(path, vec.numpy())
    return torch.from_numpy(np.load(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["test"], choices=["train", "dev", "test"])
    args = ap.parse_args()
    text_key = json.loads((CFG.models_dir / "meta.json").read_text())["text_key"]
    for split in args.splits:
        asr_features(split, text_key)
        print(f"{split}: transcripts and features ready")
    if "test" not in args.splits:
        return

    df = transcripts("test", None)
    ref, hyp = df["text"].map(words), df["asr_text"].map(words)
    wer = sum(edit_distance(r, h) for r, h in zip(ref, hyp)) / sum(len(r) for r in ref)

    heads = load_deployed()
    gold_vec = torch.from_numpy(np.load(CFG.cache_dir / f"test_{text_key}.npy"))
    asr_vec = asr_features("test", text_key)
    audio = torch.from_numpy(np.load(CFG.cache_dir / "test_audio.npy").astype(np.float32))
    y = df["label_id"].to_numpy()

    def score(role: str, text_vec: torch.Tensor) -> float:
        head = heads[role][0]
        with torch.no_grad():
            logits = head(text=text_vec, audio=audio if head.use_audio else None)
        return float(f1_score(y, logits.argmax(1).numpy(), average="weighted"))

    out = {
        "n_clips": len(df), "asr_model": CFG.asr_model, "wer": round(wer, 4), "text_variant": text_key,
        "context_from": "Whisper transcripts of the previous two utterances" if text_key == "text_ctx" else "not used",
        "weighted_f1": {"text_only_gold": score("text_only", gold_vec), "text_only_asr": score("text_only", asr_vec),
                        "fused_gold": score("fused", gold_vec), "fused_asr": score("fused", asr_vec)},
        "note": "The deployed heads (one seed each), so the gold numbers match the saved models in "
                "evals/classifiers/metrics.json, not the 3-seed means.",
    }
    write_json(CFG.out("asr", "asr_eval.json"), out)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
