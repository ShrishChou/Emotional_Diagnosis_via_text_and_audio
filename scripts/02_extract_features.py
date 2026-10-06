"""Run the frozen encoders once over every split and cache the embeddings.

Writes, aligned row for row:
  cache/{split}_text.npy      [N, 768]      roberta-base, utterance alone
  cache/{split}_text_ctx.npy  [N, 768]      roberta-base, previous 2 lines + utterance
  cache/{split}_audio.npy     [N, 13, 768]  wavlm-base-plus, every layer mean-pooled (float16)
  cache/{split}_ids.csv       dialogue_id, utterance_id, label_id
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import CFG, set_seed
from src.audio import load_wav
from src.encoders import AudioEncoder, TextEncoder

SPLITS = ["train", "dev", "test"]
TEXT_BATCH = 64
AUDIO_MAX_CLIPS = 32
AUDIO_MAX_PADDED_S = 160.0  # batch size x longest clip, in seconds of audio


def load_manifest(split: str) -> pd.DataFrame:
    df = pd.read_csv(CFG.manifest_dir / f"{split}.csv", keep_default_na=False)
    if split == "train" and CFG.audio_train_subsample:
        # CPU-only fallback: stratified subsample of train (applies to every modality,
        # so all feature files stay aligned).
        frac = CFG.audio_train_subsample / len(df)
        df = df.groupby("label_id").sample(frac=frac, random_state=0).sort_index()
        print(f"[train] CPU fallback: stratified subsample of {len(df)} rows")
    return df.reset_index(drop=True)


def extract_text(enc: TextEncoder, df: pd.DataFrame, with_context: bool) -> np.ndarray:
    out = []
    name = "text_ctx" if with_context else "text"
    for i in tqdm(range(0, len(df), TEXT_BATCH), desc=name, leave=False):
        chunk = df.iloc[i:i + TEXT_BATCH]
        ctx = chunk["context"].tolist() if with_context else None
        out.append(enc.encode(chunk["text"].tolist(), ctx))
    return torch.cat(out).numpy().astype(np.float32)


def audio_batches(durations: np.ndarray):
    """Indices sorted by duration, grouped so each padded batch stays under budget."""
    order = np.argsort(durations)
    batch = []
    for i in order:
        longest = min(durations[i], CFG.max_audio_s)
        if batch and (len(batch) + 1) * longest > AUDIO_MAX_PADDED_S or len(batch) == AUDIO_MAX_CLIPS:
            yield batch
            batch = []
        batch.append(i)
    if batch:
        yield batch


def extract_audio(enc: AudioEncoder, df: pd.DataFrame) -> tuple[np.ndarray, int]:
    feats = np.zeros((len(df), 13, 768), dtype=np.float16)
    n_cropped = 0
    batches = list(audio_batches(df["duration_s"].to_numpy()))
    for idx in tqdm(batches, desc="audio", leave=False):
        wavs = []
        for i in idx:
            w, cropped = load_wav(CFG.root / df.loc[i, "wav_path"])
            wavs.append(w)
            n_cropped += cropped
        feats[idx] = enc.encode(wavs).numpy().astype(np.float16)
    return feats, n_cropped


def check(name: str, arr: np.ndarray, n: int) -> None:
    assert len(arr) == n, f"{name}: {len(arr)} rows, expected {n}"
    assert np.isfinite(arr.astype(np.float32)).all(), f"{name}: NaN or inf found"
    print(f"  {name}: shape {arr.shape}, first-row norm {np.linalg.norm(arr[0].astype(np.float32)):.2f}")


def main():
    # `python 02_extract_features.py text_ctx` re-extracts just that feature.
    only = set(sys.argv[1:]) or {"text", "text_ctx", "audio"}
    set_seed(0)
    CFG.cache_dir.mkdir(parents=True, exist_ok=True)
    text_enc = TextEncoder() if only & {"text", "text_ctx"} else None
    audio_enc = AudioEncoder() if "audio" in only else None
    for split in SPLITS:
        df = load_manifest(split)
        t0 = time.time()
        feats = {}
        if "text" in only:
            feats["text"] = extract_text(text_enc, df, with_context=False)
        if "text_ctx" in only:
            feats["text_ctx"] = extract_text(text_enc, df, with_context=True)
        if "audio" in only:
            feats["audio"], n_cropped = extract_audio(audio_enc, df)
            print(f"[{split}] {n_cropped} clips cropped to {CFG.max_audio_s:.0f}s")
        print(f"[{split}] {len(df)} rows in {time.time() - t0:.0f}s")
        for name, arr in feats.items():
            check(name, arr, len(df))
            np.save(CFG.cache_dir / f"{split}_{name}.npy", arr)
        df[["dialogue_id", "utterance_id", "label_id"]].to_csv(CFG.cache_dir / f"{split}_ids.csv", index=False)


if __name__ == "__main__":
    main()
