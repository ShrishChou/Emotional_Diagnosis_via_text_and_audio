"""Optional: how much do Whisper transcripts (instead of gold) cost? -> results/asr_eval.json

Transcribes every MELD test clip with whisper-small, re-encodes the text with
RoBERTa, and scores the saved text-only and fused heads on ASR transcripts.
If the heads use dialogue context, the context lines are built from the ASR
transcripts of the previous two utterances (speaker names from MELD, since
there is no diarization). Also reports corpus word error rate (WER) after
lowercasing and stripping punctuation.
"""
import os

# Inference is local only: refuse any Hugging Face network access. Weights must
# already be cached (scripts/00_check_env.py downloads them).
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")  # hide model load reports
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import re
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from tqdm import tqdm

from config import CFG, write_json
from src.audio import emode_normalize, read_raw
from src.encoders import ASR, TextEncoder
from src.fusion import load_head

ASR_BATCH = 16


def words(s: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]", " ", s.lower()).split()


def edit_distance(a: list[str], b: list[str]) -> int:
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[-1]


@torch.no_grad()
def transcribe_all(asr: ASR, paths: list[Path]) -> list[str]:
    out = []
    for i in tqdm(range(0, len(paths), ASR_BATCH), desc="whisper"):
        raws = [read_raw(p)[0] for p in paths[i:i + ASR_BATCH]]
        feats = asr.proc(raws, sampling_rate=CFG.sample_rate, return_tensors="pt").input_features
        ids = asr.model.generate(feats.to(asr.device, asr.dtype), language="en", task="transcribe")
        out += [t.strip() for t in asr.proc.batch_decode(ids, skip_special_tokens=True)]
    return out


def asr_context(df: pd.DataFrame) -> list[str]:
    ctx = {}
    for _, dia in df.groupby("dialogue_id", sort=False):
        dia = dia.sort_values("utterance_id")
        lines = [f"{s}: {t}" for s, t in zip(dia["speaker"], dia["asr_text"])]
        for k, i in enumerate(dia.index):
            ctx[i] = "\n".join(lines[max(0, k - 2):k])
    return [ctx[i] for i in df.index]


def main():
    ids = pd.read_csv(CFG.cache_dir / "test_ids.csv")
    man = pd.read_csv(CFG.manifest_dir / "test.csv", keep_default_na=False)
    df = ids.merge(man, on=["dialogue_id", "utterance_id"], how="left", suffixes=("", "_m"))
    assert len(df) == len(ids)

    # Whisper over all of test takes ~8 min on the M5, so transcripts are saved
    # first and reused on a rerun.
    trans_path = CFG.results_dir / "asr_transcripts_test.csv"
    if trans_path.exists():
        saved = pd.read_csv(trans_path, keep_default_na=False)
        df = df.merge(saved[["dialogue_id", "utterance_id", "asr_text"]], on=["dialogue_id", "utterance_id"])
        print(f"reusing {trans_path.name}")
    else:
        asr = ASR()
        df["asr_text"] = transcribe_all(asr, [CFG.root / p for p in df["wav_path"]])
        del asr
        df[["dialogue_id", "utterance_id", "text", "asr_text"]].to_csv(trans_path, index=False)
    assert len(df) == len(ids)

    ref = [words(t) for t in df["text"]]
    hyp = [words(t) for t in df["asr_text"]]
    wer = sum(edit_distance(r, h) for r, h in zip(ref, hyp)) / sum(len(r) for r in ref)

    text_head, ckpt = load_head(CFG.ckpt_dir / "text_head.pt")
    fused_head, fckpt = load_head(CFG.ckpt_dir / "fused_head.pt")
    text_key = ckpt["text_key"]
    enc = TextEncoder()
    texts = df["asr_text"].tolist()
    ctxs = asr_context(df) if text_key == "text_ctx" else None
    vecs = []
    for i in range(0, len(texts), 64):
        vecs.append(enc.encode(texts[i:i + 64], ctxs[i:i + 64] if ctxs else None))
    asr_vec = torch.cat(vecs)

    gold_vec = torch.from_numpy(np.load(CFG.cache_dir / f"test_{text_key}.npy"))
    audio = torch.from_numpy(np.load(CFG.cache_dir / "test_audio.npy").astype(np.float32))
    y = ids["label_id"].to_numpy()

    def score(head, text_vec, use_audio):
        with torch.no_grad():
            logits = head(text=text_vec, audio=audio if use_audio else None)
        return float(f1_score(y, logits.argmax(1).numpy(), average="weighted"))

    out = {
        "n_clips": len(df),
        "asr_model": CFG.asr_model,
        "wer": round(wer, 4),
        "text_variant": text_key,
        "context_from": "ASR transcripts of the previous two utterances" if ctxs else "not used",
        "weighted_f1": {
            "text_only_gold": score(text_head, gold_vec, False),
            "text_only_asr": score(text_head, asr_vec, False),
            "fused_gold": score(fused_head, gold_vec, True),
            "fused_asr": score(fused_head, asr_vec, True),
        },
        "note": "Saved checkpoints (one seed each), so gold numbers match "
                "metrics.json -> saved_checkpoints, not the 3-seed means.",
    }
    write_json(CFG.results_dir / "asr_eval.json", out)
    print(out)


if __name__ == "__main__":
    main()
