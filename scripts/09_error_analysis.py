"""Per-example evaluation of the saved models on MELD test, and where they fail.

Uses the deployed checkpoints (text-only and fused, one seed each, temperature-scaled
fused probabilities) on every test utterance, with gold transcripts and with the
Whisper transcripts from 07_asr_eval.py.

Outputs:
  results/test_predictions.csv  one row per test utterance: gold, both models'
                                predictions and confidences, correctness, slices
  results/error_breakdown.md    per-class table, accuracy by slice, top confusions
  results/error_breakdown.json  the same numbers
"""
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
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
from sklearn.metrics import f1_score, precision_recall_fscore_support

from config import CFG, LABELS, write_json
from src.fusion import load_head

MAIN_SPEAKERS = ["Chandler", "Joey", "Monica", "Phoebe", "Rachel", "Ross"]


def words(s: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]", " ", s.lower()).split()


def edit_distance(a, b) -> int:
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[-1]


def asr_features(df: pd.DataFrame, text_key: str) -> torch.Tensor:
    """RoBERTa features of the Whisper transcripts (cached after the first run)."""
    path = CFG.cache_dir / f"test_{text_key}_asr.npy"
    if path.exists():
        return torch.from_numpy(np.load(path))
    from src.encoders import TextEncoder
    enc = TextEncoder()
    ctx = None
    if text_key == "text_ctx":  # previous two ASR lines, as in 07_asr_eval.py
        ctx_map = {}
        for _, dia in df.groupby("dialogue_id", sort=False):
            dia = dia.sort_values("utterance_id")
            lines = [f"{s}: {t}" for s, t in zip(dia["speaker"], dia["asr_text"])]
            for k, i in enumerate(dia.index):
                ctx_map[i] = "\n".join(lines[max(0, k - 2):k])
        ctx = [ctx_map[i] for i in df.index]
    texts = df["asr_text"].tolist()
    vec = torch.cat([enc.encode(texts[i:i + 64], ctx[i:i + 64] if ctx else None)
                     for i in range(0, len(texts), 64)])
    np.save(path, vec.numpy())
    return vec


def per_class(y, pred) -> pd.DataFrame:
    p, r, f, n = precision_recall_fscore_support(y, pred, labels=range(len(LABELS)), zero_division=0)
    return pd.DataFrame({"support": n, "recall (= class accuracy)": r, "precision": p, "f1": f}, index=LABELS)


def slice_table(df: pd.DataFrame, col: str, order=None) -> pd.DataFrame:
    g = df.groupby(col, observed=True)
    out = pd.DataFrame({
        "n": g.size(),
        "share": g.size() / len(df),
        "text_acc": g["text_correct"].mean(),
        "fused_acc": g["fused_correct"].mean(),
        "fused_asr_acc": g["fused_asr_correct"].mean(),
        "fused_wF1": g.apply(lambda s: f1_score(s["gold_id"], s["fused_id"], average="weighted",
                                                labels=range(len(LABELS)), zero_division=0)),
    })
    return out.reindex(order) if order is not None else out


def md(df: pd.DataFrame, pct_cols=(), int_cols=("n", "support")) -> str:
    cols = [df.index.name or ""] + list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] + ["---:"] * len(df.columns)) + " |"]
    for idx, row in df.iterrows():
        cells = [str(idx)]
        for c in df.columns:
            v = row[c]
            if c in int_cols:
                cells.append(f"{int(v):,d}")
            elif c in pct_cols:
                cells.append(f"{v:.0%}")
            else:
                cells.append(f"{v:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ids = pd.read_csv(CFG.cache_dir / "test_ids.csv")
    man = pd.read_csv(CFG.manifest_dir / "test.csv", keep_default_na=False)
    asr = pd.read_csv(CFG.results_dir / "asr_transcripts_test.csv", keep_default_na=False)
    df = (ids.merge(man, on=["dialogue_id", "utterance_id"], suffixes=("", "_m"))
             .merge(asr[["dialogue_id", "utterance_id", "asr_text"]], on=["dialogue_id", "utterance_id"]))
    assert len(df) == len(ids)

    text_head, tck = load_head(CFG.ckpt_dir / "text_head.pt")
    fused_head, fck = load_head(CFG.ckpt_dir / "fused_head.pt")
    key, T = fck["text_key"], fck["temperature"]
    gold_vec = torch.from_numpy(np.load(CFG.cache_dir / f"test_{key}.npy"))
    audio = torch.from_numpy(np.load(CFG.cache_dir / "test_audio.npy").astype(np.float32))
    asr_vec = asr_features(df, key)

    with torch.no_grad():
        p_text = F.softmax(text_head(text=gold_vec), 1).numpy()
        p_fused = F.softmax(fused_head(text=gold_vec, audio=audio) / T, 1).numpy()
        p_fused_asr = F.softmax(fused_head(text=asr_vec, audio=audio) / T, 1).numpy()
        p_text_asr = F.softmax(text_head(text=asr_vec), 1).numpy()

    y = df["label_id"].to_numpy()
    df["clip"] = [f"dia{d}_utt{u}" for d, u in zip(df.dialogue_id, df.utterance_id)]
    df["gold"] = df["emotion"]
    df["gold_id"] = y
    for name, p in [("text", p_text), ("fused", p_fused), ("fused_asr", p_fused_asr), ("text_asr", p_text_asr)]:
        df[f"{name}_id"] = p.argmax(1)
        df[f"{name}_pred"] = [LABELS[i] for i in p.argmax(1)]
        df[f"{name}_conf"] = p.max(1).round(4)
        df[f"{name}_correct"] = df[f"{name}_id"] == y
    df["fused_p_gold"] = p_fused[np.arange(len(y)), y].round(4)
    df["audio_changed"] = df["text_id"] != df["fused_id"]
    df["n_words"] = df["text"].map(lambda t: len(words(t)))
    ref, hyp = df["text"].map(words), df["asr_text"].map(words)
    df["wer"] = [edit_distance(r, h) / max(len(r), 1) for r, h in zip(ref, hyp)]

    # Slices
    df["confidence"] = pd.cut(df["fused_conf"], [0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                              labels=["<0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"])
    df["utterance_words"] = pd.cut(df["n_words"], [-1, 2, 5, 10, 20, 1000],
                                   labels=["1-2", "3-5", "6-10", "11-20", "21+"])
    df["clip_seconds"] = pd.cut(df["duration_s"], [0, 1, 2, 4, 8, 1e6], labels=["<1", "1-2", "2-4", "4-8", "8+"])
    df["has_context"] = np.where(df["context"] == "", "first line of dialogue", "has previous lines")
    df["speaker_group"] = np.where(df["speaker"].isin(MAIN_SPEAKERS), df["speaker"], "other")
    df["asr_wer"] = pd.cut(df["wer"], [-0.01, 0, 0.25, 0.5, 1e6], labels=["0 (exact)", "0-25%", "25-50%", ">50%"])
    df["audio_flip"] = np.where(df["audio_changed"], "audio changed prediction", "same as text-only")

    overall = {}
    for name in ("text", "fused", "text_asr", "fused_asr"):
        pred = df[f"{name}_id"]
        overall[name] = {"accuracy": float((pred == y).mean()),
                         "weighted_f1": float(f1_score(y, pred, average="weighted")),
                         "macro_f1": float(f1_score(y, pred, average="macro"))}

    pc_text, pc_fused, pc_asr = per_class(y, df.text_id), per_class(y, df.fused_id), per_class(y, df.fused_asr_id)
    classes = pd.DataFrame({
        "support": pc_fused["support"],
        "text recall": pc_text["recall (= class accuracy)"],
        "fused recall": pc_fused["recall (= class accuracy)"],
        "fused precision": pc_fused["precision"],
        "fused F1": pc_fused["f1"],
        "fused recall (ASR)": pc_asr["recall (= class accuracy)"],
        "→ neutral": [float(((y == i) & (df.fused_id == LABELS.index("neutral"))).sum() / max((y == i).sum(), 1))
                      for i in range(len(LABELS))],
    }, index=pd.Index(LABELS, name="gold class"))
    classes.loc["neutral", "→ neutral"] = np.nan

    errs = df[~df.fused_correct]
    confusions = (errs.groupby(["gold", "fused_pred"]).size().sort_values(ascending=False).head(10)
                    .rename("count").reset_index())
    confusions["share of all errors"] = confusions["count"] / len(errs)

    slices = {}
    for col, title in [("confidence", "Fused confidence (temperature-scaled)"),
                       ("utterance_words", "Utterance length (words)"),
                       ("clip_seconds", "Clip duration (s)"),
                       ("has_context", "Dialogue context"),
                       ("speaker_group", "Speaker"),
                       ("asr_wer", "Whisper word error rate on the clip"),
                       ("audio_flip", "Did audio change the text-only prediction?")]:
        t = slice_table(df, col)
        t.index.name = title
        slices[col] = t

    # Markdown report
    o = overall
    out = [
        "# Error breakdown: deployed models on MELD test", "",
        f"{len(df):,d} test utterances. Saved checkpoints: text-only (`{key}`, seed {tck['seed']}) and fused "
        f"(`{key}` + audio, seed {fck['seed']}, T = {T:.2f}). \"ASR\" = the same fused model fed Whisper-small "
        "transcripts instead of gold. Per-utterance rows: `results/test_predictions.csv`.", "",
        "| Model | Accuracy | Weighted F1 | Macro F1 |", "| --- | ---: | ---: | ---: |",
        *[f"| {n} | {v['accuracy']:.3f} | {v['weighted_f1']:.3f} | {v['macro_f1']:.3f} |"
          for n, v in [("text-only, gold transcript", o["text"]), ("fused, gold transcript", o["fused"]),
                       ("text-only, Whisper transcript", o["text_asr"]), ("fused, Whisper transcript", o["fused_asr"])]],
        "", "## Per class", "",
        "Recall is the accuracy on utterances of that gold class. \"→ neutral\" is the share of that class the "
        "fused model calls neutral.", "",
        md(classes, pct_cols=("→ neutral",)).replace("nan%", "–"), "",
        "## Most common errors (fused, gold transcripts)", "",
        "| Gold | Predicted | Count | Share of all errors |", "| --- | --- | ---: | ---: |",
        *[f"| {r.gold} | {r.fused_pred} | {r['count']} | {r['share of all errors']:.0%} |" for _, r in confusions.iterrows()],
        "", "## Accuracy by slice", "",
        "`text_acc` / `fused_acc`: accuracy with gold transcripts; `fused_asr_acc`: fused with Whisper transcripts; "
        "`fused_wF1`: weighted F1 of the fused model within the slice.", "",
    ]
    for col, t in slices.items():
        out += [md(t, pct_cols=("share",)), ""]
    (CFG.results_dir / "error_breakdown.md").write_text("\n".join(out))

    keep = ["clip", "speaker", "text", "asr_text", "wer", "duration_s", "n_words", "has_context", "gold",
            "text_pred", "text_conf", "text_correct", "fused_pred", "fused_conf", "fused_correct",
            "fused_p_gold", "audio_changed", "fused_asr_pred", "fused_asr_conf", "fused_asr_correct"]
    df[keep].to_csv(CFG.results_dir / "test_predictions.csv", index=False)
    write_json(CFG.results_dir / "error_breakdown.json", {
        "overall": overall,
        "per_class": classes.reset_index().to_dict(orient="records"),
        "top_confusions": confusions.to_dict(orient="records"),
        "slices": {k: v.reset_index().astype({v.index.name: str}).to_dict(orient="records")
                   for k, v in slices.items()},
    })
    print((CFG.results_dir / "error_breakdown.md").read_text())


if __name__ == "__main__":
    main()
