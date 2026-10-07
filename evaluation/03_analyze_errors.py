"""Where does the emotion model struggle? -> evals/struggles/

Runs the two models the app uses (fused for voice, text-only for typed messages) on
every MELD test utterance, with gold and with Whisper transcripts, and breaks the
errors down. Needs evaluation/02_asr_transcripts.py to have run first (test only).

Writes
  analytics.md             the written breakdown, every number computed here
  test_predictions.csv     one row per test utterance: gold, predictions, confidence, slices
  examples.md              utterances where audio fixed or broke the text-only answer
  charts/class_outcomes.png       what happens to each emotion's utterances
  charts/top_confusions.png       the most common mistakes
  charts/accuracy_by_slice.png    accuracy by utterance length, clip length, speaker, Whisper WER
  charts/whisper_by_class.png     recall per emotion, gold vs Whisper transcripts
  charts/audio_effect.png         errors audio fixed vs answers it broke, per emotion
  charts/label_noise.png          train utterances no model gets right (if 10_train_on_failures ran)
"""
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import importlib.util
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support

from config import CFG, LABELS, write_json
from src import plots
from src.data import edit_distance, words
from src.fusion import load_deployed
from src.plots import plt
from src.training import softmax_np

HIGH_AROUSAL = ["anger", "joy", "surprise"]
MAIN_SPEAKERS = ["Chandler", "Joey", "Monica", "Phoebe", "Rachel", "Ross"]
NEUTRAL = LABELS.index("neutral")
OUT = CFG.evals_dir / "struggles"

# Reuse the transcript loader and feature builder of 02_asr_transcripts.py.
_spec = importlib.util.spec_from_file_location("asr_eval", Path(__file__).with_name("02_asr_transcripts.py"))
asr_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(asr_eval)


# ---------------------------------------------------------------- predictions

def predict_all() -> pd.DataFrame:
    """One row per test utterance with both deployed models' outputs, gold and Whisper text."""
    heads = load_deployed()
    meta = heads["meta"]
    key = meta["text_key"]
    df = asr_eval.transcripts("test", None)
    gold = torch.from_numpy(np.load(CFG.cache_dir / f"test_{key}.npy"))
    whisper = asr_eval.asr_features("test", key)
    audio = torch.from_numpy(np.load(CFG.cache_dir / "test_audio.npy").astype(np.float32))
    text_head, text_ckpt = heads["text_only"]
    fused_head, fused_ckpt = heads["fused"]
    with torch.no_grad():
        p = {"text": softmax_np(text_head(text=gold).numpy(), text_ckpt["temperature"]),
             "fused": softmax_np(fused_head(text=gold, audio=audio).numpy(), fused_ckpt["temperature"]),
             "fused_asr": softmax_np(fused_head(text=whisper, audio=audio).numpy(), fused_ckpt["temperature"])}
    y = df["label_id"].to_numpy()
    df["clip"] = [f"dia{d}_utt{u}" for d, u in zip(df.dialogue_id, df.utterance_id)]
    df["gold"] = df["emotion"]
    for name, probs in p.items():
        df[f"{name}_pred"] = [LABELS[i] for i in probs.argmax(1)]
        df[f"{name}_conf"] = probs.max(1).round(4)
        df[f"{name}_correct"] = probs.argmax(1) == y
    df["fused_p_gold"] = p["fused"][np.arange(len(y)), y].round(4)
    df["text_p_gold"] = p["text"][np.arange(len(y)), y].round(4)
    df["audio_changed"] = df["text_pred"] != df["fused_pred"]
    df["n_words"] = df["text"].map(lambda t: len(words(t)))
    df["wer"] = [edit_distance(words(r), words(h)) / max(len(words(r)), 1) for r, h in zip(df.text, df.asr_text)]
    df["utterance_words"] = pd.cut(df.n_words, [-1, 2, 5, 10, 20, 1000], labels=["1-2", "3-5", "6-10", "11-20", "21+"])
    df["clip_seconds"] = pd.cut(df.duration_s, [0, 1, 2, 4, 8, 1e6], labels=["<1 s", "1-2 s", "2-4 s", "4-8 s", "8+ s"])
    df["speaker_group"] = np.where(df.speaker.isin(MAIN_SPEAKERS), df.speaker, "other")
    df["has_context"] = np.where(df.context == "", "first line", "has previous lines")
    df["whisper_wer"] = pd.cut(df.wer, [-0.01, 0, 0.25, 0.5, 1e6], labels=["exact", "1-25%", "25-50%", ">50%"])
    df["confidence"] = pd.cut(df.fused_conf, [0, .4, .5, .6, .7, .8, .9, 1.0],
                              labels=["<0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"])
    return df


def slice_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby(col, observed=True)
    return pd.DataFrame({"n": g.size(), "share": g.size() / len(df),
                         "accuracy, gold text": g.fused_correct.mean(),
                         "accuracy, Whisper text": g.fused_asr_correct.mean(),
                         "text-only accuracy": g.text_correct.mean()})


def md_table(t: pd.DataFrame, first: str) -> str:
    cols = [first] + list(t.columns)
    rows = ["| " + " | ".join(cols) + " |", "| --- |" + " ---: |" * len(t.columns)]
    for idx, r in t.iterrows():
        cells = [str(idx)]
        for c in t.columns:
            v = r[c]
            cells.append(f"{int(v):,d}" if c in ("n", "support") else f"{v:.0%}" if c == "share" else f"{v:.3f}")
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


# ---------------------------------------------------------------- charts

def chart_class_outcomes(df, path) -> pd.DataFrame:
    """100% bars: for each true emotion, the share predicted correctly, as neutral, or as another emotion."""
    rows = []
    for c in LABELS:
        d = df[df.gold == c]
        correct = (d.fused_pred == c).mean()
        neutral = 0.0 if c == "neutral" else (d.fused_pred == "neutral").mean()
        rows.append({"emotion": c, "n": len(d), "correct": correct, "called neutral": neutral,
                     "another emotion": 1 - correct - neutral})
    t = pd.DataFrame(rows).set_index("emotion").sort_values("correct")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    left = np.zeros(len(t))
    for col, color in (("correct", plots.ACCENT), ("called neutral", plots.OTHER), ("another emotion", plots.ACCENT_2)):
        ax.barh(t.index, t[col], left=left, color=color, height=0.62, label=col, edgecolor=plots.SURFACE, linewidth=2)
        for i, (start, v) in enumerate(zip(left, t[col])):
            if v >= 0.07:
                ax.text(start + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=8.5,
                        color="white" if color != plots.OTHER else plots.INK)
        left += t[col].to_numpy()
    for i, n in enumerate(t["n"]):
        ax.text(1.01, i, f"n={n:,d}", va="center", fontsize=8.5, color=plots.INK_2)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_title("What happens to each emotion's test utterances (fused model)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.12), ncol=3)
    plots.save(fig, path)
    return t


def chart_top_confusions(df, path) -> pd.DataFrame:
    errs = df[~df.fused_correct]
    t = (errs.groupby(["gold", "fused_pred"]).size().sort_values(ascending=False).head(10)
             .rename("count").reset_index())
    t["share of errors"] = t["count"] / len(errs)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    labels = [f"{g} → {p}" for g, p in zip(t.gold, t.fused_pred)]
    colors = [plots.ACCENT if "neutral" in (g, p) else plots.OTHER for g, p in zip(t.gold, t.fused_pred)]
    ax.barh(labels, t["count"], color=colors, height=0.62)
    for i, (c, s) in enumerate(zip(t["count"], t["share of errors"])):
        ax.text(c + 1.5, i, f"{c}  ({s:.0%} of errors)", va="center", fontsize=8.5, color=plots.INK_2)
    ax.invert_yaxis()
    ax.set_xlim(0, t["count"].max() * 1.35)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("test utterances (true → predicted)")
    ax.set_title("The ten most common mistakes (blue: involves neutral)")
    plots.save(fig, path)
    return t


def chart_slices(df, path) -> dict:
    panels = [("utterance_words", "Utterance length (words)"), ("clip_seconds", "Clip length"),
              ("speaker_group", "Speaker"), ("whisper_wer", "Whisper word error rate on the clip")]
    tables = {}
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, (col, title) in zip(axes.flat, panels):
        t = slice_table(df, col)
        tables[col] = t
        x = np.arange(len(t))
        ax.bar(x - 0.19, t["accuracy, gold text"], 0.36, color=plots.ACCENT, label="gold transcript")
        ax.bar(x + 0.19, t["accuracy, Whisper text"], 0.36, color=plots.ACCENT_2, label="Whisper transcript")
        ax.set_xticks(x, [f"{i}\nn={n:,d}" for i, n in zip(t.index, t.n)], fontsize=8.5)
        ax.set_ylim(0, 0.8)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        ax.set_title(title)
    axes[0, 0].legend(loc="upper right")
    fig.suptitle("Fused-model accuracy by kind of utterance (MELD test)", x=0.01, ha="left", fontsize=11,
                 color=plots.INK_2)
    fig.tight_layout()
    plots.save(fig, path)
    return tables


def recall_per_class(y_true, y_pred) -> np.ndarray:
    return precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)[1]


def chart_whisper(df, path) -> pd.DataFrame:
    t = pd.DataFrame({"gold transcript": recall_per_class(df.gold, df.fused_pred),
                      "Whisper transcript": recall_per_class(df.gold, df.fused_asr_pred)}, index=LABELS)
    t["drop"] = t["gold transcript"] - t["Whisper transcript"]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    x = np.arange(len(LABELS))
    ax.bar(x - 0.19, t["gold transcript"], 0.36, color=plots.ACCENT, label="gold transcript")
    ax.bar(x + 0.19, t["Whisper transcript"], 0.36, color=plots.ACCENT_2, label="Whisper transcript")
    ax.set_xticks(x, LABELS)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_ylabel("recall (accuracy on that emotion)")
    ax.set_title("What Whisper transcripts cost, per emotion (fused model)")
    ax.legend(loc="upper left")
    plots.save(fig, path)
    return t


def chart_audio_effect(df, path) -> pd.DataFrame:
    fixed = df[df.fused_correct & ~df.text_correct].groupby("gold").size().reindex(LABELS, fill_value=0)
    broke = df[~df.fused_correct & df.text_correct].groupby("gold").size().reindex(LABELS, fill_value=0)
    t = pd.DataFrame({"fixed by audio": fixed, "broken by audio": broke})
    t["net"] = t["fixed by audio"] - t["broken by audio"]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    x = np.arange(len(LABELS))
    ax.bar(x, t["fixed by audio"], 0.6, color=plots.ACCENT, label="text-only wrong → fused right")
    ax.bar(x, -t["broken by audio"], 0.6, color=plots.ACCENT_2, label="text-only right → fused wrong")
    for i, n in enumerate(t["net"]):
        ax.text(i, t["fixed by audio"].iloc[i] + 2, f"net {n:+d}", ha="center", fontsize=8.5, color=plots.INK_2)
    ax.axhline(0, color=plots.BASELINE, lw=1)
    ax.set_xticks(x, LABELS)
    ax.set_ylabel("test utterances")
    ax.set_title("Where adding the voice helps and hurts (vs the text-only model)")
    ax.legend(loc="lower left", fontsize=8.5)
    plots.save(fig, path)
    return t


def chart_label_noise(probe: dict, path) -> pd.Series:
    s = pd.Series(probe["always_wrong_by_class"]).reindex(LABELS).sort_values()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh(s.index, s.values, color=plots.ACCENT, height=0.62)
    for i, v in enumerate(s.values):
        ax.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=8.5, color=plots.INK_2)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_title("Train utterances misclassified in every fold and seed")
    plots.save(fig, path)
    return s


# ---------------------------------------------------------------- report

def write_examples(df: pd.DataFrame, path: Path) -> None:
    """3 test utterances where audio fixed the text-only answer and 3 where it broke it,
    picked by the largest change in the probability given to the true emotion."""
    gain = df.fused_p_gold - df.text_p_gold
    helped = df[df.fused_correct & ~df.text_correct].assign(g=gain).sort_values("g", ascending=False).head(3)
    hurt = df[~df.fused_correct & df.text_correct].assign(g=gain).sort_values("g").head(3)
    L = ["# Where audio changed the prediction (MELD test)", "",
         f"Out of {len(df):,d} test utterances, adding the voice turned a wrong text-only answer into a right one "
         f"{int((df.fused_correct & ~df.text_correct).sum())} times and a right one into a wrong one "
         f"{int((~df.fused_correct & df.text_correct).sum())} times (the deployed models). Rows below are picked "
         "mechanically: the largest change in the probability of the true emotion.", ""]
    for title, rows in (("Audio helped (fused right, text-only wrong)", helped),
                        ("Audio hurt (text-only right, fused wrong)", hurt)):
        L += [f"## {title}", "", "| Clip | Transcript | True | Text-only | Fused |", "| --- | --- | --- | --- | --- |"]
        for r in rows.itertuples():
            L.append(f"| {r.clip} | {r.text.replace('|', '/')} | {r.gold} | {r.text_pred} ({r.text_conf:.2f}) | "
                     f"{r.fused_pred} ({r.fused_conf:.2f}) |")
        L.append("")
    path.write_text("\n".join(L))


def main():
    plots.apply_style()
    df = predict_all()
    y = df.label_id
    acc = df.fused_correct.mean()
    wf1 = f1_score(y, df.fused_pred.map(LABELS.index), average="weighted")
    mf1 = f1_score(y, df.fused_pred.map(LABELS.index), average="macro")

    classes = chart_class_outcomes(df, OUT / "charts" / "class_outcomes.png")
    conf = chart_top_confusions(df, OUT / "charts" / "top_confusions.png")
    slices = chart_slices(df, OUT / "charts" / "accuracy_by_slice.png")
    whisper = chart_whisper(df, OUT / "charts" / "whisper_by_class.png")
    audio = chart_audio_effect(df, OUT / "charts" / "audio_effect.png")
    probe_path = CFG.evals_dir / "experiments" / "train_on_failures.json"
    probe = json.loads(probe_path.read_text())["label_noise_probe"] if probe_path.exists() else None
    noise = chart_label_noise(probe, OUT / "charts" / "label_noise.png") if probe else None
    write_examples(df, OUT / "examples.md")
    conf_table = slice_table(df, "confidence")
    context = slice_table(df, "has_context")

    # Numbers the text below quotes.
    errors = (~df.fused_correct).sum()
    to_neutral = ((df.fused_pred == "neutral") & (df.gold != "neutral")).sum()
    from_neutral = ((df.gold == "neutral") & ~df.fused_correct).sum()
    worst = classes.sort_values("correct").index[:3].tolist()
    lw = slices["utterance_words"]
    wer_t = slices["whisper_wer"]
    asr_acc = df.fused_asr_correct.mean()
    hi, lo = conf_table.iloc[-1], conf_table.iloc[0]

    L = ["# Where the emotion model struggles", "",
         f"The deployed fused model gets **{acc:.1%}** of the {len(df):,d} MELD test utterances right "
         f"(weighted F1 {wf1:.3f}, macro F1 {mf1:.3f}). Its errors are not spread evenly: they concentrate in "
         f"the rare emotions, in long utterances, and in live use with Whisper transcripts. Produced by "
         "`evaluation/03_analyze_errors.py`; every utterance is in `test_predictions.csv`.", "",
         "## 1. Rare emotions collapse into neutral", "",
         "![class outcomes](charts/class_outcomes.png)", "",
         f"Neutral is {classes.loc['neutral', 'n'] / len(df):.0%} of test and the model's safe default. The three "
         f"weakest emotions are **{', '.join(worst)}**, right only "
         + ", ".join(f"{classes.loc[c, 'correct']:.0%}" for c in worst) + " of the time. "
         f"{to_neutral} of the {errors:,d} errors ({to_neutral / errors:.0%}) are a real emotion called neutral, "
         f"and another {from_neutral} ({from_neutral / errors:.0%}) are neutral lines given an emotion.", "",
         md_table(classes[["n", "correct", "called neutral", "another emotion"]], "True emotion"),
         "", "## 2. The most common mistakes", "", "![top confusions](charts/top_confusions.png)", "",
         "| True → predicted | Count | Share of all errors |", "| --- | ---: | ---: |",
         *[f"| {r['gold']} → {r['fused_pred']} | {r['count']} | {r['share of errors']:.0%} |"
           for _, r in conf.iterrows()],
         "", f"Beyond neutral, the high-arousal emotions (anger, joy, surprise) trade places with each other: "
         f"{int(conf[conf.gold.isin(HIGH_AROUSAL) & conf.fused_pred.isin(HIGH_AROUSAL)]['count'].sum())} "
         "of the errors in the top ten are one of them mistaken for another.", "",
         "## 3. Which utterances are hard", "", "![slices](charts/accuracy_by_slice.png)", "",
         f"- **Longer lines are harder**: {lw.iloc[0]['accuracy, gold text']:.0%} accuracy on 1-2 word utterances, "
         f"{lw.iloc[-1]['accuracy, gold text']:.0%} on 21+ words, plausibly because a long line can mix "
         "several emotions under one label.",
         f"- **Speakers differ little** (from {slices['speaker_group']['accuracy, gold text'].min():.0%} to "
         f"{slices['speaker_group']['accuracy, gold text'].max():.0%}); no one character's style dominates the errors.",
         f"- **Dialogue context**: first lines of a dialogue {context.loc['first line', 'accuracy, gold text']:.0%}, "
         f"lines with context {context.loc['has previous lines', 'accuracy, gold text']:.0%}.", "",
         md_table(lw, "Utterance length (words)"), "",
         "## 4. Live speech: Whisper transcripts", "", "![whisper](charts/whisper_by_class.png)", "",
         f"With Whisper's transcripts instead of the gold ones, accuracy falls from {acc:.1%} to {asr_acc:.1%}. "
         f"Even clips Whisper transcribes word-perfectly lose accuracy "
         f"({wer_t.loc['exact', 'accuracy, gold text']:.0%} → {wer_t.loc['exact', 'accuracy, Whisper text']:.0%}): "
         "there the only differences are punctuation, casing and the context lines, which are Whisper's too; "
         f"clips with over 50% word errors fall to {wer_t.loc['>50%', 'accuracy, Whisper text']:.0%}. "
         f"The largest recall drops are "
         + ", ".join(f"{c} (−{100 * whisper.loc[c, 'drop']:.0f} points)"
                     for c in whisper.sort_values('drop', ascending=False).index[:3])
         + ". Retraining on Whisper transcripts recovers only a little (`evals/asr/asr_matched.md`).", "",
         md_table(whisper[["gold transcript", "Whisper transcript"]], "Recall"), "",
         "## 5. Where the voice helps and hurts", "", "![audio effect](charts/audio_effect.png)", "",
         f"Adding the voice fixes {int(audio['fixed by audio'].sum())} text-only errors and breaks "
         f"{int(audio['broken by audio'].sum())} right answers. It helps most on "
         + " and ".join(f"{c} (net {n:+d})" for c, n in audio['net'].sort_values(ascending=False).head(2).items())
         + ", and hurts most on "
         + " and ".join(f"{c} (net {n:+d})" for c, n in audio['net'].sort_values().head(2).items()).replace("-", "−")
         + ". Examples, picked mechanically: [examples.md](examples.md).", ""]
    if noise is not None:
        L += ["## 6. Some errors are label noise, not model failures", "", "![label noise](charts/label_noise.png)", "",
              f"In 5-fold cross-validation on train, {probe['always_wrong_share']:.0%} of train utterances are "
              f"misclassified in every fold and every seed, including {noise['disgust']:.0%} of disgust and "
              f"{noise['fear']:.0%} of fear. A random sample (`hard_train_examples.md`) is full of lines whose label "
              "is hard to defend from the words, such as \"No!\" labeled disgust. Training harder on these makes "
              "the model worse (`evals/experiments/train_on_failures.md`), so part of the remaining error is a "
              "ceiling set by the labels.", ""]
    L += ["## 7. When to trust a prediction", "",
          f"Confidence is honest (temperature-scaled; see `evals/classifiers/charts/calibration_fused.png`): "
          f"predictions with {conf_table.index[-1]} confidence are right {hi['accuracy, gold text']:.0%} "
          f"of the time, those under {lo.name.lstrip('<')} only {lo['accuracy, gold text']:.0%}. The character "
          "uses this: its face blends toward the resting expression by 1 − confidence, and below 0.5 the reply "
          "is told not to assume how the person feels.", "",
          md_table(conf_table[["n", "share", "accuracy, gold text"]], "Confidence (fused, gold transcript)"), "",
          "## What would help most", "",
          "1. **A stronger ASR model** for live use: Whisper costs more accuracy than any modeling choice here.",
          "2. **Fine-tuning the text encoder** on MELD train, the usual route to higher MELD scores "
          "(not measured here).",
          "3. **Better labels, or soft labels, for the rare emotions**: much of the disgust and fear error looks "
          "like annotation ambiguity rather than a learnable signal.",
          "4. Not more weight on hard examples, not RL: both were considered and the first was measured to hurt.", ""]
    (OUT / "analytics.md").write_text("\n".join(L))

    keep = ["clip", "speaker", "text", "asr_text", "wer", "duration_s", "n_words", "has_context", "gold",
            "text_pred", "text_conf", "text_correct", "fused_pred", "fused_conf", "fused_correct",
            "fused_p_gold", "audio_changed", "fused_asr_pred", "fused_asr_conf", "fused_asr_correct"]
    df[keep].to_csv(OUT / "test_predictions.csv", index=False)
    write_json(OUT / "summary.json", {
        "accuracy": acc, "weighted_f1": wf1, "macro_f1": mf1, "accuracy_whisper": asr_acc,
        "class_outcomes": classes.reset_index().to_dict(orient="records"),
        "top_confusions": conf.to_dict(orient="records"),
        "whisper_by_class": whisper.reset_index().to_dict(orient="records"),
        "audio_effect": audio.reset_index().to_dict(orient="records"),
        "slices": {k: v.reset_index().astype({v.index.name or "index": str}).to_dict(orient="records")
                   for k, v in {**slices, "confidence": conf_table, "has_context": context}.items()},
    })
    print((OUT / "analytics.md").read_text()[:3000])


if __name__ == "__main__":
    main()
