"""Compare every emotion classifier on the MELD test set -> evals/classifiers/

Reads the predictions saved by training/04_train_classifiers.py and writes:
  metrics.json                        every metric, mean +- std over seeds, per model
  comparison.md                       the tables, with the charts embedded
  charts/overall_metrics.png          accuracy, macro precision / recall / F1, weighted F1
  charts/per_class_scores.png         precision, recall and F1 per emotion, per model
  charts/confusion_matrices_all.png   every model's confusion matrix side by side
  charts/calibration_fused.png        confidence vs accuracy, before / after temperature
  confusion_matrices/<model>.png      one confusion matrix per model

Metrics are means over the 3 training seeds (logistic regression has one run).
Confusion matrices use each model's best-on-dev seed, i.e. the saved model.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from config import CFG, LABELS, write_json
from src import plots
from src.plots import plt
from src.training import ece, metrics, softmax_np, summarize

ORDER = ["logreg", "text", "text_ctx", "audio", "late_fusion", "fused_utt", "fused"]
NAMES = {"logreg": "Logistic regression (text)", "text": "Text: utterance", "text_ctx": "Text: + context",
         "audio": "Audio (WavLM)", "late_fusion": "Late fusion", "fused_utt": "Fused, no context",
         "fused": "Fused (deployed)"}
DEPLOYED = "fused"
OVERALL = [("accuracy", "Accuracy"), ("macro_precision", "Macro precision"), ("macro_recall", "Macro recall"),
           ("macro_f1", "Macro F1"), ("weighted_f1", "Weighted F1")]


def fmt(s: dict) -> str:
    return f"{s['mean']:.3f} ± {s['std']:.3f}"


def input_label(inp) -> str:
    if isinstance(inp, str):
        return inp
    parts = [{"text": "utterance", "text_ctx": "context + utterance"}[inp["text"]]] if inp["text"] else []
    return " + ".join(parts + (["audio"] if inp["audio"] else []))


def plot_overall(results: dict, models: list[str], path: Path) -> None:
    fig, axes = plt.subplots(1, len(OVERALL), figsize=(14, 3.4), sharey=True)
    y = np.arange(len(models))
    for ax, (key, title) in zip(axes, OVERALL):
        means = [results[m]["test"][key]["mean"] for m in models]
        stds = [results[m]["test"][key]["std"] for m in models]
        colors = [plots.ACCENT if m == DEPLOYED else plots.OTHER for m in models]
        ax.barh(y, means, height=0.62, color=colors, xerr=stds,
                error_kw={"ecolor": plots.INK_2, "elinewidth": 1, "capsize": 2})
        for yi, v, sd in zip(y, means, stds):
            ax.text(v + sd + 0.012, yi, f"{v:.3f}", va="center", fontsize=8.5, color=plots.INK_2)
        ax.set_xlim(0, 0.85)
        ax.set_title(title)
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks(y, [NAMES[m] for m in models])
    axes[0].invert_yaxis()
    fig.suptitle("MELD test (2,610 utterances), mean ± std over 3 seeds; blue = the model the app uses",
                 x=0.01, ha="left", fontsize=10, color=plots.INK_2, y=1.03)
    fig.tight_layout()
    plots.save(fig, path)


def plot_per_class(results: dict, models: list[str], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    titles = {"precision": "Precision", "recall": "Recall", "f1": "F1"}
    for ax, metric in zip(axes, titles):
        vals = [[results[m]["test"]["per_class"][c][metric] for c in LABELS] for m in models]
        rows = [NAMES[m] for m in models] if ax is axes[0] else [""] * len(models)
        im = plots.heatmap(ax, vals, rows, LABELS, title=titles[metric])
    cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01)
    cb.outline.set_visible(False)
    cb.ax.tick_params(color=plots.MUTED, labelcolor=plots.INK_2)
    fig.suptitle("Per-emotion scores on MELD test (mean over seeds). Recall = accuracy on that emotion.",
                 x=0.01, ha="left", fontsize=10, color=plots.INK_2)
    plots.save(fig, path)


def plot_confusion(ax, counts: np.ndarray, title: str, show_y: bool = True):
    cm = counts / np.maximum(counts.sum(1, keepdims=True), 1)
    im = plots.heatmap(ax, cm, LABELS if show_y else [""] * len(LABELS), LABELS, fmt="{:.0%}", title=title)
    ax.set_xlabel("predicted")
    if show_y:
        ax.set_ylabel("true emotion")
    return im


def reliability(p: np.ndarray, y: np.ndarray):
    conf, correct = p.max(1), p.argmax(1) == y
    xs, ys = [], []
    for lo in np.arange(0, 1, 0.1):
        m = (conf > lo) & (conf <= lo + 0.1)
        if m.sum() >= 20:
            xs.append(conf[m].mean())
            ys.append(correct[m].mean())
    return xs, ys


def calibrated(raw: np.ndarray, temperature: float) -> np.ndarray:
    """softmax(logits / T) from saved probabilities: log p differs from the logits by a constant."""
    return softmax_np(np.log(np.clip(raw, 1e-12, 1)), temperature)


def plot_calibration(raw: np.ndarray, temperature: float, y: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], color=plots.BASELINE, lw=1.2, ls="--", label="perfect calibration")
    for label, p, color in (("before temperature scaling", raw, plots.ACCENT_2),
                            (f"after (T = {temperature:.2f}, fit on dev)", calibrated(raw, temperature), plots.ACCENT)):
        xs, ys = reliability(p, y)
        ax.plot(xs, ys, marker="o", ms=6, lw=2, color=color, label=label)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="mean confidence in bin", ylabel="accuracy in bin")
    ax.set_title("Fused model: is the confidence honest?")
    ax.grid(True)
    ax.legend(loc="upper left")
    plots.save(fig, path)


def write_markdown(results, models, calibration, n_test, path: Path) -> None:
    L = ["# Emotion classifiers compared (MELD test)", "",
         f"All models see the same {n_test:,d} test utterances (official MELD split). Numbers are mean ± std "
         "over 3 training seeds (logistic regression: one run). Every choice was made on dev. "
         "Produced by `evaluation/01_compare_classifiers.py` from `predictions.npz`.", "",
         "![overall](charts/overall_metrics.png)", "",
         "| Model | Input | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 | Dev weighted F1 |",
         "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for m in models:
        t = results[m]["test"]
        name = f"**{NAMES[m]}**" if m == DEPLOYED else NAMES[m]
        L.append(f"| {name} | {input_label(results[m]['input'])} | {fmt(t['accuracy'])} | "
                 f"{fmt(t['macro_precision'])} | {fmt(t['macro_recall'])} | {fmt(t['macro_f1'])} | "
                 f"{fmt(t['weighted_f1'])} | {fmt(results[m]['dev_weighted_f1'])} |")
    L += ["", "Weighted scores follow the MELD convention (neutral is about half of test); macro scores weigh "
          "the seven emotions equally, which is why they are much lower.", "",
          "## Per emotion", "", "![per class](charts/per_class_scores.png)", ""]
    for metric, title in (("precision", "Precision"), ("recall", "Recall (accuracy on that emotion)"), ("f1", "F1")):
        L += [f"**{title}**, mean over seeds", "",
              "| Model | " + " | ".join(LABELS) + " |", "| --- |" + " ---: |" * len(LABELS)]
        for m in models:
            pc = results[m]["test"]["per_class"]
            L.append(f"| {NAMES[m]} | " + " | ".join(f"{pc[c][metric]:.3f}" for c in LABELS) + " |")
        L.append("")
    support = results[models[0]]["test_per_seed"][0]["per_class"]
    L += ["Test utterances per emotion: " + ", ".join(f"{c} {support[c]['support']:,d}" for c in LABELS) + ".", "",
          "## Confusion matrices", "",
          "Each model's saved (best-on-dev) seed. Rows are the true emotion and sum to 100%.", "",
          "![confusion matrices](charts/confusion_matrices_all.png)", "",
          "One file per model: " + ", ".join(f"[{NAMES[m]}](confusion_matrices/{m}.png)" for m in models) + ".", "",
          "## Calibration", "", "![calibration](charts/calibration_fused.png)", "",
          "| Model | Temperature (fit on dev) | Dev ECE before → after | Test ECE before → after |",
          "| --- | ---: | ---: | ---: |"]
    for role, c in calibration.items():
        L.append(f"| {role} | {c['temperature']:.3f} | {c['dev_ece_before']:.3f} → {c['dev_ece_after']:.3f} | "
                 f"{c['test_ece_before']:.3f} → {c['test_ece_after']:.3f} |")
    path.write_text("\n".join(L) + "\n")


def main():
    plots.apply_style()
    out_dir = CFG.evals_dir / "classifiers"
    pred = np.load(out_dir / "predictions.npz")
    summary = json.loads((out_dir / "training_summary.json").read_text())
    meta = json.loads((CFG.models_dir / "meta.json").read_text())
    y_dev, y_test = pred["y_dev"], pred["y_test"]
    models = [m for m in ORDER if f"{m}__test" in pred]

    results = {}
    for m in models:
        test, dev = pred[f"{m}__test"], pred[f"{m}__dev"]          # [seeds, N, 7]
        runs = [metrics(y_test, p.argmax(1)) for p in test]
        dev_f1 = [float(f1_score(y_dev, p.argmax(1), average="weighted")) for p in dev]
        best = int(np.argmax(dev_f1))                              # the saved model's seed
        counts = confusion_matrix(y_test, test[best].argmax(1), labels=range(len(LABELS)))
        results[m] = {"name": NAMES[m], "input": summary["models"][m]["input"], "n_seeds": len(test),
                      "test": summarize(runs), "test_per_seed": runs,
                      "dev_weighted_f1": {"mean": float(np.mean(dev_f1)), "std": float(np.std(dev_f1))},
                      "saved_model": {"seed_index": best, "test": runs[best], "confusion_counts": counts.tolist()}}
        print(f"{NAMES[m]:28s} accuracy {fmt(results[m]['test']['accuracy'])}  "
              f"weighted F1 {fmt(results[m]['test']['weighted_f1'])}")

    plot_overall(results, models, out_dir / "charts" / "overall_metrics.png")
    plot_per_class(results, models, out_dir / "charts" / "per_class_scores.png")

    for m in models:  # one confusion matrix per model ...
        fig, ax = plt.subplots(figsize=(5.4, 4.8))
        acc = results[m]["saved_model"]["test"]["accuracy"]
        plot_confusion(ax, np.array(results[m]["saved_model"]["confusion_counts"]), f"{NAMES[m]}, accuracy {acc:.3f}")
        plots.save(fig, out_dir / "confusion_matrices" / f"{m}.png")
    cols = 4                                                       # ... and all of them together
    rows = int(np.ceil(len(models) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.4 * rows))
    for k, ax in enumerate(axes.flat):
        if k >= len(models):
            ax.axis("off")
            continue
        m = models[k]
        plot_confusion(ax, np.array(results[m]["saved_model"]["confusion_counts"]),
                       f"{NAMES[m]} ({results[m]['saved_model']['test']['accuracy']:.3f})", show_y=k % cols == 0)
    fig.suptitle("Confusion matrices, MELD test. Rows sum to 100%: where each true emotion's utterances went.",
                 x=0.01, ha="left", fontsize=11, color=plots.INK_2)
    fig.tight_layout()
    plots.save(fig, out_dir / "charts" / "confusion_matrices_all.png")

    # Calibration of the two deployed heads (temperatures were fit on dev during training).
    calibration = {}
    for role, name in (("fused", "fused"), ("text_only", meta["text_key"])):
        raw = pred[f"{name}__test"][results[name]["saved_model"]["seed_index"]]
        t = meta["temperatures"][role]
        calibration[role] = {**summary["calibration_dev"][role], "test_ece_before": ece(raw, y_test),
                             "test_ece_after": ece(calibrated(raw, t), y_test)}
        if role == "fused":
            plot_calibration(raw, t, y_test, out_dir / "charts" / "calibration_fused.png")

    write_json(out_dir / "metrics.json", {
        "test_set_size": int(len(y_test)), "seeds": summary["seeds"],
        "selected_text_variant": summary["selected_text_variant"], "deployed": DEPLOYED,
        "models": results, "calibration": calibration,
        "audio_layer_weights_fused": meta["audio_layer_weights_fused"]})
    write_markdown(results, models, calibration, len(y_test), out_dir / "comparison.md")
    print(f"wrote {out_dir.relative_to(CFG.root)}/: comparison.md, metrics.json, charts/, confusion_matrices/")


if __name__ == "__main__":
    main()
