"""Train the same small head on each feature set and evaluate on MELD test.

Configurations (the only variable is the input):
  text         utterance embedding
  text_ctx     context + utterance embedding
  audio        WavLM layers, learned layer weighting
  fused        best text variant (picked on dev) + audio, concatenated
  fused_utt    utterance-only text + audio (only when text_ctx wins on dev)
  late_fusion  average of the text and audio heads' probabilities

Each trained configuration runs with 3 seeds; test metrics are mean +- std.
Model selection (early stopping, text variant, checkpoint seed, temperature)
uses dev only. Test is evaluated once per final configuration.

Outputs: results/metrics.json, results/table.md, results/confusion_{text,fused}.png,
results/examples.md, checkpoints/{text,fused}_head.pt, checkpoints/meta.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

from config import CFG, LABELS, set_seed, write_json
from src.fusion import EmotionHead, save_head

SPLITS = ["train", "dev", "test"]


# ---------------------------------------------------------------- data

def load_split(split: str) -> dict:
    d = {name: torch.from_numpy(np.load(CFG.cache_dir / f"{split}_{name}.npy").astype(np.float32))
         for name in ("text", "text_ctx", "audio")}
    ids = pd.read_csv(CFG.cache_dir / f"{split}_ids.csv")
    d["y"] = torch.tensor(ids["label_id"].to_numpy())
    d["ids"] = ids
    return d


def inputs(d: dict, text_key: str | None, use_audio: bool, idx=slice(None)) -> dict:
    return {"text": d[text_key][idx] if text_key else None,
            "audio": d["audio"][idx] if use_audio else None}


# ---------------------------------------------------------------- metrics

def metrics(y_true, y_pred) -> dict:
    per_class = f1_score(y_true, y_pred, labels=range(len(LABELS)), average=None, zero_division=0)
    return {
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class_f1": {c: float(f) for c, f in zip(LABELS, per_class)},
    }


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error with equal-width confidence bins."""
    conf, pred = probs.max(1), probs.argmax(1)
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)


def summarize(runs: list[dict]) -> dict:
    """Mean and std over seeds for the scalar metrics and per-class F1."""
    out = {}
    for k in ("weighted_f1", "macro_f1", "accuracy"):
        v = np.array([r[k] for r in runs])
        out[k] = {"mean": float(v.mean()), "std": float(v.std())}
    out["per_class_f1"] = {c: float(np.mean([r["per_class_f1"][c] for r in runs])) for c in LABELS}
    return out


# ---------------------------------------------------------------- training

@torch.no_grad()
def predict_logits(head: EmotionHead, d: dict, text_key, use_audio, bs: int = 1024) -> torch.Tensor:
    head.eval()
    out = [head(**inputs(d, text_key, use_audio, slice(i, i + bs))) for i in range(0, len(d["y"]), bs)]
    return torch.cat(out)


def train_head(train: dict, dev: dict, text_key: str | None, use_audio: bool, seed: int):
    """AdamW + class-weighted CE, early stopping on dev weighted F1."""
    set_seed(seed)
    head = EmotionHead(use_text=text_key is not None, use_audio=use_audio)
    opt = torch.optim.AdamW(head.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)

    counts = torch.bincount(train["y"], minlength=len(LABELS)).float()
    class_w = 1.0 / counts.sqrt()
    class_w = class_w / class_w.mean()

    best = {"f1": -1.0, "epoch": -1, "state": None}
    n = len(train["y"])
    gen = torch.Generator().manual_seed(seed)
    for epoch in range(CFG.max_epochs):
        head.train()
        for idx in torch.randperm(n, generator=gen).split(CFG.batch_size):
            logits = head(**inputs(train, text_key, use_audio, idx))
            loss = F.cross_entropy(logits, train["y"][idx], weight=class_w)
            opt.zero_grad()
            loss.backward()
            opt.step()
        dev_pred = predict_logits(head, dev, text_key, use_audio).argmax(1)
        f1 = f1_score(dev["y"], dev_pred, average="weighted")
        if f1 > best["f1"]:
            best = {"f1": float(f1), "epoch": epoch,
                    "state": {k: v.clone() for k, v in head.state_dict().items()}}
        elif epoch - best["epoch"] >= CFG.patience:
            break
    head.load_state_dict(best["state"])
    head.eval()
    return head, best["f1"], best["epoch"]


def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Single scalar T minimizing NLL of softmax(logits / T) on dev."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp())


# ---------------------------------------------------------------- reporting

def plot_confusion(y_true, y_pred, title: str, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=range(len(LABELS)), normalize="true")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold")
    ax.set_title(title)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fmt(s: dict) -> str:
    return f"{s['mean']:.3f} ± {s['std']:.3f}"


def write_table(results: dict, lr_test: dict, text_key: str, path: Path) -> None:
    lines = [
        "| Configuration | Input | Test weighted F1 | Test macro F1 | Test accuracy | Dev weighted F1 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    desc = {"text": "utterance (RoBERTa)", "text_ctx": "context + utterance (RoBERTa)",
            "audio": "WavLM, layer-weighted", "fused": f"{text_key} + audio, concat MLP (main model)",
            "fused_utt": "text + audio, concat MLP (no context)",
            "late_fusion": f"mean of {text_key} and audio probabilities"}
    for name, r in results.items():
        t = r["test"]
        lines.append(f"| {name} | {desc[name]} | {fmt(t['weighted_f1'])} | {fmt(t['macro_f1'])} | "
                     f"{fmt(t['accuracy'])} | {fmt(r['dev_weighted_f1'])} |")
    lines.append(f"| logreg (sanity) | utterance (RoBERTa), sklearn | {lr_test['weighted_f1']:.3f} | "
                 f"{lr_test['macro_f1']:.3f} | {lr_test['accuracy']:.3f} | – |")
    lines += ["", f"Mean ± std over seeds {list(CFG.seeds)}. MELD official test split, "
              f"{len(load_split('test')['y'])} utterances.", "",
              "Per-class test F1 (mean over seeds):", "",
              "| Configuration | " + " | ".join(LABELS) + " |",
              "| --- |" + " --- |" * len(LABELS)]
    for name, r in results.items():
        pc = r["test"]["per_class_f1"]
        lines.append(f"| {name} | " + " | ".join(f"{pc[c]:.3f}" for c in LABELS) + " |")
    path.write_text("\n".join(lines) + "\n")


def write_examples(test: dict, text_pred, text_probs, fused_pred, fused_probs, path: Path) -> None:
    """3 test utterances where fused is right and text-only wrong, and 3 the other way.
    Picked by the largest change in the probability given to the gold label."""
    man = pd.read_csv(CFG.manifest_dir / "test.csv", keep_default_na=False)
    man = test["ids"].merge(man, on=["dialogue_id", "utterance_id"], how="left", suffixes=("", "_m"))
    y = test["y"].numpy()
    gold_gain = fused_probs[np.arange(len(y)), y] - text_probs[np.arange(len(y)), y]

    def rows(mask, order):
        idx = np.where(mask)[0]
        return idx[np.argsort(order[idx])][:3]

    helped = rows((fused_pred == y) & (text_pred != y), -gold_gain)
    hurt = rows((fused_pred != y) & (text_pred == y), gold_gain)

    out = ["# Where audio changed the prediction (MELD test)", "",
           f"Out of {len(y)} test utterances, audio flipped a wrong text-only prediction to correct "
           f"{int(((fused_pred == y) & (text_pred != y)).sum())} times and a correct one to wrong "
           f"{int(((fused_pred != y) & (text_pred == y)).sum())} times "
           "(saved text-only and fused checkpoints).", ""]
    for title, idx in [("Audio helped (fused correct, text-only wrong)", helped),
                       ("Audio hurt (text-only correct, fused wrong)", hurt)]:
        out += [f"## {title}", "",
                "| Clip | Transcript | Gold | Text-only | Fused |", "| --- | --- | --- | --- | --- |"]
        for i in idx:
            r = man.iloc[i]
            clip = f"dia{r.dialogue_id}_utt{r.utterance_id}"
            tp, fp = text_pred[i], fused_pred[i]
            out.append(f"| {clip} | {r.text.replace('|', '/')} | {LABELS[y[i]]} | "
                       f"{LABELS[tp]} ({text_probs[i, tp]:.2f}) | {LABELS[fp]} ({fused_probs[i, fp]:.2f}) |")
        out.append("")
    path.write_text("\n".join(out))


# ---------------------------------------------------------------- main

def run_config(name, train, dev, test, text_key, use_audio):
    runs, dev_f1s, heads = [], [], []
    for seed in CFG.seeds:
        head, dev_f1, epoch = train_head(train, dev, text_key, use_audio, seed)
        pred = predict_logits(head, test, text_key, use_audio).argmax(1)
        runs.append(metrics(test["y"], pred))
        dev_f1s.append(dev_f1)
        heads.append(head)
        print(f"  {name} seed {seed}: dev wF1 {dev_f1:.3f} (epoch {epoch}), test wF1 {runs[-1]['weighted_f1']:.3f}")
    return {"test": summarize(runs), "test_per_seed": runs,
            "dev_weighted_f1": {"mean": float(np.mean(dev_f1s)), "std": float(np.std(dev_f1s))},
            "dev_per_seed": dev_f1s}, heads


def main():
    torch.set_num_threads(8)
    train, dev, test = (load_split(s) for s in SPLITS)
    print(f"rows: train {len(train['y'])}, dev {len(dev['y'])}, test {len(test['y'])}")
    CFG.ckpt_dir.mkdir(exist_ok=True)
    results, heads = {}, {}

    for name, text_key, use_audio in [("text", "text", False), ("text_ctx", "text_ctx", False),
                                      ("audio", None, True)]:
        results[name], heads[name] = run_config(name, train, dev, test, text_key, use_audio)

    # Pick the text variant on dev, never on test.
    text_key = max(["text", "text_ctx"], key=lambda k: results[k]["dev_weighted_f1"]["mean"])
    print(f"best text variant on dev: {text_key}")
    results["fused"], heads["fused"] = run_config("fused", train, dev, test, text_key, True)
    if text_key != "text":
        # Same fusion with the utterance-only text feature, to see what audio adds
        # when there is no dialogue context. Reported, never used for selection.
        results["fused_utt"], heads["fused_utt"] = run_config("fused_utt", train, dev, test, "text", True)

    # Late fusion: average probabilities of the text and audio heads, seed by seed.
    late_runs, late_dev = [], []
    for th, ah in zip(heads[text_key], heads["audio"]):
        p = lambda d: (F.softmax(predict_logits(th, d, text_key, False), 1)
                       + F.softmax(predict_logits(ah, d, None, True), 1)) / 2
        late_runs.append(metrics(test["y"], p(test).argmax(1)))
        late_dev.append(float(f1_score(dev["y"], p(dev).argmax(1), average="weighted")))
    results["late_fusion"] = {"test": summarize(late_runs), "test_per_seed": late_runs,
                              "dev_weighted_f1": {"mean": float(np.mean(late_dev)), "std": float(np.std(late_dev))},
                              "dev_per_seed": late_dev}

    # Sanity baseline: logistic regression on the utterance embedding.
    scaler = StandardScaler().fit(train["text"].numpy())
    lr = LogisticRegression(max_iter=2000).fit(scaler.transform(train["text"].numpy()), train["y"].numpy())
    lr_test = metrics(test["y"], lr.predict(scaler.transform(test["text"].numpy())))
    print(f"  logreg (text) test wF1 {lr_test['weighted_f1']:.3f}")

    # Save the best-on-dev seed of the text-only and fused heads for the demo.
    best_seed = {k: int(np.argmax(results[k]["dev_per_seed"])) for k in (text_key, "fused")}
    text_head = heads[text_key][best_seed[text_key]]
    fused_head = heads["fused"][best_seed["fused"]]

    # Temperature scaling for the fused head, fit on dev.
    dev_logits = predict_logits(fused_head, dev, text_key, True)
    test_logits = predict_logits(fused_head, test, text_key, True)
    T = fit_temperature(dev_logits, dev["y"])
    calib = {
        "temperature": T,
        "dev_ece_before": ece(F.softmax(dev_logits, 1).numpy(), dev["y"].numpy()),
        "dev_ece_after": ece(F.softmax(dev_logits / T, 1).numpy(), dev["y"].numpy()),
        "test_ece_before": ece(F.softmax(test_logits, 1).numpy(), test["y"].numpy()),
        "test_ece_after": ece(F.softmax(test_logits / T, 1).numpy(), test["y"].numpy()),
    }
    print(f"  temperature {T:.3f}; dev ECE {calib['dev_ece_before']:.3f} -> {calib['dev_ece_after']:.3f}")

    # The text-only head also serves typed messages (no audio), so calibrate it too.
    dev_logits_t = predict_logits(text_head, dev, text_key, False)
    test_logits_t = predict_logits(text_head, test, text_key, False)
    T_text = fit_temperature(dev_logits_t, dev["y"])
    calib_text = {
        "temperature": T_text,
        "dev_ece_before": ece(F.softmax(dev_logits_t, 1).numpy(), dev["y"].numpy()),
        "dev_ece_after": ece(F.softmax(dev_logits_t / T_text, 1).numpy(), dev["y"].numpy()),
        "test_ece_before": ece(F.softmax(test_logits_t, 1).numpy(), test["y"].numpy()),
        "test_ece_after": ece(F.softmax(test_logits_t / T_text, 1).numpy(), test["y"].numpy()),
    }
    print(f"  text-only temperature {T_text:.3f}; dev ECE {calib_text['dev_ece_before']:.3f} -> "
          f"{calib_text['dev_ece_after']:.3f}")

    save_head(text_head, CFG.ckpt_dir / "text_head.pt",
              {"text_key": text_key, "seed": CFG.seeds[best_seed[text_key]], "temperature": T_text})
    save_head(fused_head, CFG.ckpt_dir / "fused_head.pt",
              {"text_key": text_key, "seed": CFG.seeds[best_seed["fused"]], "temperature": T})
    meta = {"text_key": text_key, "temperature": T, "text_temperature": T_text,
            "text_head_seed": CFG.seeds[best_seed[text_key]], "fused_head_seed": CFG.seeds[best_seed["fused"]],
            "text_head_params": sum(p.numel() for p in text_head.parameters()),
            "fused_head_params": sum(p.numel() for p in fused_head.parameters()),
            "audio_layer_weights": torch.softmax(fused_head.layer_weights.logits, 0).tolist()}
    write_json(CFG.ckpt_dir / "meta.json", meta)

    # Confusion matrices and examples from the saved checkpoints.
    text_probs = F.softmax(predict_logits(text_head, test, text_key, False), 1).numpy()
    fused_probs = F.softmax(test_logits / T, 1).numpy()
    text_pred, fused_pred = text_probs.argmax(1), fused_probs.argmax(1)
    y = test["y"].numpy()
    plot_confusion(y, text_pred, f"Text-only ({text_key}), test", CFG.results_dir / "confusion_text.png")
    plot_confusion(y, fused_pred, "Fused (text + audio), test", CFG.results_dir / "confusion_fused.png")
    write_examples(test, text_pred, text_probs, fused_pred, fused_probs, CFG.results_dir / "examples.md")

    write_json(CFG.results_dir / "metrics.json", {
        "split_sizes": {s: len(d["y"]) for s, d in zip(SPLITS, (train, dev, test))},
        "seeds": list(CFG.seeds),
        "selected_text_variant": text_key,
        "configs": results,
        "logreg_text_test": lr_test,
        "calibration_fused": calib,
        "calibration_text": calib_text,
        "saved_checkpoints": {
            "text_head": {"seed": meta["text_head_seed"], "test": metrics(y, text_pred)},
            "fused_head": {"seed": meta["fused_head_seed"], "test": metrics(y, fused_pred)},
        },
        "audio_layer_weights_fused": meta["audio_layer_weights"],
    })
    write_table(results, lr_test, text_key, CFG.results_dir / "table.md")
    print((CFG.results_dir / "table.md").read_text())


if __name__ == "__main__":
    main()
