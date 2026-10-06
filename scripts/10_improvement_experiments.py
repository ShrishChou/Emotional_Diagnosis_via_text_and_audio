"""Does training harder on the failures help? -> results/improvement_experiments.{json,md}

Tests the "find where the model fails and train on those cases" idea in its
legitimate form: failures are found on TRAIN (out-of-fold), never on test.
Every variant is the main fused model (text_ctx + audio, same head, same
hyperparameters) trained 3 seeds; choices are made on dev, and test is reported
once per variant.

  baseline        the fused model as in 03_train_eval.py
  hard_x2/x3      5-fold out-of-fold predictions on train; train examples the
                  model gets wrong are upweighted 2x / 3x, then retrain on all of train
  focal_g1/g2     focal loss (gamma 1 / 2): smoothly downweights easy examples
  neutral_offset  baseline heads, but the neutral logit is lowered by an offset
                  tuned on dev (neutral absorbs most errors)

Also a label-noise probe: train examples misclassified out-of-fold by all three
seeds, by class, with a sample written to results/hard_train_examples.md.
"""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from config import CFG, LABELS, set_seed, write_json
from src.fusion import EmotionHead

# Reuse data loading, metrics and prediction from 03_train_eval.py.
_spec = importlib.util.spec_from_file_location("train_eval", Path(__file__).with_name("03_train_eval.py"))
te = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(te)

TEXT_KEY = "text_ctx"
N_FOLDS = 5
NEUTRAL = LABELS.index("neutral")


def subset(d: dict, idx) -> dict:
    return {k: (v[idx] if isinstance(v, torch.Tensor) else v) for k, v in d.items()}


def train_head(train, dev, seed, example_w=None, focal_gamma=0.0):
    """03's training loop plus optional per-example weights and focal loss."""
    set_seed(seed)
    head = EmotionHead(use_text=True, use_audio=True)
    opt = torch.optim.AdamW(head.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    counts = torch.bincount(train["y"], minlength=len(LABELS)).float()
    class_w = 1.0 / counts.sqrt()
    class_w = class_w / class_w.mean()
    ex_w = example_w if example_w is not None else torch.ones(len(train["y"]))

    best = {"f1": -1.0, "epoch": -1, "state": None}
    gen = torch.Generator().manual_seed(seed)
    for epoch in range(CFG.max_epochs):
        head.train()
        for idx in torch.randperm(len(train["y"]), generator=gen).split(CFG.batch_size):
            logits = head(**te.inputs(train, TEXT_KEY, True, idx))
            y = train["y"][idx]
            ce = F.cross_entropy(logits, y, reduction="none")
            if focal_gamma:
                p_true = torch.exp(-ce)
                ce = (1 - p_true) ** focal_gamma * ce
            w = class_w[y] * ex_w[idx]
            loss = (ce * w).sum() / w.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        f1 = f1_score(dev["y"], te.predict_logits(head, dev, TEXT_KEY, True).argmax(1), average="weighted")
        if f1 > best["f1"]:
            best = {"f1": float(f1), "epoch": epoch, "state": {k: v.clone() for k, v in head.state_dict().items()}}
        elif epoch - best["epoch"] >= CFG.patience:
            break
    head.load_state_dict(best["state"])
    return head.eval()


def out_of_fold_wrong(train, dev, seed) -> torch.Tensor:
    """Boolean mask over train: misclassified by a head that did not train on it.
    (Early stopping inside each fold uses dev, as everywhere else.)"""
    wrong = torch.zeros(len(train["y"]), dtype=torch.bool)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for fit_idx, held_idx in skf.split(np.zeros(len(train["y"])), train["y"].numpy()):
        head = train_head(subset(train, torch.tensor(fit_idx)), dev, seed)
        held = subset(train, torch.tensor(held_idx))
        pred = te.predict_logits(head, held, TEXT_KEY, True).argmax(1)
        wrong[held_idx] = pred != held["y"]
    return wrong


def evaluate(heads, dev, test, neutral_offset: float = 0.0) -> dict:
    def preds(head, d):
        logits = te.predict_logits(head, d, TEXT_KEY, True).clone()
        logits[:, NEUTRAL] -= neutral_offset
        return logits.argmax(1)
    dev_f1 = [float(f1_score(dev["y"], preds(h, dev), average="weighted")) for h in heads]
    test_runs = [te.metrics(test["y"], preds(h, test)) for h in heads]
    return {"dev_weighted_f1": {"mean": float(np.mean(dev_f1)), "std": float(np.std(dev_f1))},
            "test": te.summarize(test_runs)}


def main():
    torch.set_num_threads(4)
    train, dev, test = (te.load_split(s) for s in ("train", "dev", "test"))
    results, heads = {}, {}

    print("baseline")
    heads["baseline"] = [train_head(train, dev, s) for s in CFG.seeds]
    results["baseline"] = evaluate(heads["baseline"], dev, test)

    print("out-of-fold predictions on train (5 folds x 3 seeds)")
    wrong = {s: out_of_fold_wrong(train, dev, s) for s in CFG.seeds}
    for factor in (2.0, 3.0):
        name = f"hard_x{int(factor)}"
        print(name)
        heads[name] = [train_head(train, dev, s, example_w=1.0 + (factor - 1.0) * wrong[s].float())
                       for s in CFG.seeds]
        results[name] = evaluate(heads[name], dev, test)

    for gamma in (1.0, 2.0):
        name = f"focal_g{int(gamma)}"
        print(name)
        heads[name] = [train_head(train, dev, s, focal_gamma=gamma) for s in CFG.seeds]
        results[name] = evaluate(heads[name], dev, test)

    # Neutral offset: tune on dev (mean over seeds), then apply to test.
    grid = np.round(np.arange(0.0, 2.01, 0.1), 2)
    dev_curve = [evaluate(heads["baseline"], dev, dev, off)["dev_weighted_f1"]["mean"] for off in grid]
    best_off = float(grid[int(np.argmax(dev_curve))])
    results["neutral_offset"] = {**evaluate(heads["baseline"], dev, test, best_off), "offset": best_off}

    # Label-noise probe: wrong out-of-fold for every seed.
    always_wrong = torch.stack([wrong[s] for s in CFG.seeds]).all(0).numpy()
    y = train["y"].numpy()
    probe = {
        "oof_error_rate_per_seed": {str(s): float(wrong[s].float().mean()) for s in CFG.seeds},
        "always_wrong_share": float(always_wrong.mean()),
        "always_wrong_by_class": {c: float(always_wrong[y == i].mean()) for i, c in enumerate(LABELS)},
    }
    man = pd.read_csv(CFG.manifest_dir / "train.csv", keep_default_na=False)
    tr = train["ids"].merge(man, on=["dialogue_id", "utterance_id"], suffixes=("", "_m"))
    sample = tr[always_wrong].sample(n=20, random_state=0)
    lines = ["# Train utterances the model gets wrong in every fold and seed", "",
             f"{always_wrong.sum():,d} of {len(y):,d} train utterances ({always_wrong.mean():.0%}). "
             "A random 20 below. TODO(Shrish): listen to some and judge how many gold labels are "
             "defensible; that bounds how much 'training on failures' can help.", "",
             "| Clip | Speaker | Transcript | Gold |", "| --- | --- | --- | --- |"]
    for r in sample.itertuples():
        lines.append(f"| dia{r.dialogue_id}_utt{r.utterance_id} | {r.speaker} | {r.text.replace('|', '/')} | {r.emotion} |")
    (CFG.results_dir / "hard_train_examples.md").write_text("\n".join(lines) + "\n")

    write_json(CFG.results_dir / "improvement_experiments.json",
               {"text_key": TEXT_KEY, "seeds": list(CFG.seeds), "variants": results, "label_noise_probe": probe,
                "neutral_offset_dev_curve": dict(zip(map(str, grid), dev_curve))})
    t = ["| Variant | Dev weighted F1 | Test weighted F1 | Test macro F1 | Test accuracy |",
         "| --- | ---: | ---: | ---: | ---: |"]
    for name, r in results.items():
        label = f"{name} ({r['offset']:.1f})" if "offset" in r else name
        t.append(f"| {label} | {te.fmt(r['dev_weighted_f1'])} | {te.fmt(r['test']['weighted_f1'])} | "
                 f"{te.fmt(r['test']['macro_f1'])} | {te.fmt(r['test']['accuracy'])} |")
    t += ["", "Per-class test F1 (mean over seeds):", "",
          "| Variant | " + " | ".join(LABELS) + " |", "| --- |" + " ---: |" * len(LABELS)]
    for name, r in results.items():
        t.append(f"| {name} | " + " | ".join(f"{r['test']['per_class_f1'][c]:.3f}" for c in LABELS) + " |")
    t += ["", f"Label-noise probe: out-of-fold train error rate "
          f"{np.mean(list(probe['oof_error_rate_per_seed'].values())):.0%}; "
          f"{probe['always_wrong_share']:.0%} of train is wrong in every seed. By class: "
          + ", ".join(f"{c} {v:.0%}" for c, v in probe["always_wrong_by_class"].items()) + "."]
    (CFG.results_dir / "improvement_experiments.md").write_text("\n".join(t) + "\n")
    print("\n".join(t))


if __name__ == "__main__":
    main()
