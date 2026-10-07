"""Does training harder on the failures help? -> evals/experiments/train_on_failures.{json,md}

Tests "find where the model fails and train on those cases" in its legitimate form:
the failures are found on TRAIN (5-fold out-of-fold predictions), never on test.
Every variant is the deployed fused model (same input, head and hyperparameters),
trained with 3 seeds; choices are made on dev and test is reported once per variant.

  baseline        the fused model as trained in training/04_train_classifiers.py
  hard_x2 / x3    train utterances misclassified out-of-fold weigh 2x / 3x, then retrain
  focal_g1 / g2   focal loss (gamma 1 / 2): smoothly down-weights easy examples
  neutral_offset  the baseline, with the neutral logit lowered by an offset tuned on dev

Also a label-noise probe: the train utterances misclassified in every fold of every
seed, by emotion, with a random sample in evals/struggles/hard_train_examples.md.
About 5 minutes on the CPU.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from config import CFG, LABELS, write_json
from src.data import clip_id, load_manifest
from src.training import load_split, metrics, predict_logits, summarize, train_head

TEXT_KEY = json.loads((CFG.models_dir / "meta.json").read_text())["text_key"]   # the deployed fused input
N_FOLDS = 5
NEUTRAL = LABELS.index("neutral")


def subset(d: dict, idx) -> dict:
    return {k: (v[idx] if isinstance(v, torch.Tensor) else v) for k, v in d.items()}


def fused(train, dev, seed, **kwargs):
    return train_head(train, dev, TEXT_KEY, True, seed, **kwargs)[0]


def out_of_fold_wrong(train: dict, dev: dict, seed: int) -> torch.Tensor:
    """Mask over train: misclassified by a model that never trained on that utterance."""
    wrong = torch.zeros(len(train["y"]), dtype=torch.bool)
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for fit_idx, held_idx in folds.split(np.zeros(len(train["y"])), train["y"].numpy()):
        head = fused(subset(train, torch.tensor(fit_idx)), dev, seed)
        held = subset(train, torch.tensor(held_idx))
        wrong[held_idx] = predict_logits(head, held, TEXT_KEY, True).argmax(1) != held["y"]
    return wrong


def evaluate(heads: list, dev: dict, test: dict, neutral_offset: float = 0.0) -> dict:
    def predict(head, d):
        logits = predict_logits(head, d, TEXT_KEY, True).clone()
        logits[:, NEUTRAL] -= neutral_offset
        return logits.argmax(1)
    dev_f1 = [float(f1_score(dev["y"], predict(h, dev), average="weighted")) for h in heads]
    return {"dev_weighted_f1": {"mean": float(np.mean(dev_f1)), "std": float(np.std(dev_f1))},
            "test": summarize([metrics(test["y"], predict(h, test)) for h in heads])}


def fmt(s: dict) -> str:
    return f"{s['mean']:.3f} ± {s['std']:.3f}"


def main():
    torch.set_num_threads(4)
    train, dev, test = (load_split(s) for s in ("train", "dev", "test"))
    results, heads = {}, {}

    print("baseline")
    heads["baseline"] = [fused(train, dev, s) for s in CFG.seeds]
    results["baseline"] = evaluate(heads["baseline"], dev, test)

    print("out-of-fold predictions on train (5 folds x 3 seeds)")
    wrong = {s: out_of_fold_wrong(train, dev, s) for s in CFG.seeds}
    for factor in (2, 3):
        print(f"hard_x{factor}")
        heads[f"hard_x{factor}"] = [fused(train, dev, s, example_weights=1.0 + (factor - 1) * wrong[s].float())
                                    for s in CFG.seeds]
        results[f"hard_x{factor}"] = evaluate(heads[f"hard_x{factor}"], dev, test)
    for gamma in (1, 2):
        print(f"focal_g{gamma}")
        heads[f"focal_g{gamma}"] = [fused(train, dev, s, focal_gamma=float(gamma)) for s in CFG.seeds]
        results[f"focal_g{gamma}"] = evaluate(heads[f"focal_g{gamma}"], dev, test)

    grid = np.round(np.arange(0.0, 2.01, 0.1), 2)     # tune the neutral offset on dev only
    dev_curve = [evaluate(heads["baseline"], dev, dev, off)["dev_weighted_f1"]["mean"] for off in grid]
    best_off = float(grid[int(np.argmax(dev_curve))])
    results["neutral_offset"] = {**evaluate(heads["baseline"], dev, test, best_off), "offset": best_off}

    # Label-noise probe.
    always_wrong = torch.stack([wrong[s] for s in CFG.seeds]).all(0).numpy()
    y = train["y"].numpy()
    probe = {"oof_error_rate_per_seed": {str(s): float(wrong[s].float().mean()) for s in CFG.seeds},
             "always_wrong_share": float(always_wrong.mean()),
             "always_wrong_by_class": {c: float(always_wrong[y == i].mean()) for i, c in enumerate(LABELS)}}
    rows = train["ids"].merge(load_manifest("train"), on=["dialogue_id", "utterance_id"], suffixes=("", "_m"))
    sample = rows[always_wrong].sample(n=20, random_state=0)
    L = ["# Train utterances the model gets wrong in every fold and seed", "",
         f"{always_wrong.sum():,d} of {len(y):,d} train utterances ({always_wrong.mean():.0%}), from "
         "`evaluation/10_train_on_failures.py`. A random 20 below. TODO(Shrish): listen to some and judge how "
         "many labels are defensible; that bounds how much training on failures can help.", "",
         "| Clip | Speaker | Transcript | Label |", "| --- | --- | --- | --- |"]
    L += [f"| {clip_id(r.dialogue_id, r.utterance_id)} | {r.speaker} | {r.text.replace('|', '/')} | {r.emotion} |"
          for r in sample.itertuples()]
    CFG.out("struggles", "hard_train_examples.md").write_text("\n".join(L) + "\n")

    write_json(CFG.out("experiments", "train_on_failures.json"),
               {"text_key": TEXT_KEY, "seeds": list(CFG.seeds), "variants": results, "label_noise_probe": probe,
                "neutral_offset_dev_curve": dict(zip(map(str, grid), dev_curve))})
    T = ["# Training harder on the failures", "",
         "Variants of the deployed fused model, 3 seeds each. Failures were found on train (out-of-fold); "
         "every choice was made on dev. From `evaluation/10_train_on_failures.py`.", "",
         "| Variant | Dev weighted F1 | Test weighted F1 | Test macro F1 | Test accuracy |",
         "| --- | ---: | ---: | ---: | ---: |"]
    for name, r in results.items():
        label = f"{name} (offset {r['offset']:.1f})" if "offset" in r else name
        T.append(f"| {label} | {fmt(r['dev_weighted_f1'])} | {fmt(r['test']['weighted_f1'])} | "
                 f"{fmt(r['test']['macro_f1'])} | {fmt(r['test']['accuracy'])} |")
    T += ["", "Per-emotion test F1 (mean over seeds):", "",
          "| Variant | " + " | ".join(LABELS) + " |", "| --- |" + " ---: |" * len(LABELS)]
    T += [f"| {name} | " + " | ".join(f"{r['test']['per_class'][c]['f1']:.3f}" for c in LABELS) + " |"
          for name, r in results.items()]
    T += ["", f"Label-noise probe: out-of-fold train error rate "
          f"{np.mean(list(probe['oof_error_rate_per_seed'].values())):.0%}; {probe['always_wrong_share']:.0%} "
          "of train is wrong in every fold of every seed. By emotion: "
          + ", ".join(f"{c} {v:.0%}" for c, v in probe["always_wrong_by_class"].items()) + "."]
    CFG.out("experiments", "train_on_failures.md").write_text("\n".join(T) + "\n")
    print("\n".join(T))


if __name__ == "__main__":
    main()
