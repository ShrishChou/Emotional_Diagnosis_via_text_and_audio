"""Step 4: train the emotion classifiers on the cached embeddings (about a minute, CPU).

Every classifier is the same small head (src/fusion.py); only its input changes:

  text       the utterance (RoBERTa)
  text_ctx   the previous two lines + the utterance (RoBERTa)
  audio      the voice (WavLM, learned weighting over its 13 layers)
  fused      the better text variant on dev + audio          <- what the app uses for voice
  fused_utt  utterance-only text + audio (trained only when text_ctx wins on dev)

Two more models need no head training:
  late_fusion  the average of the text and audio heads' probabilities
  logreg       logistic regression on the utterance embedding (sanity baseline)

Each head is trained with 3 seeds. Everything that is chosen (the early-stopping
epoch, the text variant, the saved seed, the temperatures) is chosen on dev.
Test predictions are saved but not scored here: evaluation/01_compare_classifiers.py
turns them into metrics, charts and confusion matrices.

Writes
  models/<name>.pt                          best-on-dev seed of every head
  models/meta.json                          which heads the app uses, and their temperatures
  evals/classifiers/predictions.npz         dev and test probabilities, every model and seed
  evals/classifiers/training_summary.json   dev scores and best epochs per seed
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from config import CFG, write_json
from src.fusion import save_head
from src.training import ece, fit_temperature, load_split, predict_logits, train_head


def train_seeds(name: str, text_key: str | None, use_audio: bool, train: dict, dev: dict, test: dict) -> dict:
    """Train one configuration with every seed; keep each seed's head and predictions."""
    runs = []
    for seed in CFG.seeds:
        head, dev_f1, epoch = train_head(train, dev, text_key, use_audio, seed)
        runs.append({"seed": seed, "head": head, "dev_f1": dev_f1, "epoch": epoch,
                     "dev_logits": predict_logits(head, dev, text_key, use_audio),
                     "test_logits": predict_logits(head, test, text_key, use_audio)})
        print(f"  {name:10s} seed {seed}: dev weighted F1 {dev_f1:.3f} (best epoch {epoch})")
    return {"text_key": text_key, "use_audio": use_audio, "runs": runs,
            "best": int(np.argmax([r["dev_f1"] for r in runs]))}


def calibrate(logits_dev: torch.Tensor, y_dev: torch.Tensor) -> dict:
    """Fit a temperature on dev and report dev ECE before and after."""
    t = fit_temperature(logits_dev, y_dev)
    y = y_dev.numpy()
    return {"temperature": t,
            "dev_ece_before": ece(F.softmax(logits_dev, 1).numpy(), y),
            "dev_ece_after": ece(F.softmax(logits_dev / t, 1).numpy(), y)}


def main():
    torch.set_num_threads(8)
    train, dev, test = (load_split(s) for s in ("train", "dev", "test"))
    print(f"rows: train {len(train['y'])}, dev {len(dev['y'])}, test {len(test['y'])}")

    # 1. Single-modality heads.
    models = {name: train_seeds(name, key, audio, train, dev, test)
              for name, key, audio in [("text", "text", False), ("text_ctx", "text_ctx", False),
                                       ("audio", None, True)]}

    # 2. The fused model reads the text variant that is better on dev (never on test).
    text_key = max(["text", "text_ctx"], key=lambda k: np.mean([r["dev_f1"] for r in models[k]["runs"]]))
    print(f"better text variant on dev: {text_key}")
    models["fused"] = train_seeds("fused", text_key, True, train, dev, test)
    if text_key != "text":  # what audio adds without dialogue context; reported, never selected
        models["fused_utt"] = train_seeds("fused_utt", "text", True, train, dev, test)

    # 3. Save every head's best-on-dev seed. The two the app uses also get a temperature.
    CFG.models_dir.mkdir(exist_ok=True)
    calibration = {
        "fused": calibrate(models["fused"]["runs"][models["fused"]["best"]]["dev_logits"], dev["y"]),
        "text_only": calibrate(models[text_key]["runs"][models[text_key]["best"]]["dev_logits"], dev["y"]),
    }
    for name, m in models.items():
        best = m["runs"][m["best"]]
        extra = {"text_key": m["text_key"], "seed": best["seed"], "dev_weighted_f1": best["dev_f1"]}
        if name == "fused":
            extra["temperature"] = calibration["fused"]["temperature"]
        if name == text_key:
            extra["temperature"] = calibration["text_only"]["temperature"]
        save_head(best["head"], CFG.models_dir / f"{name}.pt", extra)
    fused_head = models["fused"]["runs"][models["fused"]["best"]]["head"]
    write_json(CFG.models_dir / "meta.json", {
        "deployed": {"fused": "fused.pt", "text_only": f"{text_key}.pt"},
        "text_key": text_key,
        "temperatures": {k: v["temperature"] for k, v in calibration.items()},
        "saved_seed": {name: m["runs"][m["best"]]["seed"] for name, m in models.items()},
        "params": {name: sum(p.numel() for p in m["runs"][0]["head"].parameters()) for name, m in models.items()},
        "audio_layer_weights_fused": torch.softmax(fused_head.layer_weights.logits, 0).tolist(),
    })

    # 4. Probabilities for the evaluation: every head and seed, plus late fusion and logreg.
    pred = {"y_dev": dev["y"].numpy(), "y_test": test["y"].numpy()}
    for name, m in models.items():
        for split in ("dev", "test"):
            pred[f"{name}__{split}"] = np.stack([F.softmax(r[f"{split}_logits"], 1).numpy() for r in m["runs"]])
    for split in ("dev", "test"):
        pred[f"late_fusion__{split}"] = (pred[f"{text_key}__{split}"] + pred[f"audio__{split}"]) / 2
    scaler = StandardScaler().fit(train["text"].numpy())
    logreg = LogisticRegression(max_iter=2000).fit(scaler.transform(train["text"].numpy()), train["y"].numpy())
    for split, d in (("dev", dev), ("test", test)):
        pred[f"logreg__{split}"] = logreg.predict_proba(scaler.transform(d["text"].numpy()))[None]
    out = CFG.out("classifiers", "predictions.npz")
    np.savez_compressed(out, **{k: v.astype(np.float32) if v.dtype.kind == "f" else v for k, v in pred.items()})

    late_dev = [float(f1_score(dev["y"], p.argmax(1), average="weighted")) for p in pred["late_fusion__dev"]]
    write_json(CFG.out("classifiers", "training_summary.json"), {
        "seeds": list(CFG.seeds),
        "selected_text_variant": text_key,
        "models": {
            **{name: {"input": {"text": m["text_key"], "audio": m["use_audio"]},
                      "dev_weighted_f1_per_seed": [r["dev_f1"] for r in m["runs"]],
                      "best_epoch_per_seed": [r["epoch"] for r in m["runs"]],
                      "saved_seed": m["runs"][m["best"]]["seed"]} for name, m in models.items()},
            "late_fusion": {"input": f"mean of {text_key} and audio probabilities",
                            "dev_weighted_f1_per_seed": late_dev},
            "logreg": {"input": "utterance embedding, sklearn LogisticRegression"},
        },
        "calibration_dev": calibration,
    })
    t = calibration
    print(f"temperatures: fused {t['fused']['temperature']:.3f} (dev ECE {t['fused']['dev_ece_before']:.3f} -> "
          f"{t['fused']['dev_ece_after']:.3f}), text-only {t['text_only']['temperature']:.3f}")
    print(f"saved {len(models)} heads to models/, predictions to {out.relative_to(CFG.root)}")


if __name__ == "__main__":
    main()
