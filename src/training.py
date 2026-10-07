"""Training and scoring helpers shared by the training script and the experiments.

Everything here works on the cached embeddings (cache/{split}_{feature}.npy), so a
head trains in seconds on the CPU.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from config import CFG, LABELS, set_seed
from src.fusion import EmotionHead

SPLITS = ("train", "dev", "test")
FEATURES = ("text", "text_ctx", "audio")


# ---------------------------------------------------------------- data

def load_split(split: str) -> dict:
    """Cached embeddings of one split plus its labels and ids (row-aligned)."""
    d = {name: torch.from_numpy(np.load(CFG.cache_dir / f"{split}_{name}.npy").astype(np.float32))
         for name in FEATURES}
    ids = pd.read_csv(CFG.cache_dir / f"{split}_ids.csv")
    d["y"] = torch.tensor(ids["label_id"].to_numpy())
    d["ids"] = ids
    return d


def model_inputs(d: dict, text_key: str | None, use_audio: bool, idx=slice(None)) -> dict:
    """Keyword arguments for EmotionHead.forward: the chosen text feature and/or audio."""
    return {"text": d[text_key][idx] if text_key else None,
            "audio": d["audio"][idx] if use_audio else None}


# ---------------------------------------------------------------- training

@torch.no_grad()
def predict_logits(head: EmotionHead, d: dict, text_key: str | None, use_audio: bool,
                   batch: int = 1024) -> torch.Tensor:
    head.eval()
    out = [head(**model_inputs(d, text_key, use_audio, slice(i, i + batch)))
           for i in range(0, len(d["y"]), batch)]
    return torch.cat(out)


def train_head(train: dict, dev: dict, text_key: str | None, use_audio: bool, seed: int,
               example_weights: torch.Tensor | None = None, focal_gamma: float = 0.0):
    """Train one head: AdamW, cross-entropy weighted by 1/sqrt(class frequency),
    early stopping on dev weighted F1 (the best epoch's weights are restored).

    `example_weights` and `focal_gamma` are only used by the train-on-failures
    experiment; with their defaults this is plain class-weighted cross-entropy.
    Returns (head, best dev weighted F1, best epoch).
    """
    set_seed(seed)
    head = EmotionHead(use_text=text_key is not None, use_audio=use_audio)
    opt = torch.optim.AdamW(head.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)

    counts = torch.bincount(train["y"], minlength=len(LABELS)).float()
    class_w = 1.0 / counts.sqrt()
    class_w = class_w / class_w.mean()
    plain = example_weights is None and not focal_gamma

    best = {"f1": -1.0, "epoch": -1, "state": None}
    gen = torch.Generator().manual_seed(seed)
    for epoch in range(CFG.max_epochs):
        head.train()
        for idx in torch.randperm(len(train["y"]), generator=gen).split(CFG.batch_size):
            logits = head(**model_inputs(train, text_key, use_audio, idx))
            y = train["y"][idx]
            if plain:
                loss = F.cross_entropy(logits, y, weight=class_w)
            else:
                ce = F.cross_entropy(logits, y, reduction="none")
                if focal_gamma:
                    ce = (1 - torch.exp(-ce)) ** focal_gamma * ce   # down-weight easy examples
                w = class_w[y] * (example_weights[idx] if example_weights is not None else 1.0)
                loss = (ce * w).sum() / w.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        f1 = f1_score(dev["y"], predict_logits(head, dev, text_key, use_audio).argmax(1), average="weighted")
        if f1 > best["f1"]:
            best = {"f1": float(f1), "epoch": epoch, "state": {k: v.clone() for k, v in head.state_dict().items()}}
        elif epoch - best["epoch"] >= CFG.patience:
            break
    head.load_state_dict(best["state"])
    head.eval()
    return head, best["f1"], best["epoch"]


def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Temperature scaling: one scalar T minimizing the NLL of softmax(logits / T)."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp())


# ---------------------------------------------------------------- scoring

def metrics(y_true, y_pred) -> dict:
    """Accuracy, macro and weighted precision / recall / F1, and per-class scores."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = range(len(LABELS))
    p, r, f, n = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    out = {"accuracy": float(accuracy_score(y_true, y_pred))}
    for avg in ("macro", "weighted"):
        ap, ar, af, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=avg,
                                                        zero_division=0)
        out.update({f"{avg}_precision": float(ap), f"{avg}_recall": float(ar), f"{avg}_f1": float(af)})
    out["per_class"] = {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                            "support": int(n[i])} for i, c in enumerate(LABELS)}
    return out


def summarize(runs: list[dict]) -> dict:
    """Mean and std over seeds of every scalar metric, and mean per-class scores."""
    out = {k: {"mean": float(np.mean([r[k] for r in runs])), "std": float(np.std([r[k] for r in runs]))}
           for k in runs[0] if k != "per_class"}
    out["per_class"] = {c: {m: float(np.mean([r["per_class"][c][m] for r in runs]))
                            for m in ("precision", "recall", "f1")} for c in LABELS}
    return out


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


def softmax_np(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = x / temperature
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)
