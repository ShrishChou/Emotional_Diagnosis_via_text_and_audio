"""Small classification heads on top of the cached (frozen) embeddings.

One head class covers every configuration so the only thing that changes
between text-only, audio-only and fused is the input:

  audio [B, 13, 768] -> softmax-weighted sum over layers -> [B, 768]
  text  [B, 768]
  inputs present are concatenated -> LayerNorm -> Linear(256) -> GELU -> Dropout -> Linear(7)
"""
import json

import torch
import torch.nn as nn

from config import CFG, LABELS

N_AUDIO_LAYERS = 13
DIM = 768


class LayerWeights(nn.Module):
    """Learnable softmax weighting over WavLM's 13 hidden layers."""

    def __init__(self, n_layers: int = N_AUDIO_LAYERS):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_layers))  # starts as a plain average

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, L, D] -> [B, D]
        w = torch.softmax(self.logits, dim=0)
        return (x * w[None, :, None]).sum(1)


class EmotionHead(nn.Module):
    def __init__(self, use_text: bool, use_audio: bool,
                 hidden: int = CFG.hidden, dropout: float = CFG.dropout):
        super().__init__()
        assert use_text or use_audio
        self.use_text, self.use_audio = use_text, use_audio
        self.layer_weights = LayerWeights() if use_audio else None
        in_dim = DIM * (int(use_text) + int(use_audio))
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, len(LABELS)),
        )

    def forward(self, text: torch.Tensor | None = None, audio: torch.Tensor | None = None) -> torch.Tensor:
        parts = []
        if self.use_text:
            parts.append(text)
        if self.use_audio:
            parts.append(self.layer_weights(audio.float()))
        return self.mlp(torch.cat(parts, dim=-1))  # logits [B, 7]

    def config(self) -> dict:
        return {"use_text": self.use_text, "use_audio": self.use_audio,
                "hidden": self.mlp[1].out_features, "dropout": self.mlp[3].p}


def save_head(head: EmotionHead, path, extra: dict | None = None) -> None:
    """Save the architecture, the weights and any extra fields (text feature, seed, temperature)."""
    torch.save({"config": head.config(), "state_dict": head.state_dict(), **(extra or {})}, path)


def load_head(path, device: str = "cpu") -> tuple[EmotionHead, dict]:
    ckpt = torch.load(path, map_location=device)
    head = EmotionHead(**ckpt["config"]).to(device).eval()
    head.load_state_dict(ckpt["state_dict"])
    return head, ckpt


def load_deployed(device: str = "cpu") -> dict:
    """The two heads the app uses, as named in models/meta.json:
    {"text_only": (head, ckpt), "fused": (head, ckpt), "meta": {...}}.
    Each ckpt holds "text_key" (which text feature it reads) and "temperature"."""
    meta = json.loads((CFG.models_dir / "meta.json").read_text())
    out = {role: load_head(CFG.models_dir / name, device) for role, name in meta["deployed"].items()}
    out["meta"] = meta
    return out
