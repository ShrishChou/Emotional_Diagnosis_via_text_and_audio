"""Single place for paths, model names, device choice and hyperparameters.

Every script imports `CFG` from here so there is one source of truth.
"""
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent

# Fixed label order used everywhere (ids 0..6).
LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_llm(device: str) -> str:
    """Hardware-dependent LLM choice (see the build plan, Phase 0)."""
    if device == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb >= 12:
            return "Qwen/Qwen2.5-3B-Instruct"
    return "Qwen/Qwen2.5-1.5B-Instruct"


@dataclass
class Config:
    # Paths
    root: Path = ROOT
    raw_dir: Path = ROOT / "data" / "raw"
    wav_dir: Path = ROOT / "data" / "wav"
    manifest_dir: Path = ROOT / "data" / "manifests"
    cache_dir: Path = ROOT / "cache"
    models_dir: Path = ROOT / "models"   # trained classifier heads + meta.json
    evals_dir: Path = ROOT / "evals"     # every result, chart and report (see evals/README.md)

    # Models (all run locally)
    text_model: str = "roberta-base"
    audio_model: str = "microsoft/wavlm-base-plus"
    asr_model: str = "openai/whisper-small"
    device: str = field(default_factory=pick_device)
    llm_model: str = ""  # filled in __post_init__ from the hardware

    # Audio
    sample_rate: int = 16000
    max_audio_s: float = 15.0

    # Text
    max_text_tokens: int = 256

    # Feature extraction
    # CPU-only machines extract audio features for a stratified train subsample.
    audio_train_subsample: int | None = None

    # Training
    hidden: int = 256
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 30
    patience: int = 5
    seeds: tuple = (0, 1, 2)

    # Reply generation: prompt + decoding settings live in src/responder.py STYLES.
    reply_style: str = "behavior"

    # Speech output (app/server.py; optional, the page can mute it)
    tts_model: str = "hexgrad/Kokoro-82M"
    tts_voice: str = "af_heart"

    def __post_init__(self):
        if not self.llm_model:
            self.llm_model = pick_llm(self.device)
        if self.device == "cpu" and self.audio_train_subsample is None:
            self.audio_train_subsample = 4000

    def out(self, *parts: str) -> Path:
        """A path under evals/, with its folder created: CFG.out("system", "latency.json")."""
        path = self.evals_dir.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def llm_dtype(self):
        # fp16 on GPU / MPS, fp32 on CPU (fp16 matmuls are slow or unsupported on CPU).
        return torch.float16 if self.device in ("cuda", "mps") else torch.float32


CFG = Config()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")
