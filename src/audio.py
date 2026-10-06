"""Audio loading with the EMODE normalization.

The normalization is reused from EMODE (github.com/ShrishChou/EMODE, dataset.py):
downmix to mono, zero NaNs, subtract the mean, divide by the std (floored at
1e-3), then clamp to +-5.
"""
import numpy as np
import soundfile as sf
import torch

from config import CFG


def emode_normalize(wav: np.ndarray) -> np.ndarray:
    """EMODE normalization. `wav` is [samples] or [samples, channels]."""
    if wav.ndim == 2:
        wav = wav.mean(axis=1)  # downmix to mono
    wav = np.nan_to_num(wav.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    wav = wav - wav.mean()
    wav = wav / max(float(wav.std()), 1e-3)
    return np.clip(wav, -5.0, 5.0)


def read_raw(path, max_s: float = CFG.max_audio_s) -> tuple[np.ndarray, bool]:
    """Read audio as 16 kHz mono float32, cropped to its first `max_s` seconds.

    Returns (raw waveform, was_cropped). No normalization (Whisper wants raw audio).
    """
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != CFG.sample_rate:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=CFG.sample_rate)
    max_len = int(max_s * CFG.sample_rate)
    cropped = len(wav) > max_len
    return wav[:max_len], cropped


def load_wav(path, max_s: float = CFG.max_audio_s) -> tuple[torch.Tensor, bool]:
    """Read, crop to `max_s` seconds, then EMODE-normalize (crop first, so the
    statistics come from the audio the model actually sees).

    Returns (waveform [samples] float32 tensor, was_cropped).
    """
    wav, cropped = read_raw(path, max_s)
    return torch.from_numpy(emode_normalize(wav)), cropped
