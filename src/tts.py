"""Local text-to-speech: Kokoro-82M (hexgrad/Kokoro-82M, Apache-2.0).

Runs on the CPU on purpose: a sentence takes a few hundred milliseconds there, and
the GPU stays free for the LLM, which is still generating the next sentence while
this one is synthesized. Kokoro's English front end (misaki) needs the spaCy model
`en_core_web_sm`, which requirements.txt installs, so nothing downloads at run time.
"""
import io

import numpy as np
import soundfile as sf

from config import CFG

SAMPLE_RATE = 24000


class Speaker:
    def __init__(self, voice: str = CFG.tts_voice, device: str = "cpu"):
        from kokoro import KPipeline
        self.voice = voice
        self.pipe = KPipeline(lang_code="a", repo_id=CFG.tts_model, device=device)  # "a" = American English
        self.model = self.pipe.model

    def synth(self, text: str) -> np.ndarray:
        """Text -> float32 mono audio at 24 kHz."""
        parts = [r.audio.numpy() for r in self.pipe(text, voice=self.voice) if r.audio is not None]
        return np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, np.float32)


def wav_bytes(audio: np.ndarray) -> bytes:
    """16-bit PCM WAV, ready to send to a browser."""
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()
