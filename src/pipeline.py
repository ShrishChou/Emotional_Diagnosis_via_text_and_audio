"""End to end: one utterance (audio + optional transcript) -> JSON state -> streamed reply.

Every model is loaded once in `Pipeline.__init__`; `cold_load_s` records how long
each took. `analyze` produces the JSON state; `respond` streams the reply.
"""
import time

import torch
import torch.nn.functional as F

from config import CFG, LABELS
from src.audio import emode_normalize, read_raw
from src.encoders import ASR, AudioEncoder, TextEncoder
from src.fusion import load_head
from src.reply_checks import sentence_ends
from src.responder import Responder

# Below this raw RMS (about -66 dBFS) there is no usable speech: skip Whisper. The quietest of
# 3,718 MELD dev + test clips is 0.0007, so real speech never trips it; a muted mic does.
SILENCE_RMS = 5e-4
NOT_HEARD_REPLY = "Sorry, I didn't catch that. Could you say it again?"


def _sync(device: str) -> None:
    """Wait for queued GPU work so wall-clock timings are honest."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


class Pipeline:
    def __init__(self, device: str = CFG.device, load_asr: bool = True, load_llm: bool = True,
                 load_tts: bool = False):
        self.device = device
        self.cold_load_s = {}

        def timed(name, fn):
            t0 = time.perf_counter()
            obj = fn()
            self.cold_load_s[name] = round(time.perf_counter() - t0, 2)
            return obj

        self.text_enc = timed("text_encoder", lambda: TextEncoder(device))
        self.audio_enc = timed("audio_encoder", lambda: AudioEncoder(device))
        self.text_head, text_ckpt = timed("text_head", lambda: load_head(CFG.ckpt_dir / "text_head.pt", device))
        self.fused_head, fused_ckpt = timed("fused_head", lambda: load_head(CFG.ckpt_dir / "fused_head.pt", device))
        self.text_key = fused_ckpt["text_key"]  # "text" or "text_ctx", chosen on dev
        assert text_ckpt["text_key"] == self.text_key
        self.temperature = fused_ckpt["temperature"]
        self.text_temperature = text_ckpt.get("temperature", 1.0)
        self.asr = timed("asr", lambda: ASR(device)) if load_asr else None
        self.responder = timed("llm", lambda: Responder(device=device)) if load_llm else None
        if load_tts:
            from src.tts import Speaker
            self.tts = timed("tts", Speaker)
        else:
            self.tts = None

    def warmup(self) -> float:
        """One throwaway pass (1 s of noise, a short reply) so GPU kernels are
        compiled before anything is timed. Returns its duration in seconds."""
        import tempfile

        import numpy as np
        import soundfile as sf
        t0 = time.perf_counter()
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            sf.write(f.name, np.random.default_rng(0).normal(0, 0.1, CFG.sample_rate).astype("float32"),
                     CFG.sample_rate)
            state = self.analyze(f.name, text=None if self.asr else "Hello there.")
        if self.responder:
            for _ in self.respond(state):
                pass
        if self.tts:
            self.tts.synth("Hello there.")  # the first synthesis is ~2x slower
        return time.perf_counter() - t0

    @torch.no_grad()
    def ping(self) -> float:
        """A tiny pass through each GPU model, to wake the GPU before a turn arrives.
        On Apple Silicon the first turn after a few seconds idle is several times slower
        (see scripts/12_idle_latency.py); the face page calls this when the person
        starts talking. Returns its duration in seconds."""
        t0 = time.perf_counter()
        self.audio_enc.encode([torch.zeros(CFG.sample_rate)])
        self.text_enc.encode(["hello"], [""] if self.text_key == "text_ctx" else None)
        if self.responder:
            r = self.responder
            r.model(**r.tok("hello", return_tensors="pt").to(self.device))
        _sync(self.device)
        return time.perf_counter() - t0

    @torch.no_grad()
    def analyze(self, wav_path, text: str | None = None, context: str = "",
                utterance_id: str | None = None, transcript_source: str | None = None) -> dict:
        """Audio (+ optional transcript) -> JSON-serializable emotion state.

        Timing starts when the finished utterance is handed over (end of speech).
        """
        lat = {}
        t_start = time.perf_counter()

        t0 = time.perf_counter()
        raw, _ = read_raw(wav_path)
        wav = torch.from_numpy(emode_normalize(raw))
        lat["audio_load"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        if text is None:
            assert self.asr is not None, "no transcript given and ASR not loaded"
            silent = len(raw) == 0 or float((raw ** 2).mean()) ** 0.5 < SILENCE_RMS
            text = "" if silent else self.asr.transcribe(raw)
            transcript_source = "asr"
        else:
            transcript_source = transcript_source or "user"
        lat["asr"] = (time.perf_counter() - t0) * 1000
        if not text.strip():
            # An empty transcript is far outside training (RoBERTa on "" gave a confident,
            # arbitrary emotion), so report that nothing was heard instead of guessing.
            return self._nothing_heard(utterance_id, transcript_source, lat, t_start)

        t0 = time.perf_counter()
        ctx = [context] if self.text_key == "text_ctx" else None
        text_vec = self.text_enc.encode([text], ctx).to(self.device)
        lat["text_enc"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        audio_vec = self.audio_enc.encode([wav]).to(self.device)
        lat["audio_enc"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        text_logits = self.text_head(text=text_vec)
        fused_logits = self.fused_head(text=text_vec, audio=audio_vec)
        probs = F.softmax(fused_logits / self.temperature, dim=-1)[0].cpu()
        text_pred = int(text_logits.argmax(-1).item())
        _sync(self.device)
        lat["heads"] = (time.perf_counter() - t0) * 1000
        lat["state_total"] = (time.perf_counter() - t_start) * 1000

        pred = int(probs.argmax())
        return {
            "utterance_id": utterance_id,
            "transcript": text,
            "transcript_source": transcript_source,
            "emotion": LABELS[pred],
            "probs": {c: round(float(p), 4) for c, p in zip(LABELS, probs)},
            "confidence": round(float(probs[pred]), 4),
            "text_only_emotion": LABELS[text_pred],
            "audio_changed_prediction": pred != text_pred,
            "modalities": ["text", "audio"],
            "heard": True,
            "latency_ms": {k: round(v, 1) for k, v in lat.items()},
        }

    @torch.no_grad()
    def analyze_text(self, text: str, context: str = "", utterance_id: str | None = None) -> dict:
        """A typed message (no audio): the text-only head decides, with its own
        dev-fit temperature. Same state schema as `analyze`; the audio fields say
        that no audio was used."""
        lat = {}
        t_start = time.perf_counter()
        if not text.strip():
            return self._nothing_heard(utterance_id, "typed", lat, t_start)
        t0 = time.perf_counter()
        ctx = [context] if self.text_key == "text_ctx" else None
        text_vec = self.text_enc.encode([text], ctx).to(self.device)
        lat["text_enc"] = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        probs = F.softmax(self.text_head(text=text_vec) / self.text_temperature, dim=-1)[0].cpu()
        _sync(self.device)
        lat["heads"] = (time.perf_counter() - t0) * 1000
        lat["state_total"] = (time.perf_counter() - t_start) * 1000
        pred = int(probs.argmax())
        return {
            "utterance_id": utterance_id,
            "transcript": text,
            "transcript_source": "typed",
            "emotion": LABELS[pred],
            "probs": {c: round(float(p), 4) for c, p in zip(LABELS, probs)},
            "confidence": round(float(probs[pred]), 4),
            "text_only_emotion": LABELS[pred],
            "audio_changed_prediction": False,
            "modalities": ["text"],
            "heard": True,
            "latency_ms": {k: round(v, 1) for k, v in lat.items()},
        }

    def _nothing_heard(self, utterance_id, transcript_source, lat, t_start) -> dict:
        """State for silence or an empty transcript: uniform probabilities, so the face
        stays neutral, and `heard: false`, so the reply asks the person to repeat."""
        lat["state_total"] = (time.perf_counter() - t_start) * 1000
        return {
            "utterance_id": utterance_id,
            "transcript": "",
            "transcript_source": transcript_source,
            "emotion": "neutral",
            "probs": {c: round(1 / len(LABELS), 4) for c in LABELS},
            "confidence": round(1 / len(LABELS), 4),
            "text_only_emotion": "neutral",
            "audio_changed_prediction": False,
            "modalities": [],
            "heard": False,
            "latency_ms": {k: round(v, 1) for k, v in lat.items()},
        }

    def respond(self, state: dict, context: str = "", use_emotion: bool = True, seed: int = 0,
                style: str = CFG.reply_style):
        """Stream the reply; fills state['latency_ms'] llm_first_token / llm_total as it goes."""
        if not state.get("heard", True):  # fixed reply, no LLM call
            state["latency_ms"].update(llm_first_token=0.0, llm_total=0.0)
            yield NOT_HEARD_REPLY
            return
        t0 = time.perf_counter()
        first = None
        for chunk in self.responder.stream(state, context, use_emotion, seed, style):
            if first is None:
                first = (time.perf_counter() - t0) * 1000
                state["latency_ms"]["llm_first_token"] = round(first, 1)
            yield chunk
        state["latency_ms"]["llm_total"] = round((time.perf_counter() - t0) * 1000, 1)

    def respond_spoken(self, state: dict, context: str = "", seed: int = 0):
        """`respond` plus speech: yields {"type": "token"} events as text streams and an
        {"type": "audio"} event (sentence text + 24 kHz WAV bytes) as soon as each
        sentence is complete, so speaking starts while the LLM writes the next one.
        Adds latency_ms["tts_first_audio"] (from the start of the reply to the first
        audio being ready) and ["tts_total"] (time spent synthesizing)."""
        from src.tts import SAMPLE_RATE, wav_bytes
        t0 = time.perf_counter()
        text, spoken, synth_ms, first_audio = "", 0, 0.0, None

        def speak(upto: int):
            nonlocal spoken, synth_ms, first_audio
            sentence = text[spoken:upto].strip()
            spoken = upto
            if not sentence:
                return None
            ts = time.perf_counter()
            audio = self.tts.synth(sentence)
            synth_ms += (time.perf_counter() - ts) * 1000
            if first_audio is None:
                first_audio = (time.perf_counter() - t0) * 1000
            return {"type": "audio", "text": sentence, "wav": wav_bytes(audio),
                    "duration_s": round(len(audio) / SAMPLE_RATE, 3)}

        for chunk in self.respond(state, context, seed=seed):
            yield {"type": "token", "text": chunk}
            text += chunk
            ends = [e for e in sentence_ends(text) if e > spoken]
            if ends and (ev := speak(ends[-1])):
                yield ev
        if ev := speak(len(text)):  # whatever is left without a final terminator
            yield ev
        state["latency_ms"]["tts_first_audio"] = round(first_audio or 0.0, 1)
        state["latency_ms"]["tts_total"] = round(synth_ms, 1)
