"""Frozen pretrained encoders: RoBERTa (text), WavLM (audio), Whisper (ASR).

All run locally in eval mode with gradients off. None of them was trained on
MELD or on any emotion labels (they are general-purpose checkpoints).
"""
import torch
from transformers import AutoModel, AutoTokenizer, WavLMModel, WhisperForConditionalGeneration, WhisperProcessor

from config import CFG

MIN_SAMPLES = 1600  # 0.1 s; WavLM's conv stack needs at least ~400 samples
MAX_UTT_TOKENS = 192  # cap on the utterance inside the 256-token context window


def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average `hidden` [B, T, D] over time, ignoring positions where mask == 0."""
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)


class TextEncoder:
    """roberta-base, mean-pooled last hidden state -> [B, 768]."""

    def __init__(self, device: str = CFG.device):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(CFG.text_model)
        # Truncate from the left so the *current* utterance (at the end) is always kept.
        self.tok.truncation_side = "left"
        # No pooler: we mean-pool the last hidden state ourselves (the pooler would be untrained).
        self.model = AutoModel.from_pretrained(CFG.text_model, add_pooling_layer=False).to(device).eval()

    def _tail(self, text: str, n: int) -> str:
        ids = self.tok.encode(text, add_special_tokens=False)
        return text if len(ids) <= n else self.tok.decode(ids[-n:])

    @torch.no_grad()
    def encode(self, texts: list[str], contexts: list[str] | None = None) -> torch.Tensor:
        """`texts` alone -> the `text` feature. With `contexts`, encodes
        "context </s></s> utterance" (RoBERTa's pair format) and pools over the
        utterance tokens only -> the `text_ctx` feature."""
        if contexts is None:
            batch = self.tok(texts, padding=True, truncation=True,
                             max_length=CFG.max_text_tokens, return_tensors="pt")
        else:
            # Keep at most the last MAX_UTT_TOKENS of the utterance itself, so the
            # context can always absorb the truncation (a runaway ASR transcript would
            # otherwise exceed the budget on its own). No gold MELD utterance is this long.
            texts = [self._tail(t, MAX_UTT_TOKENS) for t in texts]
            batch = self.tok(contexts, texts, padding=True, truncation="only_first",
                             max_length=CFG.max_text_tokens, return_tensors="pt")
        pool_mask = batch["attention_mask"].clone()
        if contexts is not None:
            # Pool over the current utterance's tokens only. The context still shapes
            # them through self-attention, but does not swamp the average (pooling over
            # every token did much worse on dev; see NOTES.md).
            for i in range(len(texts)):
                seq_ids = batch.sequence_ids(i)
                pool_mask[i] = torch.tensor([int(s == 1) for s in seq_ids])
        batch = {k: v.to(self.device) for k, v in batch.items()}
        out = self.model(**batch)
        return _mean_pool(out.last_hidden_state, pool_mask.to(self.device)).float().cpu()


class AudioEncoder:
    """wavlm-base-plus, every hidden layer mean-pooled over time -> [B, 13, 768].

    Expects waveforms already EMODE-normalized (see src/audio.py); WavLM's own
    feature extractor has do_normalize=False, so no second normalization happens.
    """

    def __init__(self, device: str = CFG.device):
        self.device = device
        self.model = WavLMModel.from_pretrained(CFG.audio_model).to(device).eval()

    @torch.no_grad()
    def encode(self, wavs: list[torch.Tensor]) -> torch.Tensor:
        lengths = torch.tensor([max(len(w), MIN_SAMPLES) for w in wavs])
        batch = torch.zeros(len(wavs), int(lengths.max()))
        for i, w in enumerate(wavs):
            batch[i, :len(w)] = w
        sample_mask = (torch.arange(batch.shape[1])[None, :] < lengths[:, None]).long()

        out = self.model(batch.to(self.device), attention_mask=sample_mask.to(self.device),
                         output_hidden_states=True)
        # Frame-level mask: how many output frames each clip's real samples produce.
        n_frames = self.model._get_feat_extract_output_lengths(lengths).to(self.device)
        T = out.hidden_states[0].shape[1]
        frame_mask = torch.arange(T, device=self.device)[None, :] < n_frames[:, None]
        pooled = [_mean_pool(h, frame_mask) for h in out.hidden_states]  # 13 x [B, 768]
        return torch.stack(pooled, dim=1).float().cpu()


class ASR:
    """whisper-small, English transcription of one 16 kHz clip."""

    def __init__(self, device: str = CFG.device):
        self.device = device
        self.dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        self.proc = WhisperProcessor.from_pretrained(CFG.asr_model)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            CFG.asr_model, dtype=self.dtype).to(device).eval()

    @torch.no_grad()
    def transcribe(self, raw_wav) -> str:
        """`raw_wav`: 16 kHz mono numpy array *before* EMODE normalization
        (Whisper computes its own log-mel features)."""
        feats = self.proc(raw_wav, sampling_rate=CFG.sample_rate, return_tensors="pt").input_features
        ids = self.model.generate(feats.to(self.device, self.dtype), language="en", task="transcribe")
        return self.proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

    @torch.no_grad()
    def transcribe_batch(self, raw_wavs: list) -> list[str]:
        """Several raw 16 kHz clips at once (used to transcribe whole MELD splits)."""
        feats = self.proc(raw_wavs, sampling_rate=CFG.sample_rate, return_tensors="pt").input_features
        ids = self.model.generate(feats.to(self.device, self.dtype), language="en", task="transcribe")
        return [t.strip() for t in self.proc.batch_decode(ids, skip_special_tokens=True)]
