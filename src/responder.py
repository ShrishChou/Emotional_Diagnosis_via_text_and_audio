"""Local reply LLM (Qwen Instruct): prompt construction, decoding constraints, streaming.

Two reply styles (compared in evaluation/07_compare_reply_models.py):

  baseline  the first version: the emotion label and confidence go into the prompt,
            with an instruction not to name the emotion. Sampling T=0.7, 60 tokens.
  behavior  the label never reaches the LLM. It is translated into tone guidance
            ("stay calm, acknowledge the problem, offer to help"), three short
            example exchanges show the format, and two rules are enforced in code
            rather than asked for: generation stops after the second sentence, and
            a list of emotion words is blocked (src/reply_checks.py). Sampling T=0.4, 40 tokens.
"""
from dataclasses import dataclass
from threading import Thread

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

from config import CFG
from src.reply_checks import BANNED_EMOTION_WORDS, sentence_ends


@dataclass(frozen=True)
class ReplyStyle:
    name: str
    prompt: str            # "label" or "behavior"
    few_shot: bool
    max_sentences: int | None
    ban_emotion_words: bool
    temperature: float
    top_p: float
    max_new_tokens: int


STYLES = {
    "baseline": ReplyStyle("baseline", "label", few_shot=False, max_sentences=None,
                           ban_emotion_words=False, temperature=0.7, top_p=0.9, max_new_tokens=60),
    "behavior": ReplyStyle("behavior", "behavior", few_shot=True, max_sentences=2,
                           ban_emotion_words=True, temperature=0.4, top_p=0.9, max_new_tokens=40),
}

# ------------------------------------------------------------------ baseline prompt

LABEL_SYSTEM_PROMPT = (
    "You are a warm, concise character robot in a conversation. Speak directly to the "
    "person who just spoke, as 'you'. Reply in one or two short sentences, never more. "
    "Respond to what they actually said. Let how they seem to feel shape your tone, but "
    "never name the emotion, never mention that you detected it, and do not describe "
    "the conversation from the outside."
)


def label_user_message(state: dict, context: str = "", use_emotion: bool = True) -> str:
    lines = []
    if context:
        lines.append(f"Earlier in the conversation:\n{context}")
    lines.append(f'The person talking to you just said: "{state["transcript"]}"')
    if use_emotion:
        lines.append(f'Detected emotion: {state["emotion"]} (confidence {state["confidence"]:.2f}).')
        if state.get("audio_changed_prediction"):
            lines.append(f'Their tone of voice disagreed with their words: the words alone '
                         f'suggest {state["text_only_emotion"]}.')
        if state["confidence"] < 0.5:
            lines.append("The emotion estimate is uncertain: respond gently and do not "
                         "assume how they feel.")
    lines.append("Reply to them now, in one or two short sentences.")
    return "\n".join(lines)

# ------------------------------------------------------------------ behavior prompt

BEHAVIOR_SYSTEM_PROMPT = (
    "You are a warm, concise character robot in a conversation. Speak directly to the "
    "person who just spoke, as 'you'. Reply in one or two short sentences. Respond to what "
    "they actually said, and follow the tone guidance when there is one. Do not tell them "
    "how they feel."
)

# Each emotion becomes a way of responding; the label word itself never reaches the LLM.
TONE_GUIDANCE = {
    "anger": "stay calm and steady, acknowledge the problem plainly, and offer to help.",
    "disgust": "stay matter-of-fact and non-judgmental; acknowledge their reaction lightly and don't dwell on details.",
    "fear": "be reassuring and steady; keep it simple and offer support.",
    "joy": "match their warmth and energy; share in it and show interest.",
    "neutral": "be friendly and natural; respond to the content directly.",
    "sadness": "be gentle and keep it short; don't rush to fix things, show you're listening.",
    "surprise": "engage with what caught them off guard; show curiosity.",
}
UNCERTAIN_GUIDANCE = "their mood is unclear, so keep a gentle, even tone and don't assume how they feel."
TONE_MISMATCH = "Their tone of voice says more than their words; respond to how they said it, not only what they said."


def behavior_user_message(state: dict, context: str = "", use_emotion: bool = True) -> str:
    lines = []
    if context:
        lines.append(f"Earlier in the conversation:\n{context}")
    lines.append(f'The person talking to you just said: "{state["transcript"]}"')
    if use_emotion:
        guidance = UNCERTAIN_GUIDANCE if state["confidence"] < 0.5 else TONE_GUIDANCE[state["emotion"]]
        lines.append(f"Tone guidance: {guidance}")
        if state.get("audio_changed_prediction"):
            lines.append(TONE_MISMATCH)
    lines.append("Reply to them now.")
    return "\n".join(lines)


# Invented exchanges (not from MELD): two sentences each, no emotion words.
FEW_SHOT = [
    ({"transcript": "I've been on hold with the bank for two hours and they just hung up on me.",
      "emotion": "anger", "confidence": 0.8},
     "Two hours and then a dropped call is a lot to put up with. "
     "Want me to help you find a direct number to call back?"),
    ({"transcript": "We got the apartment! They called this morning.", "emotion": "joy", "confidence": 0.9},
     "That's fantastic news, congratulations! When do you get the keys?"),
    ({"transcript": "I don't really feel like going out tonight.", "emotion": "sadness", "confidence": 0.7},
     "That's okay, we can keep tonight quiet. I'm here if you want to talk."),
]


def build_messages(state: dict, context: str, use_emotion: bool, style: ReplyStyle) -> list[dict]:
    if style.prompt == "label":
        system, user_fn = LABEL_SYSTEM_PROMPT, label_user_message
    else:
        system, user_fn = BEHAVIOR_SYSTEM_PROMPT, behavior_user_message
    messages = [{"role": "system", "content": system}]
    if style.few_shot:
        for ex_state, reply in FEW_SHOT:
            messages.append({"role": "user", "content": user_fn(ex_state, "", use_emotion)})
            messages.append({"role": "assistant", "content": reply})
    messages.append({"role": "user", "content": user_fn(state, context, use_emotion)})
    return messages

# ------------------------------------------------------------------ decoding constraints


class SentenceLimit(StoppingCriteria):
    """Stop once the generated text holds `n` finished sentences (works while streaming)."""

    def __init__(self, tok, prompt_len: int, n: int):
        self.tok, self.prompt_len, self.n = tok, prompt_len, n

    def __call__(self, input_ids, scores, **kwargs) -> torch.BoolTensor:
        text = self.tok.decode(input_ids[0, self.prompt_len:], skip_special_tokens=True)
        done = len(sentence_ends(text)) >= self.n
        return torch.full((input_ids.shape[0],), done, dtype=torch.bool, device=input_ids.device)


def banned_word_ids(tok) -> list[list[int]]:
    """Token sequences for every banned word, with and without a leading space and
    capitalized, since each spelling tokenizes differently."""
    seqs = set()
    for word in BANNED_EMOTION_WORDS:
        for form in (word, word.capitalize(), word.upper()):
            for prefix in ("", " "):
                ids = tok.encode(prefix + form, add_special_tokens=False)
                if ids:
                    seqs.add(tuple(ids))
    return [list(s) for s in sorted(seqs)]


class BannedWords(LogitsProcessor):
    """Blocks token sequences (like transformers' `bad_words_ids`) with one CPU
    copy of the recent tokens and one masking op per step. The built-in processor
    issues many small GPU ops per step and doubled generation time on MPS."""

    def __init__(self, sequences: list[list[int]]):
        self.single = sorted({s[0] for s in sequences if len(s) == 1})
        self.multi = [s for s in sequences if len(s) > 1]
        self.lookback = max((len(s) - 1 for s in self.multi), default=0)

    def __call__(self, input_ids, scores):
        for row in range(input_ids.shape[0]):
            banned = list(self.single)
            if self.lookback:
                tail = input_ids[row, -self.lookback:].tolist()
                for seq in self.multi:
                    prefix = seq[:-1]
                    if tail[len(tail) - len(prefix):] == prefix:
                        banned.append(seq[-1])
            scores[row, banned] = float("-inf")
        return scores


class Responder:
    def __init__(self, model_name: str = CFG.llm_model, device: str = CFG.device):
        self.device = device
        self.model_name = model_name
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=CFG.llm_dtype).to(device).eval()
        self._banned = BannedWords(banned_word_ids(self.tok))

    def stream(self, state: dict, context: str = "", use_emotion: bool = True,
               seed: int = 0, style: str = CFG.reply_style):
        """Yield the reply chunk by chunk as it is generated."""
        st = STYLES[style]
        messages = build_messages(state, context, use_emotion, st)
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch = self.tok(prompt, return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(**batch, streamer=streamer, max_new_tokens=st.max_new_tokens,
                      do_sample=True, temperature=st.temperature, top_p=st.top_p,
                      pad_token_id=self.tok.eos_token_id)
        if st.max_sentences:
            kwargs["stopping_criteria"] = StoppingCriteriaList(
                [SentenceLimit(self.tok, batch["input_ids"].shape[1], st.max_sentences)])
        if st.ban_emotion_words:
            kwargs["logits_processor"] = LogitsProcessorList([self._banned])
        torch.manual_seed(seed)
        thread = Thread(target=self.model.generate, kwargs=kwargs)
        thread.start()
        for chunk in streamer:
            if chunk:
                yield chunk
        thread.join()
