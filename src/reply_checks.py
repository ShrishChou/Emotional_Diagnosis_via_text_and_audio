"""Text checks on generated replies, shared by the decoding constraints and the
compliance metrics so both use exactly the same definitions."""
import re

# Words the reply must not use (blocked at decoding time in the "behavior" style).
# The negative MELD labels and their common variants, plus near-synonyms a small
# model reaches for when it announces a feeling.
BANNED_EMOTION_WORDS = [
    "anger", "angry", "angrily", "frustrated", "frustrating", "frustration",
    "upset", "upsetting", "sad", "sadness", "saddened", "scared", "afraid", "fear",
    "fearful", "frightened", "disgust", "disgusted", "disgusting", "emotion",
    "emotions", "emotional",
]

_WORD_RE = re.compile(r"\b(" + "|".join(BANNED_EMOTION_WORDS) + r")\b", re.IGNORECASE)

# "Announcing" a feeling to the listener ("I can see you're upset", "you seem really
# happy"). Not enforced in decoding, so it measures what the prompt achieves. Includes
# words that are *not* banned (happy, excited, worried...), so it is not 0 by design.
_FEELINGS = (r"feeling|feel|angry|upset|frustrated|sad|scared|afraid|worried|nervous|anxious|"
             r"happy|excited|disgusted|annoyed|hurt|down|surprised|shocked|stressed|"
             r"overwhelmed|apprehensive|thrilled|delighted|confused")
_ANNOUNCE_RE = re.compile(
    r"\b(?:you(?:'re| are| seem| sound| look| must be| must feel| feel)|"
    r"i (?:can )?(?:see|sense|hear|tell|understand)(?: that)? you(?:'re| are)?)"
    r"(?:\s+\w+){0,2}?\s+(?:" + _FEELINGS + r")\b",
    re.IGNORECASE)

_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "st", "jr", "sr", "vs", "etc"}


def sentence_ends(text: str) -> list[int]:
    """Character offsets where a sentence ends: a run of . ! ? that is not an
    ellipsis ("...") and does not follow a common abbreviation ("Mr.")."""
    ends = []
    for m in re.finditer(r"[.!?]+", text):
        run = m.group()
        if set(run) == {"."} and len(run) > 1:
            continue  # ellipsis: a pause, not a sentence end
        prev_word = re.search(r"(\w+)$", text[:m.start()])
        if run == "." and prev_word and prev_word.group(1).lower() in _ABBREVIATIONS:
            continue
        ends.append(m.end())
    return ends


def count_sentences(text: str) -> int:
    """Finished sentences, plus one if text trails on after the last terminator."""
    text = text.strip()
    ends = sentence_ends(text)
    trailing = text[ends[-1]:].strip(" \"')]") if ends else text
    return len(ends) + (1 if trailing else 0)


def ends_cleanly(text: str) -> bool:
    """True when the reply ends on a sentence terminator (not cut off mid-sentence)."""
    text = text.strip().rstrip("\"')]")
    return bool(text) and text[-1] in ".!?"


def has_emotion_word(text: str) -> bool:
    return bool(_WORD_RE.search(text))


def announces_feeling(text: str) -> bool:
    return bool(_ANNOUNCE_RE.search(text))
