"""Small data helpers shared by data preparation and the evaluations."""
import re

import pandas as pd

from config import CFG


def load_manifest(split: str) -> pd.DataFrame:
    """data/manifests/{split}.csv (empty context stays an empty string, not NaN)."""
    return pd.read_csv(CFG.manifest_dir / f"{split}.csv", keep_default_na=False)


def dialogue_context(df: pd.DataFrame, text_col: str, n_prev: int = 2, dialogue_col: str = "dialogue_id",
                     turn_col: str = "utterance_id", speaker_col: str = "speaker") -> list[str]:
    """For every row, the previous `n_prev` lines of the same dialogue as
    "Speaker: text" lines joined by newlines ("" for the first line). Aligned with df.index."""
    ctx = {}
    for _, dia in df.groupby(dialogue_col, sort=False):
        dia = dia.sort_values(turn_col)
        lines = [f"{s}: {t}" for s, t in zip(dia[speaker_col], dia[text_col])]
        for k, i in enumerate(dia.index):
            ctx[i] = "\n".join(lines[max(0, k - n_prev):k])
    return [ctx[i] for i in df.index]


def clip_id(dialogue_id, utterance_id) -> str:
    return f"dia{dialogue_id}_utt{utterance_id}"


def words(text: str) -> list[str]:
    """Lowercased words without punctuation, for word error rate."""
    return re.sub(r"[^a-z0-9' ]", " ", text.lower()).split()


def edit_distance(a: list[str], b: list[str]) -> int:
    """Word-level Levenshtein distance."""
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[-1]
