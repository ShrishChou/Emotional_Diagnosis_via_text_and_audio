"""Step 2: prepare MELD: every clip -> 16 kHz mono wav, plus one manifest CSV per split.

Works with either source under data/raw/ (see NOTES.md):
  * the official MELD.Raw archive: {split}_sent_emo.csv + per-split mp4 folders
  * the Hugging Face audio mirror ajyy/MELD_audio: {split}.csv + per-split flac
    folders (same official CSVs, audio already extracted at 16 kHz mono)
Writes data/wav/{split}/dia{D}_utt{U}.wav and data/manifests/{split}.csv.
"""
import subprocess
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ftfy
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from config import CFG, LABEL2ID, LABELS
from src.data import dialogue_context

SPLITS = ["train", "dev", "test"]


CLIP_EXTS = (".mp4", ".flac")


def find_csv(split: str) -> Path:
    for name in (f"{split}_sent_emo.csv", f"{split}.csv"):
        hits = [p for p in CFG.raw_dir.rglob(name) if not p.name.startswith("._")]
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no label csv for split {split} under {CFG.raw_dir}")


def find_clip_dir(split: str) -> Path:
    """The folder of clips for a split. Folder names differ between MELD
    releases (e.g. train_splits, dev_splits_complete, output_repeated_splits_test),
    so we look for a directory whose name contains the split and holds clips."""
    candidates = []
    for d in CFG.raw_dir.rglob("*"):
        if d.is_dir() and split in d.name.lower() and not d.name.startswith("._"):
            n = sum(1 for p in d.glob("dia*_utt*.*") if p.suffix in CLIP_EXTS)
            if n:
                candidates.append((n, d))
    assert candidates, f"no clip folder found for split {split}"
    return max(candidates)[1]  # the folder with the most clips


def clean_text(s: str) -> str:
    return ftfy.fix_text(str(s)).strip()


def convert(job) -> tuple[str, bool, float]:
    """ffmpeg one clip (mp4 or flac) to 16 kHz mono wav. Returns (out_path, ok, duration_s)."""
    src, dst = job
    dst = Path(dst)
    if not dst.exists():
        cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src),
               "-vn", "-ac", "1", "-ar", str(CFG.sample_rate), str(dst)]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        except Exception:
            dst.unlink(missing_ok=True)
            return str(dst), False, 0.0
    try:
        info = sf.info(str(dst))
        if info.frames == 0:
            return str(dst), False, 0.0
        return str(dst), True, info.frames / info.samplerate
    except Exception:
        return str(dst), False, 0.0


def prepare_split(split: str, missing: list[str]) -> pd.DataFrame:
    csv = find_csv(split)
    clip_dir = find_clip_dir(split)
    print(f"[{split}] csv={csv.relative_to(CFG.root)} clips={clip_dir.relative_to(CFG.root)}")

    df = pd.read_csv(csv)
    df["text"] = df["Utterance"].map(clean_text)
    df["Speaker"] = df["Speaker"].map(clean_text)
    df["emotion"] = df["Emotion"].str.strip().str.lower()
    bad = set(df["emotion"]) - set(LABELS)
    assert not bad, f"unexpected labels in {split}: {bad}"
    df["label_id"] = df["emotion"].map(LABEL2ID)
    df["context"] = dialogue_context(df, "text", dialogue_col="Dialogue_ID", turn_col="Utterance_ID",
                                     speaker_col="Speaker")

    out_dir = CFG.wav_dir / split
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs, idx = [], []
    for i, r in df.iterrows():
        name = f"dia{r.Dialogue_ID}_utt{r.Utterance_ID}"
        src = next((clip_dir / f"{name}{ext}" for ext in CLIP_EXTS
                    if (clip_dir / f"{name}{ext}").exists()), None)
        if src is None:
            missing.append(f"{split}/{name} (clip not found)")
            continue
        jobs.append((str(src), str(out_dir / f"{name}.wav")))
        idx.append(i)

    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(convert, jobs, chunksize=8), total=len(jobs), desc=split))

    rows = []
    for i, (wav_path, ok, dur) in zip(idx, results):
        r = df.loc[i]
        name = f"dia{r.Dialogue_ID}_utt{r.Utterance_ID}"
        if not ok:
            missing.append(f"{split}/{name} (ffmpeg failed or empty audio)")
            continue
        rows.append({
            "split": split,
            "dialogue_id": int(r.Dialogue_ID),
            "utterance_id": int(r.Utterance_ID),
            "wav_path": str(Path(wav_path).relative_to(CFG.root)),
            "text": r.text,
            "context": r.context,
            "speaker": r.Speaker,
            "emotion": r.emotion,
            "label_id": int(r.label_id),
            "sentiment": r.Sentiment,
            "duration_s": round(dur, 3),
        })
    out = pd.DataFrame(rows)
    print(f"[{split}] {len(df)} rows in csv, {len(out)} with audio")
    print(out["emotion"].value_counts().reindex(LABELS).to_string())
    return out


def main():
    CFG.manifest_dir.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    for split in SPLITS:
        out = prepare_split(split, missing)
        out.to_csv(CFG.manifest_dir / f"{split}.csv", index=False)
        long = (out["duration_s"] > CFG.max_audio_s).sum()
        print(f"[{split}] clips longer than {CFG.max_audio_s:.0f}s (cropped at load time): {long}\n")
    CFG.out("system", "missing_clips.txt").write_text("\n".join(missing) + ("\n" if missing else ""))
    print(f"missing / failed clips: {len(missing)} (see evals/system/missing_clips.txt)")


if __name__ == "__main__":
    main()
