"""Blind pairwise check: does the emotion state make replies better? -> evals/replies/blind_review.json

Interactive, for one human rater (about 10 minutes). Reads evals/replies/blind_pairs.json
(written by 07_compare_reply_models.py). For each clip it plays the audio (macOS
`afplay`; elsewhere it prints the path), shows the context and transcript, and the
two replies in random order as A and B. The rater picks the reply that better fits
what the person said *and how they said it*. Labels, predictions and which reply had
the emotion state stay hidden until the end.

Progress is saved after every answer, so it can be stopped (q) and resumed.
"""
import json
import random
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CFG, write_json

OUT = CFG.out("replies", "blind_review.json")


def play(wav: Path) -> None:
    if shutil.which("afplay"):
        subprocess.run(["afplay", str(wav)], check=False)
    else:
        print(f"  (play this file yourself: {wav})")


def summarize(state: dict) -> dict:
    answers = [a for a in state["answers"] if a["choice"] in ("emotion", "no_emotion", "tie")]
    n = len(answers)
    return {
        "n_rated": n,
        "preferred_with_emotion": sum(a["choice"] == "emotion" for a in answers),
        "preferred_without_emotion": sum(a["choice"] == "no_emotion" for a in answers),
        "ties": sum(a["choice"] == "tie" for a in answers),
    }


def main():
    pairs_file = json.loads(CFG.out("replies", "blind_pairs.json").read_text())
    pairs = {p["clip"]: p for p in pairs_file["pairs"]}

    if OUT.exists():
        state = json.loads(OUT.read_text())
        assert state["pairs_model"] == pairs_file["model"], "blind_pairs.json changed; delete blind_review.json"
    else:
        rng = random.SystemRandom()  # the rater cannot predict the A/B order
        order = list(pairs)
        rng.shuffle(order)
        state = {"pairs_model": pairs_file["model"], "pairs_style": pairs_file["style"],
                 "rater": "single rater", "date": str(date.today()),
                 "order": order, "emotion_is_a": {c: rng.random() < 0.5 for c in order},
                 "answers": []}

    done = {a["clip"] for a in state["answers"]}
    todo = [c for c in state["order"] if c not in done]
    print(__doc__)
    print(f"{len(done)} done, {len(todo)} to go. Keys: a / b / t (tie) / r (replay) / q (quit)\n")

    for k, clip in enumerate(todo, len(done) + 1):
        p = pairs[clip]
        a, b = ((p["with_emotion"], p["without_emotion"]) if state["emotion_is_a"][clip]
                else (p["without_emotion"], p["with_emotion"]))
        print(f"--- {k}/{len(state['order'])} ---")
        if p["context"]:
            print("Context:\n  " + p["context"].replace("\n", "\n  "))
        print(f'They said: "{p["transcript"]}"')
        play(CFG.root / p["wav_path"])
        print(f"\n  A: {a}\n  B: {b}\n")
        while True:
            key = input("Better fit? [a/b/t/r/q] ").strip().lower()
            if key == "r":
                play(CFG.root / p["wav_path"])
            elif key in ("a", "b", "t", "q"):
                break
        if key == "q":
            break
        if key == "t":
            choice = "tie"
        else:
            picked_a = key == "a"
            choice = "emotion" if picked_a == state["emotion_is_a"][clip] else "no_emotion"
        state["answers"].append({"clip": clip, "key": key, "choice": choice})
        state["summary"] = summarize(state)
        write_json(OUT, state)
        print()

    state["summary"] = summarize(state)
    write_json(OUT, state)
    s = state["summary"]
    print(f"\nRated {s['n_rated']}: preferred the emotion-conditioned reply in "
          f"{s['preferred_with_emotion']}/{s['n_rated']}, the plain one in "
          f"{s['preferred_without_emotion']}, ties {s['ties']}. Saved to {OUT.relative_to(CFG.root)}")


if __name__ == "__main__":
    main()
