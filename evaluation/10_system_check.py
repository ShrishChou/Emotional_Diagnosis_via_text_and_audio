"""End-to-end system check -> evals/system/system_check.{md,json}

Part A: requirements. The brief's hard constraints and the plan's acceptance
checklist, checked against the files in the repo.

Part B: is the model working as expected? The live inference path is checked
against the offline evaluation and probed for sensible behavior:
  consistency   live Pipeline.analyze / analyze_text on 300 test utterances vs the
                predictions computed offline from cached features (03_analyze_errors.py)
  calibration   reliability of the fused confidence on all of test
  audio         does the audio branch matter? same words, audio swapped for another
                clip's (of a different gold emotion) or for silence
  edge cases    silence, noise, very short / long / loud audio, empty or odd text:
                no crash, valid probabilities, replies still follow the rules
  determinism   the same input twice gives the same state
  latency       warm, back-to-back state latency for voice and typed input
  server        if app/server.py is running: event order of /api/text and /api/clip

  python evaluation/10_system_check.py [--server http://localhost:8000]
"""
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import argparse
import json
import re
import subprocess
import sys
import tempfile
import urllib.request
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import soundfile as sf

from config import CFG, LABELS, write_json

N_CONSISTENCY = 300
N_SWAP = 150
ROOT = CFG.root
EVALS = CFG.evals_dir

checks = []  # (section, name, status, detail); status is PASS / FAIL / WARN / INFO


def check(section: str, name: str, ok, detail: str) -> None:
    status = ok if isinstance(ok, str) else ("PASS" if ok else "FAIL")
    checks.append({"section": section, "check": name, "status": status, "detail": detail})
    print(f"[{status}] {name}: {detail}", flush=True)


# ====================================================================== Part A

def requirements() -> None:
    sec = "A. Requirements"
    params = json.loads((EVALS / "system" / "params.json").read_text())
    check(sec, "Total parameters <= 6B", params["total"] <= 6e9,
          f"{params['total']:,d} ({params['fraction_of_limit']:.0%} of the limit), "
          f"{len(params['components'])} components counted in code")
    check(sec, "Speech model (TTS) is counted", params["components"].get("tts", {}).get("source") == CFG.tts_model,
          f"tts: {params['components'].get('tts', {}).get('params', 0):,d} parameters ({CFG.tts_model})")
    check(sec, "Counted LLM is the configured LLM",
          params["components"]["reply_llm"]["source"] == CFG.llm_model,
          f"params.json: {params['components']['reply_llm']['source']}; config: {CFG.llm_model}")

    m = json.loads((EVALS / "classifiers" / "metrics.json").read_text())["models"]
    have = {k: round(m[k]["test"]["weighted_f1"]["mean"], 3) for k in ("text", "audio", "fused") if k in m}
    check(sec, "Test weighted F1 for text, audio and fused", len(have) == 3, str(have))

    lat = json.loads((EVALS / "system" / "latency.json").read_text())
    hw = json.loads((EVALS / "system" / "hardware.json").read_text())
    llm = lat.get("llm", {})
    check(sec, "Latency and hardware reports exist and agree", lat["device"] == hw["device"],
          f"device {lat['device']}; state p50 {lat['latency_ms']['state_no_asr']['p50']} ms, "
          f"first token p50 {lat['latency_ms']['llm_first_token']['p50']} ms")
    check(sec, "Latency was measured with the current reply setup",
          llm.get("model") == CFG.llm_model and llm.get("reply_style") == CFG.reply_style,
          f"latency.json: {llm.get('model')} / {llm.get('reply_style')}; config: {CFG.llm_model} / {CFG.reply_style}")

    trace = (EVALS / "system" / "demo_trace.txt").read_text()
    check(sec, "Demo trace shows a JSON state and a streamed reply",
          '"emotion"' in trace and "== reply (streamed) ==" in trace and "llm_first_token" in trace,
          "evals/system/demo_trace.txt")

    # Every script that loads a pretrained model, except the setup step that downloads them.
    files = [ROOT / "demo.py", ROOT / "app" / "server.py", *(ROOT / "training").glob("*.py"),
             *(ROOT / "evaluation").glob("*.py")]
    uses_models = re.compile(r"src\.(pipeline|encoders|responder|tts)|from transformers|load_deployed")
    entry = sorted(str(f.relative_to(ROOT)) for f in files
                   if uses_models.search(f.read_text()) and f.name != "01_check_environment.py")
    missing = [f for f in entry if 'os.environ["HF_HUB_OFFLINE"] = "1"' not in (ROOT / f).read_text()]
    check(sec, "Offline mode forced in every inference entry point", not missing,
          f"{len(entry) - len(missing)}/{len(entry)} files" + (f"; missing: {missing}" if missing else ""))

    code = [p for p in list((ROOT / "src").glob("*.py")) + [ROOT / "demo.py", ROOT / "app" / "server.py"]]
    remote = [str(p.relative_to(ROOT)) for p in code
              if re.search(r"\b(import requests|openai|anthropic|httpx|urlopen\()", p.read_text())]
    bind = re.search(r'ThreadingHTTPServer\(\("([\d.]+)"', (ROOT / "app" / "server.py").read_text())
    check(sec, "No remote API on the inference path", not remote and bind and bind[1] == "127.0.0.1",
          "no HTTP client libraries in src/, demo.py, app/server.py; "
          f"the app server binds to {bind[1] if bind else '?'}")

    sizes = {s: len(pd.read_csv(CFG.manifest_dir / f"{s}.csv")) for s in ("train", "dev", "test")}
    check(sec, "Official MELD splits", sizes == {"train": 9988, "dev": 1108, "test": 2610},
          f"{sizes} (official CSVs: 9,989 / 1,109 / 2,610; 2 clips missing in MELD)")

    notes = (ROOT / "NOTES.md").read_text()
    readme = (ROOT / "README.md").read_text()
    todos = sum(t.count("TODO(Shrish)") for t in (readme, notes))
    check(sec, "NOTES.md logs deviations; TODO(Shrish) markers present", len(notes) > 2000 and todos > 0,
          f"NOTES.md {len(notes.splitlines())} lines; {todos} TODO(Shrish) markers")

    gi = (ROOT / ".gitignore").read_text()
    big = [str(p.relative_to(ROOT)) for p in (ROOT / "models").glob("*") if p.stat().st_size > 50e6]
    check(sec, "Data and caches ignored; no model file over 50 MB",
          "data/" in gi and "cache/" in gi and not big, f"big files: {big or 'none'}")

    # Every README number should trace to a generated file in evals/ (not evals/README.md,
    # which is written by hand) or to models/meta.json.
    pool = []
    generated = [p for ext in ("json", "md", "txt") for p in EVALS.rglob(f"*.{ext}") if p != EVALS / "README.md"]
    for p in generated + [CFG.models_dir / "meta.json"]:
        pool += [float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*\.?\d*", p.read_text())
                 if x.replace(",", "").replace(".", "").replace("-", "").isdigit()]
    pool = np.array(sorted(set(pool)))
    body = re.sub(r"```.*?```", "", readme, flags=re.S)       # skip code blocks (the demo trace)
    nums = re.findall(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+\.\d+|\d{3,})(?![\w])", body)
    unmatched = []
    for n in nums:
        x = float(n.replace(",", ""))
        dec = len(n.split(".")[1]) if "." in n else 0
        tol = 0.5 * 10 ** -dec + 1e-9
        if not (np.abs(pool - x) <= tol).any() and not (np.abs(pool * 100 - x) <= tol).any() \
                and not (np.abs(pool / 1e9 - x) <= tol).any():
            unmatched.append(n)
    frac = 1 - len(unmatched) / max(len(nums), 1)
    check(sec, "README numbers trace to evals/", "PASS" if frac > 0.95 else "WARN",
          f"{len(nums) - len(unmatched)}/{len(nums)} numbers found in evals/ or models/meta.json; "
          f"not found (review by hand: derived or stale): {sorted(set(unmatched))[:25]}")

    try:
        n_commits = int(subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=ROOT,
                                       capture_output=True, text=True).stdout.strip() or 0)
    except ValueError:
        n_commits = 0
    check(sec, "Work is committed (a fresh clone reproduces it)", n_commits > 0,
          f"{n_commits} commits" + ("" if n_commits else "; nothing is committed yet, so a clone would be empty"))


# ====================================================================== Part B

def model_checks(server: str | None) -> dict:
    from src.audio import read_raw
    from src.pipeline import Pipeline

    sec = "B. Model behavior"
    pipe = Pipeline(load_asr=True, load_llm=True, load_tts=True)
    pipe.warmup()

    man = pd.read_csv(CFG.manifest_dir / "test.csv", keep_default_na=False)
    pred = pd.read_csv(EVALS / "struggles" / "test_predictions.csv", keep_default_na=False)
    man["clip"] = [f"dia{d}_utt{u}" for d, u in zip(man.dialogue_id, man.utterance_id)]
    df = man.merge(pred[["clip", "fused_pred", "fused_conf", "text_pred"]], on="clip")
    sample = df.groupby("emotion").sample(frac=N_CONSISTENCY / len(df), random_state=0)

    # --- consistency: live path vs offline cached-feature predictions
    live, typed = [], []
    for r in sample.itertuples():
        s = pipe.analyze(ROOT / r.wav_path, text=r.text, context=r.context, transcript_source="gold")
        t = pipe.analyze_text(r.text, r.context)
        live.append(s)
        typed.append(t)
    sample = sample.assign(live_pred=[s["emotion"] for s in live], live_conf=[s["confidence"] for s in live],
                           typed_pred=[t["emotion"] for t in typed])
    agree = (sample.live_pred == sample.fused_pred).mean()
    dconf = (sample.live_conf - sample.fused_conf).abs()
    check(sec, "Live voice path matches the offline evaluation", agree >= 0.98,
          f"{agree:.1%} of {len(sample)} test utterances get the same emotion; confidence differs by "
          f"median {dconf.median():.4f}, max {dconf.max():.3f} (offline features were batched and padded)")
    agree_t = (sample.typed_pred == sample.text_pred).mean()
    check(sec, "Typed path matches the offline text-only head", agree_t >= 0.98,
          f"{agree_t:.1%} agreement on the same utterances")
    acc_live = (sample.live_pred == sample.emotion).mean()
    acc_off = (sample.fused_pred == sample.emotion).mean()
    check(sec, "Live accuracy equals offline accuracy on the sample", abs(acc_live - acc_off) < 0.01,
          f"live {acc_live:.3f} vs offline {acc_off:.3f}; "
          f"typed (words only) {(sample.typed_pred == sample.emotion).mean():.3f}")

    # --- calibration on all of test (offline predictions)
    conf = pred.fused_conf.to_numpy()
    correct = pred.fused_correct.astype(str).str.lower().eq("true").to_numpy()
    bins = np.linspace(0, 1, 11)
    rel = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mk = (conf > lo) & (conf <= hi)
        if mk.sum() >= 20:
            rel.append((f"{lo:.1f}-{hi:.1f}", int(mk.sum()), float(conf[mk].mean()), float(correct[mk].mean())))
    gaps = [abs(c - a) for _, _, c, a in rel]
    check(sec, "Confidence is calibrated (fused, test)", max(gaps) < 0.1,
          "bins with >= 20 utterances, mean confidence -> accuracy: " +
          "; ".join(f"{b}: {c:.2f}->{a:.2f} (n={n})" for b, n, c, a in rel))

    # --- audio influence: swap audio, keep words
    rng = np.random.default_rng(0)
    swap = sample.sample(n=min(N_SWAP, len(sample)), random_state=1)
    changed, toward, silence_changed = 0, 0, 0
    with tempfile.TemporaryDirectory() as tmp:
        silent = Path(tmp) / "silence.wav"
        sf.write(silent, np.zeros(CFG.sample_rate * 2, dtype=np.float32), CFG.sample_rate)
        for r in swap.itertuples():
            base = pipe.analyze(ROOT / r.wav_path, text=r.text, context=r.context)
            others = df[df.emotion != r.emotion]
            donor = others.iloc[rng.integers(len(others))]
            sw = pipe.analyze(ROOT / donor.wav_path, text=r.text, context=r.context)
            changed += sw["emotion"] != base["emotion"]
            toward += sw["probs"][donor.emotion] > base["probs"][donor.emotion]
            si = pipe.analyze(silent, text=r.text, context=r.context)
            silence_changed += si["emotion"] != base["emotion"]
    n = len(swap)
    check(sec, "Audio branch is live: swapping audio changes predictions", changed / n > 0.05,
          f"same words, another clip's audio: prediction changed in {changed}/{n} ({changed / n:.0%}); "
          f"silence instead of the voice changed it in {silence_changed}/{n}")
    check(sec, "Swapped audio pulls toward the donor clip's emotion", toward / n > 0.6,
          f"the donor's gold emotion gained probability in {toward}/{n} ({toward / n:.0%}) swaps")

    # --- edge cases: no crash, valid probabilities, replies follow the rules
    from src.reply_checks import count_sentences, has_emotion_word
    clip = ROOT / sample.iloc[0].wav_path
    raw, _ = read_raw(clip)
    long_raw = np.tile(raw, int(np.ceil(25 * CFG.sample_rate / len(raw))))[:25 * CFG.sample_rate]
    cases = {
        "silence 1 s": (np.zeros(CFG.sample_rate), "Hello."),
        "white noise 2 s": (rng.normal(0, 0.3, 2 * CFG.sample_rate), "Hello."),
        "0.05 s of audio": (raw[:800], "Hi."),
        "25 s of audio (cropped to 15 s)": (long_raw, sample.iloc[0].text),
        "clipped, very loud audio": (np.clip(raw * 50, -1, 1), sample.iloc[0].text),
        "empty transcript": (raw, ""),
        "one character": (raw, "?"),
        "300-word transcript": (raw, " ".join(["this is a very long line"] * 60)),
        "non-English text": (raw, "¿Dónde está la biblioteca? No lo sé."),
        "emoji only": (raw, "😂😂😂"),
    }
    edge = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, (wav, text) in cases.items():
            path = Path(tmp) / "case.wav"
            sf.write(path, np.asarray(wav, dtype=np.float32), CFG.sample_rate)
            try:
                s = pipe.analyze(path, text=text, context="")
                p = np.array(list(s["probs"].values()))
                ok = np.isfinite(p).all() and abs(p.sum() - 1) < 0.01
                reply = "".join(pipe.respond(s, "")).strip()
                rules = count_sentences(reply) <= 2 and not has_emotion_word(reply)
                edge.append({"case": name, "emotion": s["emotion"], "confidence": s["confidence"],
                             "valid_probs": bool(ok), "reply": reply, "reply_ok": bool(rules and reply)})
            except Exception as e:
                edge.append({"case": name, "error": f"{type(e).__name__}: {e}"})
    crashed = [e["case"] for e in edge if "error" in e]
    valid = all(e["valid_probs"] for e in edge if "error" not in e)
    check(sec, "Edge cases: no crash, valid probabilities", not crashed and valid,
          f"{len(edge) - len(crashed)}/{len(edge)} ran" + (f"; crashed: {crashed}" if crashed else ""))
    bad_reply = [e["case"] for e in edge if "error" not in e and not e["reply_ok"]]
    check(sec, "Edge cases: replies non-empty, <= 2 sentences, no banned words", not bad_reply,
          f"{len(edge) - len(crashed) - len(bad_reply)}/{len(edge) - len(crashed)} ok" +
          (f"; failing: {bad_reply}" if bad_reply else ""))
    # Criterion revised after the first run (see NOTES.md): confidently *neutral* on
    # silence or noise is safe for the face; a strong non-neutral emotion is not.
    noisy = [e for e in edge if e["case"] in ("silence 1 s", "white noise 2 s") and "error" not in e]
    calm = all(e["emotion"] == "neutral" or e["confidence"] < 0.5 for e in noisy)
    check(sec, "Silence / noise does not trigger a strong emotion", calm,
          "; ".join(f"{e['case']}: {e['emotion']} ({e['confidence']:.2f})" for e in noisy))
    empty = next(e for e in edge if e["case"] == "empty transcript")
    check(sec, "Empty transcript reports 'nothing heard' instead of guessing",
          "error" not in empty and empty["emotion"] == "neutral" and empty["confidence"] < 0.2,
          f"{empty.get('emotion')} ({empty.get('confidence', 0):.2f}); reply: {empty.get('reply', '')}")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "muted.wav"
        sf.write(path, np.zeros(2 * CFG.sample_rate, dtype=np.float32), CFG.sample_rate)
        s = pipe.analyze(path, text=None)   # the live-mic path: Whisper would run here
    check(sec, "Muted mic (live path) skips Whisper and reports nothing heard",
          s["heard"] is False and s["latency_ms"]["asr"] < 5,
          f"heard={s['heard']}, asr {s['latency_ms']['asr']} ms, transcript '{s['transcript']}'")

    # --- the "nothing heard" gate never fires on real speech
    from src.pipeline import SILENCE_RMS
    rms = []
    for split in ("dev", "test"):
        for wp in pd.read_csv(CFG.manifest_dir / f"{split}.csv")["wav_path"]:
            w, _ = read_raw(ROOT / wp)
            rms.append(float(np.sqrt(np.mean(w ** 2))) if len(w) else 0.0)
    rms = np.array(rms)
    check(sec, "Silence gate never fires on real MELD speech", (rms < SILENCE_RMS).sum() == 0,
          f"{len(rms):,d} dev + test clips; quietest raw RMS {rms.min():.4f}, 1st percentile "
          f"{np.percentile(rms, 1):.4f}, median {np.median(rms):.4f}; gate {SILENCE_RMS}")

    # --- speech output works offline and covers the whole reply
    s = pipe.analyze_text("I just got the keys to my first apartment!")
    evs = list(pipe.respond_spoken(s))
    spoken = " ".join(e["text"] for e in evs if e["type"] == "audio")
    written = "".join(e["text"] for e in evs if e["type"] == "token").strip()
    durs = [e["duration_s"] for e in evs if e["type"] == "audio"]
    check(sec, "Speech output: every sentence of the reply is spoken (offline)",
          bool(durs) and min(durs) > 0.3 and spoken.split() == written.split(),
          f"{len(durs)} sentences, {sum(durs):.1f} s of audio, first ready {s['latency_ms']['tts_first_audio']:.0f} ms "
          f"after the reply started; spoken text equals written text: {spoken.split() == written.split()}")

    # --- determinism
    r0 = sample.iloc[0]
    a = pipe.analyze(ROOT / r0.wav_path, text=r0.text, context=r0.context)
    b = pipe.analyze(ROOT / r0.wav_path, text=r0.text, context=r0.context)
    diff = max(abs(a["probs"][k] - b["probs"][k]) for k in LABELS)
    check(sec, "Deterministic state for the same input", diff < 1e-3, f"max probability difference {diff:.5f}")

    # --- latency of the live paths (warm, back to back)
    sv = np.array([s["latency_ms"]["state_total"] for s in live])
    st = np.array([t["latency_ms"]["state_total"] for t in typed])
    check(sec, "Warm state latency (voice, typed)", "INFO",
          f"voice p50 {np.percentile(sv, 50):.0f} / p95 {np.percentile(sv, 95):.0f} ms; "
          f"typed p50 {np.percentile(st, 50):.0f} / p95 {np.percentile(st, 95):.0f} ms "
          f"(back to back; see 06_idle_latency for gaps)")

    # --- live server protocol
    if server:
        def events(path, body):
            req = urllib.request.Request(server + path, data=json.dumps(body).encode(), method="POST")
            with urllib.request.urlopen(req, timeout=60) as resp:
                return [json.loads(line) for line in resp.read().decode().splitlines() if line.strip()]
        try:
            for path, body in [("/api/text", {"text": "I can't believe you remembered my birthday!"}),
                               ("/api/clip", {"split": "test", "clip": "dia113_utt10"})]:
                ev = [e["type"] for e in events(path, body)]
                order_ok = ev[0] == "state" and ev[-1] == "done" and set(ev[1:-1]) == {"token"}
                check(sec, f"Server {path}: state, then tokens, then done", order_ok,
                      f"{len(ev)} events: {ev[0]}, {len(ev) - 2} tokens, {ev[-1]}")
            ev = events("/api/text", {"text": ""})
            check(sec, "Server rejects an empty message cleanly", ev and ev[0]["type"] == "error",
                  ev[0].get("message", ""))
            events("/api/reset", {})
        except Exception as e:
            check(sec, "Server reachable", "WARN", f"{server}: {e}")
    return {"edge_cases": edge,
            "reliability": [{"bin": b, "n": n, "mean_confidence": c, "accuracy": a} for b, n, c, a in rel]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", help="URL of a running app/server.py, e.g. http://localhost:8000")
    args = ap.parse_args()
    requirements()
    extra = model_checks(args.server)

    counts = pd.Series([c["status"] for c in checks]).value_counts().to_dict()
    lines = ["# System check", "",
             "Generated by `evaluation/10_system_check.py`. " +
             ", ".join(f"{v} {k}" for k, v in counts.items()) + ".", ""]
    for sec in dict.fromkeys(c["section"] for c in checks):
        lines += [f"## {sec}", "", "| Check | Result | Detail |", "| --- | --- | --- |"]
        for c in checks:
            if c["section"] == sec:
                lines.append(f"| {c['check']} | {c['status']} | {c['detail'].replace('|', '/')} |")
        lines.append("")
    lines += ["## Edge cases", "", "| Case | Emotion (confidence) | Reply |", "| --- | --- | --- |"]
    for e in extra["edge_cases"]:
        if "error" in e:
            lines.append(f"| {e['case']} | crashed: {e['error']} | |")
        else:
            lines.append(f"| {e['case']} | {e['emotion']} ({e['confidence']:.2f}) | {e['reply'].replace('|', '/')} |")
    CFG.out("system", "system_check.md").write_text("\n".join(lines) + "\n")
    write_json(CFG.out("system", "system_check.json"), {"summary": counts, "checks": checks, **extra})
    print(f"\n{counts}")


if __name__ == "__main__":
    main()
