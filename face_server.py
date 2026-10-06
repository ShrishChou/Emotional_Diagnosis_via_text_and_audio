"""Local web face: talk to the pipeline from a browser and watch it react.

  python face_server.py            # then open http://localhost:8000

Serves face/ (a chat page with an SVG face) and the endpoints below. Each turn is
one POST whose response streams newline-delimited JSON events while the pipeline
runs, so the face can react to the emotion state before the reply text arrives.
Add `?tts=1` to a turn's URL to also receive the reply as speech (Kokoro-82M), one
sentence at a time; the page leaves it off when muted.

  POST /api/utterance   body: recorded audio (webm/mp4/wav), transcribed by Whisper
  POST /api/text        body: {"text": "..."} a typed message (text-only head, no audio)
  POST /api/clip        body: {"split": "test", "clip": "dia113_utt10"} (gold transcript)
  POST /api/reset       forget the conversation
  POST /api/ping        wake the GPU (the page calls it every second while the person
                        talks; after a few idle seconds, the first turn is several
                        times slower on Apple Silicon, see scripts/12_idle_latency.py)

  events: {"type": "transcribing"}            (only when Whisper runs)
          {"type": "state", "state": {...}}   the JSON state from Pipeline.analyze
          {"type": "token", "text": "..."}    streamed reply chunks
          {"type": "audio", "text": "...", "duration_s": 2.6, "wav_b64": "..."}
                                              one spoken sentence (with ?tts=1)
          {"type": "done", "latency_ms": {...}}
          {"type": "error", "message": "..."}

Standard library HTTP server only; everything runs on this machine, and the server
listens on localhost only.
"""
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import argparse
import base64
import json
import re
import subprocess
import tempfile
import threading
import time
import warnings
from collections import deque
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

warnings.filterwarnings("ignore")

import pandas as pd

from config import CFG

FACE_DIR = CFG.root / "face"
DEMO_CLIPS = ["dia113_utt10", "dia2_utt1", "dia232_utt1", "dia170_utt0", "dia84_utt1"]  # results/examples.md

PIPE = None                  # loaded in main()
PIPE_LOCK = threading.Lock()  # one turn at a time: the models are not thread-safe
HISTORY = deque(maxlen=2)     # last two lines, as "Speaker: text", fed as context


def decode_to_wav(audio_bytes: bytes, out_path: Path) -> None:
    """Browser recordings arrive as webm/opus (Chrome) or mp4/aac (Safari)."""
    with tempfile.NamedTemporaryFile(suffix=".bin") as f:
        f.write(audio_bytes)
        f.flush()
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", f.name,
                        "-vn", "-ac", "1", "-ar", str(CFG.sample_rate), str(out_path)],
                       check=True, timeout=60)


def meld_clip(split: str, clip: str) -> dict:
    m = re.fullmatch(r"dia(\d+)_utt(\d+)", clip)
    if split not in ("train", "dev", "test") or not m:
        raise ValueError("bad clip id")
    df = pd.read_csv(CFG.manifest_dir / f"{split}.csv", keep_default_na=False)
    row = df[(df.dialogue_id == int(m[1])) & (df.utterance_id == int(m[2]))]
    if len(row) != 1:
        raise ValueError(f"{split}/{clip} not found")
    return row.iloc[0].to_dict()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FACE_DIR), **kwargs)

    def log_message(self, fmt, *args):  # keep the console for turn summaries
        pass

    # ------------------------------------------------------------- static files

    def do_GET(self):
        if self.path == "/api/demo_clips":
            return self.send_json({"clips": DEMO_CLIPS})
        m = re.fullmatch(r"/audio/(train|dev|test)/(dia\d+_utt\d+)\.wav", self.path)
        if m:  # MELD demo audio, so the browser can play the clip it sends
            path = CFG.wav_dir / m[1] / f"{m[2]}.wav"
            if not path.exists():
                return self.send_error(404)
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        return super().do_GET()

    # ------------------------------------------------------------- API

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        url = urlsplit(self.path)
        self.path = url.path
        self.tts = parse_qs(url.query).get("tts") == ["1"] and PIPE.tts is not None
        if self.path == "/api/reset":
            HISTORY.clear()
            return self.send_json({"ok": True})
        if self.path == "/api/ping":
            # Skip if a turn is running: the GPU is awake anyway, and a ping must never delay it.
            if PIPE_LOCK.acquire(blocking=False):
                try:
                    took = PIPE.ping()
                finally:
                    PIPE_LOCK.release()
                return self.send_json({"ok": True, "ms": round(took * 1000, 1)})
            return self.send_json({"ok": True, "skipped": True})
        if self.path not in ("/api/utterance", "/api/clip", "/api/text"):
            return self.send_error(404)

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with PIPE_LOCK, tempfile.TemporaryDirectory() as tmp:
            try:
                history = "\n".join(HISTORY)
                if self.path == "/api/clip":
                    # A MELD clip keeps its own dialogue context and is not added to the chat history.
                    req = json.loads(body)
                    row = meld_clip(req.get("split", "test"), req["clip"])
                    state = PIPE.analyze(CFG.root / row["wav_path"], text=row["text"], context=row["context"],
                                         utterance_id=req["clip"], transcript_source="gold")
                    self.run_turn(state, row["context"], remember=False)
                elif self.path == "/api/text":
                    text = json.loads(body).get("text", "").strip()[:1000]
                    if not text:
                        raise ValueError("empty message")
                    self.run_turn(PIPE.analyze_text(text, history), history, remember=True)
                else:
                    wav = Path(tmp) / "utterance.wav"
                    decode_to_wav(body, wav)
                    self.event({"type": "transcribing"})
                    self.run_turn(PIPE.analyze(wav, text=None, context=history, transcript_source="asr"),
                                  history, remember=True)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the page was closed mid-turn
            except Exception as e:  # report to the page instead of dying
                self.event({"type": "error", "message": f"{type(e).__name__}: {e}"})

    def run_turn(self, state: dict, context: str, remember: bool):
        """Send the state, stream the reply, and (for live chat) remember the exchange."""
        self.event({"type": "state", "state": state})
        reply = []
        events = (PIPE.respond_spoken(state, context) if self.tts
                  else ({"type": "token", "text": c} for c in PIPE.respond(state, context)))
        for ev in events:
            if ev["type"] == "audio":
                ev = {"type": "audio", "text": ev["text"], "duration_s": ev["duration_s"],
                      "wav_b64": base64.b64encode(ev["wav"]).decode()}
            else:
                reply.append(ev["text"])
            self.event(ev)
        self.event({"type": "done", "latency_ms": state["latency_ms"]})
        if remember:  # live conversation: the next turn sees this exchange as context
            HISTORY.append(f"Person: {state['transcript']}")
            HISTORY.append(f"Robot: {''.join(reply).strip()}")
        lat = state["latency_ms"]
        print(f"[{time.strftime('%H:%M:%S')}] {state['emotion']} ({state['confidence']:.2f}) "
              f"\"{state['transcript'][:60]}\"  state {lat['state_total']:.0f} ms, "
              f"first token {lat.get('llm_first_token', 0):.0f} ms"
              + (f", first audio {lat['tts_first_audio']:.0f} ms" if "tts_first_audio" in lat else ""), flush=True)

    def event(self, obj: dict) -> None:
        self.wfile.write((json.dumps(obj) + "\n").encode())
        self.wfile.flush()

    def send_json(self, obj: dict) -> None:
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    global PIPE
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-tts", action="store_true", help="do not load the speech model")
    args = ap.parse_args()

    from src.pipeline import Pipeline
    print(f"loading models on {CFG.device} ...", flush=True)
    PIPE = Pipeline(load_asr=True, load_llm=True, load_tts=not args.no_tts)
    print(f"cold load {sum(PIPE.cold_load_s.values()):.1f} s, warm-up {PIPE.warmup():.1f} s", flush=True)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"face ready: http://localhost:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
