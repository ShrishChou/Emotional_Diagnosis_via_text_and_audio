// Chat page controller: type or talk, see each message's emotion state, watch the face react,
// and read the reply as it is "spoken".
//
// Timeline of one turn:
//   input       typing or talking: the face listens (eyes widen, pulse with the voice or keys)
//   thinking    glance up-right while the server transcribes and classifies
//   reacting    detected emotion, blended toward neutral by (1 - confidence), held >= 0.9 s
//   replying    the robot's own (empathic) face. With sound on, each sentence is spoken
//               (Kokoro-82M on the server) as soon as it is written; the mouth follows the
//               loudness of the audio being played and the words appear across its duration.
//               Muted, the words appear at roughly speaking pace and each pulses the mouth.
//   idle        back to neutral a moment after the reply ends
//
// While the person types or talks, the page pings the server every second: on Apple Silicon
// the first turn after a few idle seconds is several times slower (scripts/12_idle_latency.py).

const face = new Face(document.getElementById("face"));
const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const REACT_HOLD_MS = 900;

let busy = false;

// ------------------------------------------------------------------ mute

const SPEAKER_ON = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"
  stroke-linejoin="round"><path d="M4 9v6h4l5 4V5L8 9H4z"/><path d="M16.5 8.5a5 5 0 0 1 0 7M19 6a8.5 8.5 0 0 1 0 12"/></svg>`;
const SPEAKER_OFF = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"
  stroke-linejoin="round"><path d="M4 9v6h4l5 4V5L8 9H4z"/><path d="M17 9l5 6M22 9l-5 6"/></svg>`;
let muted = false;
try { muted = localStorage.getItem("muted") === "1"; } catch {}
let currentSource = null;   // the sentence being spoken, so mute can cut it off

function showMute() {
  $("mute").innerHTML = muted ? SPEAKER_OFF : SPEAKER_ON;
  $("mute").classList.toggle("muted", muted);
  $("mute").title = muted ? "Unmute the voice" : "Mute the voice";
}
$("mute").addEventListener("click", () => {
  muted = !muted;
  try { localStorage.setItem("muted", muted ? "1" : "0"); } catch {}
  if (muted && currentSource) currentSource.stop();
  if (!muted) ctx().resume().catch(() => {});
  showMute();
});
showMute();

/** API URL for a turn: ask for speech unless muted. */
function turnUrl(path) { return muted ? path : `${path}?tts=1`; }

function status(html) { $("status").innerHTML = html; }

// ------------------------------------------------------------------ chat rendering

function addMessage(role) {
  $("empty")?.remove();
  const msg = document.createElement("div");
  msg.className = `msg ${role}`;
  msg.innerHTML = `<div class="bubble"></div><div class="meta"></div>`;
  $("messages").appendChild(msg);
  scrollDown();
  return { el: msg, bubble: msg.querySelector(".bubble"), meta: msg.querySelector(".meta") };
}

function scrollDown() { $("messages").scrollTop = $("messages").scrollHeight; }

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function chip(text, color) {
  const dot = color ? `<span class="dot" style="background:var(--${color})"></span>` : "";
  return `<span class="chip">${dot}${escapeHtml(text)}</span>`;
}

/** Emotion chips under a user message. */
function stateMeta(s) {
  if (s.heard === false) return chip("nothing heard: no speech or empty transcript");
  const how = s.transcript_source === "typed" ? "typed · words only"
            : s.transcript_source === "asr" ? "voice · tone + words"
            : "MELD clip · tone + words";
  let html = chip(`${s.emotion} ${s.confidence.toFixed(2)}`, s.emotion) + chip(how);
  if (s.audio_changed_prediction) html += chip(`tone changed it from ${s.text_only_emotion}`);
  return html;
}

// ------------------------------------------------------------------ loudness meter

let audioCtx = null;
function ctx() {
  audioCtx = audioCtx || new AudioContext();
  return audioCtx;
}

/** Feed face.setLevel from an audio node until stop() is called. */
function meter(sourceNode) {
  const analyser = ctx().createAnalyser();
  analyser.fftSize = 1024;
  sourceNode.connect(analyser);
  const buf = new Float32Array(analyser.fftSize);
  let running = true;
  (function tick() {
    if (!running) return;
    analyser.getFloatTimeDomainData(buf);
    let sum = 0;
    for (const v of buf) sum += v * v;
    face.setLevel(Math.min(1, Math.sqrt(sum / buf.length) * 6));
    requestAnimationFrame(tick);
  })();
  return { analyser, stop() { running = false; sourceNode.disconnect(analyser); } };
}

// ------------------------------------------------------------------ keep the GPU warm

let lastPing = 0;
function ping() {
  const now = performance.now();
  if (now - lastPing < 1000) return;
  lastPing = now;
  fetch("/api/ping", { method: "POST" }).catch(() => {});
}
let pinger = null;
function startPinging() { ping(); pinger = setInterval(ping, 1000); }
function stopPinging() { clearInterval(pinger); pinger = null; }

// ------------------------------------------------------------------ one turn

/**
 * Run one turn. `request` is the fetch of an API endpoint whose response streams NDJSON
 * events; `userMsg` is the user's chat message (its text is replaced by the transcript
 * when the state arrives, and its chips show the emotion state).
 */
async function runTurn(request, userMsg) {
  busy = true;
  setButtons();
  face.setListening(false);
  face.setThinking(true);
  face.neutral();
  status("<b>thinking</b>");

  // Items to "say": {words: [...]} (text-paced) or {text, buffer: Promise<AudioBuffer>}.
  // With sound on, spoken sentences drive the display; the streamed text is kept only as
  // a fallback in case no audio arrives.
  const voiced = !muted;
  const queue = [];
  const textWords = [];
  let pending = "";
  let done = false;
  let reactAt = null;
  let state = null;
  let gotAudio = false;
  const bot = { msg: null };
  const speaker = speak(queue, () => done, () => reactAt, () => state, bot);

  try {
    const res = await request;
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done: end } = await reader.read();
      if (end) break;
      buf += dec.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const ev = JSON.parse(buf.slice(0, nl));
        buf = buf.slice(nl + 1);
        if (ev.type === "transcribing") {
          status("<b>transcribing</b>");
        } else if (ev.type === "state") {
          state = ev.state;
          face.setThinking(false);
          face.react(state.emotion, state.confidence);
          reactAt = performance.now();
          userMsg.bubble.classList.remove("pending");
          userMsg.bubble.textContent = state.transcript || "(nothing heard)";
          userMsg.meta.innerHTML = stateMeta(state);
          status(`heard <b>${state.emotion}</b> (${state.confidence.toFixed(2)}) · state in ` +
                 `${Math.round(state.latency_ms.state_total)} ms`);
          $("state").textContent = JSON.stringify(state, null, 2);
          scrollDown();
        } else if (ev.type === "token") {
          pending += ev.text;
          const parts = pending.split(/(?<=\s)/);   // keep whitespace on each word
          pending = parts.pop();
          for (const w of parts) if (w.trim()) textWords.push(w);
          if (!voiced) queue.push(...textWords.splice(0).map((w) => ({ words: [w] })));
        } else if (ev.type === "audio") {
          gotAudio = true;
          const bytes = Uint8Array.from(atob(ev.wav_b64), (c) => c.charCodeAt(0));
          queue.push({ text: ev.text, buffer: ctx().decodeAudioData(bytes.buffer) });
        } else if (ev.type === "done") {
          if (pending.trim()) textWords.push(pending);
          pending = "";
          if (!voiced || !gotAudio) queue.push(...textWords.splice(0).map((w) => ({ words: [w] })));
          if (state) {
            state.latency_ms = ev.latency_ms;
            $("state").textContent = JSON.stringify(state, null, 2);
          }
        } else if (ev.type === "error") {
          userMsg.bubble.classList.remove("pending");
          if (!state && userMsg.bubble.textContent.endsWith("…")) userMsg.bubble.textContent = "(not understood)";
          userMsg.meta.innerHTML = chip(`error: ${ev.message}`);
          status(`<b>error</b> ${escapeHtml(ev.message)}`);
        }
      }
    }
  } catch (e) {
    status(`<b>error</b> ${escapeHtml(e)}`);
  }
  done = true;
  await speaker;
  if (bot.msg && state?.latency_ms) {
    const l = state.latency_ms;
    bot.msg.meta.textContent = `state ${Math.round(l.state_total)} ms` +
      (l.asr ? ` (incl. Whisper ${Math.round(l.asr)} ms)` : "") +
      (l.llm_first_token ? ` · first token ${Math.round(l.llm_first_token)} ms` : "") +
      (l.tts_first_audio ? ` · first audio ${Math.round(l.tts_first_audio)} ms after that` : "");
  }
  busy = false;
  setButtons();
}

/** Say the queued items in order: spoken sentences, or text-paced words when muted. */
async function speak(queue, isDone, reactAt, getState, bot) {
  // Wait for the state, then let the reaction show before the reply face takes over.
  while (reactAt() === null && !isDone()) await sleep(30);
  if (reactAt() !== null) {
    const wait = REACT_HOLD_MS - (performance.now() - reactAt());
    if (wait > 0) await sleep(wait);
  }
  for (;;) {
    if (!queue.length) {
      if (isDone()) break;
      await sleep(30);
      continue;
    }
    if (!bot.msg) {
      bot.msg = addMessage("bot");
      bot.msg.bubble.classList.add("cursor");
      const s = getState();
      if (s) face.respond(s.emotion, s.confidence);
    }
    const item = queue.shift();
    if (item.words) {
      for (const w of item.words) await sayWord(w, bot);
    } else {
      await saySentence(item, bot);
    }
  }
  bot.msg?.bubble.classList.remove("cursor");
  await sleep(1800);
  face.neutral();
}

/** One word, text-paced: about one mouth pulse per syllable. */
async function sayWord(w, bot) {
  bot.msg.el.dataset.voice ||= "text-paced";
  bot.msg.bubble.textContent += w;
  scrollDown();
  const letters = w.replace(/[^A-Za-z']/g, "").length;
  for (let i = 0; i < Math.max(1, Math.round(letters / 4)); i++) {
    face.pulse(0.55 + Math.random() * 0.3);
    await sleep(150);
  }
  if (/[.!?]["')]?\s*$/.test(w)) await sleep(320);
  else if (/[,;:]\s*$/.test(w)) await sleep(160);
}

/** One spoken sentence: play it, open the mouth with its loudness, show its words over its length. */
async function saySentence(item, bot) {
  const sep = bot.msg.bubble.textContent && !/\s$/.test(bot.msg.bubble.textContent) ? " " : "";
  const words = item.text.split(/(?<=\s)/);
  let buffer = null;
  try { buffer = await item.buffer; } catch {}
  bot.msg.el.dataset.voice = buffer && !muted ? "spoken" : "text-paced";  // for debugging and tests
  if (!buffer || muted) {  // muted mid-reply, or undecodable audio: fall back to text pacing
    bot.msg.bubble.textContent += sep;
    for (const w of words) await sayWord(w, bot);
    return;
  }
  ctx().resume().catch(() => {});
  const src = ctx().createBufferSource();
  src.buffer = buffer;
  const analyser = ctx().createAnalyser();
  analyser.fftSize = 512;
  src.connect(analyser);
  analyser.connect(ctx().destination);
  currentSource = src;
  const data = new Float32Array(analyser.fftSize);
  const t0 = performance.now();
  const ms = buffer.duration * 1000;
  let shown = 0;
  bot.msg.bubble.textContent += sep;
  const ended = new Promise((r) => { src.onended = r; setTimeout(r, ms + 400); });
  src.start();
  let playing = true;
  ended.then(() => { playing = false; });
  while (playing) {
    analyser.getFloatTimeDomainData(data);
    let sum = 0;
    for (const v of data) sum += v * v;
    face.pulse(Math.min(1, Math.sqrt(sum / data.length) * 7));   // mouth follows the voice
    const due = Math.min(words.length, Math.ceil(((performance.now() - t0) / ms) * words.length));
    while (shown < due) bot.msg.bubble.textContent += words[shown++];
    scrollDown();
    await new Promise((r) => requestAnimationFrame(r));
  }
  while (shown < words.length) bot.msg.bubble.textContent += words[shown++];
  currentSource = null;
}

// ------------------------------------------------------------------ typing

const input = $("input");
let typingTimer = null;

function autosize() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 160) + "px";
}

input.addEventListener("input", () => {
  autosize();
  setButtons();
  if (busy) return;
  // The face listens while you type; each keystroke is a small pulse.
  face.setListening(true);
  face.setLevel(0.45);
  ping();
  clearTimeout(typingTimer);
  typingTimer = setTimeout(() => face.setListening(false), 1500);
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    sendText();
  }
});

function sendText() {
  const text = input.value.trim();
  if (!text || busy) return;
  input.value = "";
  autosize();
  clearTimeout(typingTimer);
  const msg = addMessage("user");
  msg.bubble.textContent = text;
  msg.meta.innerHTML = chip("…");
  runTurn(fetch(turnUrl("/api/text"), { method: "POST", body: JSON.stringify({ text }) }), msg);
}

$("send").addEventListener("click", sendText);

// ------------------------------------------------------------------ talking

let rec = null;

async function startTalking() {
  if (busy || rec) return;
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    status(`<b>no microphone</b> ${escapeHtml(e.message)}`);
    return;
  }
  const chunks = [];
  const recorder = new MediaRecorder(stream);
  const m = meter(ctx().createMediaStreamSource(stream));
  recorder.ondataavailable = (e) => chunks.push(e.data);
  rec = { recorder, stream, m, chunks, t0: performance.now() };
  recorder.start();
  startPinging();
  face.setListening(true);
  $("mic").classList.add("on");
  setButtons();
  status("<b>listening</b> (click the mic again, or release Space, to send)");
}

function stopTalking() {
  if (!rec) return;
  const { recorder, stream, m, chunks, t0 } = rec;
  rec = null;
  stopPinging();
  $("mic").classList.remove("on");
  recorder.onstop = () => {
    m.stop();
    stream.getTracks().forEach((t) => t.stop());
    face.setListening(false);
    if (performance.now() - t0 < 400) {
      status("too short: talk for a moment before sending");
      setButtons();
      return;
    }
    const msg = addMessage("user");
    msg.bubble.textContent = "Transcribing…";
    msg.bubble.classList.add("pending");
    const blob = new Blob(chunks, { type: recorder.mimeType });
    runTurn(fetch(turnUrl("/api/utterance"), { method: "POST", body: blob }), msg);
  };
  recorder.stop();
}

$("mic").addEventListener("click", () => (rec ? stopTalking() : startTalking()));
window.addEventListener("keydown", (e) => {
  if (e.code === "Space" && !e.repeat && document.activeElement !== input &&
      !["INPUT", "BUTTON", "TEXTAREA"].includes(e.target.tagName)) {
    e.preventDefault();
    startTalking();
  }
});
window.addEventListener("keyup", (e) => {
  if (e.code === "Space" && rec && document.activeElement !== input) { e.preventDefault(); stopTalking(); }
});

// ------------------------------------------------------------------ demo clips

/** Play a MELD clip (the face "listens"), then send it through the pipeline. */
async function playClip(clip) {
  if (busy) return;
  busy = true;
  setButtons();
  const msg = addMessage("user");
  msg.bubble.textContent = `Playing MELD clip ${clip}…`;
  msg.bubble.classList.add("pending");
  const audio = new Audio(`/audio/test/${clip}.wav`);
  const m = meter(ctx().createMediaElementSource(audio));
  m.analyser.connect(ctx().destination);
  face.setListening(true);
  startPinging();
  status(`<b>listening</b> to ${clip}`);
  ctx().resume().catch(() => {});  // not awaited: it can stall when there is no audio device
  // Move on when the clip ends, fails, or overruns its length (so a turn never hangs).
  await new Promise((resolve) => {
    audio.onended = resolve;
    audio.onerror = resolve;
    audio.onloadedmetadata = () => setTimeout(resolve, (audio.duration + 0.5) * 1000);
    audio.play().catch(resolve);
  });
  m.stop();
  stopPinging();
  face.setListening(false);
  await runTurn(fetch(turnUrl("/api/clip"), { method: "POST", body: JSON.stringify({ split: "test", clip }) }), msg);
}

function setButtons() {
  $("send").disabled = busy || !input.value.trim();
  $("mic").disabled = busy && !rec;
  $("reset").disabled = busy;
  document.querySelectorAll(".clips button").forEach((b) => (b.disabled = busy));
}

let demoClips = [];
function addClipButtons() {
  const box = $("clips");
  if (!box) return;
  box.innerHTML = `<span class="hint" style="width:100%">or try a MELD clip (gold transcript):</span>`;
  for (const c of demoClips) {
    const b = document.createElement("button");
    b.textContent = c;
    b.onclick = () => playClip(c);
    box.appendChild(b);
  }
}

$("reset").addEventListener("click", async () => {
  await fetch("/api/reset", { method: "POST" });
  $("messages").innerHTML = `<div class="empty" id="empty"><p>New conversation. Type or talk.</p>
    <div class="row clips" id="clips" style="justify-content:center"></div></div>`;
  addClipButtons();
  face.neutral();
  status("new conversation");
});

fetch("/api/demo_clips").then((r) => r.json()).then(({ clips }) => {
  demoClips = clips;
  addClipButtons();
  setButtons();
  status("ready");
  // Demo links: #clip=dia113_utt10 plays that clip on load; #say=... sends a typed message.
  const auto = location.hash.match(/^#clip=(dia\d+_utt\d+)$/);
  if (auto) playClip(auto[1]);
  const say = location.hash.match(/^#say=(.+)$/);
  if (say) { input.value = decodeURIComponent(say[1]); sendText(); }
}).catch(() => status("server not reachable: run python face_server.py"));

// ------------------------------------------------------------------ debug panel

let previewEmotion = "neutral";
for (const e of FACE_EMOTIONS) {
  const b = document.createElement("button");
  b.textContent = e;
  b.onclick = () => { previewEmotion = e; face.react(e, +$("conf").value); };
  $("emotions").appendChild(b);
}
$("conf").addEventListener("input", () => {
  $("confval").textContent = (+$("conf").value).toFixed(2);
  face.react(previewEmotion, +$("conf").value);
});
$("asreply").onclick = () => face.respond(previewEmotion, +$("conf").value);
$("talktest").onclick = async () => {
  for (let i = 0; i < 12; i++) { face.pulse(0.55 + Math.random() * 0.3); await sleep(150); }
};

// #preview=joy:0.9 (or anger:0.8:reply, add :talk) shows one expression, for screenshots.
const preview = location.hash.match(/^#preview=(\w+):([\d.]+)(:reply)?(:talk)?$/);
if (preview) {
  const [, emo, conf, asReply, talk] = preview;
  asReply ? face.respond(emo, +conf) : face.react(emo, +conf);
  face.snap();
  if (talk) setInterval(() => face.pulse(0.8), 30);
}
