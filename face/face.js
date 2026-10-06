// A minimal procedural robot face: two rounded eyes and a one-line mouth, drawn in SVG.
//
// The face is a handful of numbers (PARAMS). An expression is a target value for some of
// them; every frame each number eases toward its target (~200 ms to settle), so changing
// expression mid-way never jumps. Blinks, small gaze shifts and breathing are layered on top.
//
// Core parameters (the four that carry the design):
//   eye_open     1 open, 0 closed
//   lid_tilt     +1 angry (inner lids slant down), -1 sad (outer corners droop)
//   mouth_curve  +1 smile, -1 frown
//   mouth_open   0 closed, 1 open (speech pulses take the max with this)
// Extras the expression table needs:
//   eye_scale (surprise/fear enlarge), happy (lower lid pushes up into ^ ^ arcs),
//   asym (one eye squints more: disgust), mouth_round ("o"), mouth_shift (sideways)

const NEUTRAL = {
  eye_open: 1, eye_scale: 1, lid_tilt: 0, happy: 0, asym: 0,
  mouth_curve: 0, mouth_open: 0, mouth_round: 0, mouth_shift: 0,
};

// The resting face: a slight, friendly smile rather than a flat line. Idle, a detected
// "neutral", and the blend for low confidence all settle here.
const REST = { ...NEUTRAL, mouth_curve: 0.32, happy: 0.1 };

// What the face shows when an emotion is detected (blended toward REST by 1 - confidence).
// Unlisted parameters take their NEUTRAL value, so e.g. anger has no resting smile.
const REACT = {
  neutral:  { mouth_curve: REST.mouth_curve, happy: REST.happy },
  joy:      { happy: 0.75, mouth_curve: 1.0 },
  sadness:  { eye_open: 0.55, lid_tilt: -0.8, mouth_curve: -0.7 },
  anger:    { eye_open: 0.8, lid_tilt: 0.85, mouth_curve: -0.25 },
  surprise: { eye_scale: 1.3, mouth_round: 1, mouth_open: 0.6 },
  fear:     { eye_scale: 1.12, lid_tilt: -0.75, mouth_open: 0.22, mouth_curve: -0.4 },
  disgust:  { eye_open: 0.6, asym: 0.7, lid_tilt: 0.3, mouth_curve: -0.6, mouth_shift: 0.6 },
};

// The robot's own face while it replies: empathic rather than mirrored (it does not get
// angry back). Matches the tone guidance the LLM receives (src/responder.py).
const RESPOND = {
  neutral:   { happy: 0.15, mouth_curve: 0.25 },                   // friendly
  joy:       { happy: 0.6, mouth_curve: 0.8 },                     // shares it
  sadness:   { eye_open: 0.85, lid_tilt: -0.45, mouth_curve: 0.1 }, // gentle
  anger:     { eye_open: 0.95, lid_tilt: -0.15, mouth_curve: 0.05 },// calm, steady
  surprise:  { eye_scale: 1.1, happy: 0.2, mouth_curve: 0.4 },      // curious
  fear:      { lid_tilt: -0.2, happy: 0.3, mouth_curve: 0.35 },     // reassuring
  disgust:   { eye_open: 0.9, lid_tilt: -0.1, mouth_curve: 0.0 },   // matter-of-fact
  uncertain: { lid_tilt: -0.15, mouth_curve: 0.15 },                // gentle, assumes nothing
};

const TAU_MS = 70;          // easing time constant: ~95% of the way in 200 ms
const SPEECH_TAU_MS = 60;   // speech pulses decay faster than expressions
const COLOR = "#7cf3e6";
const BG = "#0b1016";

function blend(expr, amount) {
  const out = {};
  for (const k in NEUTRAL) {
    const v = expr[k] ?? NEUTRAL[k];
    out[k] = REST[k] + (v - REST[k]) * amount;
  }
  return out;
}

class Face {
  constructor(svg) {
    this.svg = svg;
    this.cur = { ...REST };
    this.target = { ...REST };
    this.speech = 0;          // current speech mouth opening
    this.level = 0;           // smoothed mic / audio loudness, 0..1
    this.listening = false;
    this.gaze = { x: 0, y: 0 };
    this.gazeTarget = { x: 0, y: 0 };
    this.gazeHold = null;     // fixed gaze (thinking, listening) instead of idle wander
    this.blink = 1;           // multiplier on eye_open
    this.nextBlink = performance.now() + 2000;
    this.nextSaccade = performance.now() + 1500;
    this.last = performance.now();
    this.build();
    requestAnimationFrame((t) => this.frame(t));
  }

  // ---------------------------------------------------------------- public API

  /** Detected emotion: blend toward neutral by 1 - confidence. */
  react(emotion, confidence) {
    this.target = blend(REACT[emotion] || {}, Math.max(0, Math.min(1, confidence)));
  }

  /** The robot's own expression while replying. */
  respond(emotion, confidence) {
    this.target = blend(confidence < 0.5 ? RESPOND.uncertain : (RESPOND[emotion] || {}), 1);
  }

  /** Back to the resting face (a slight smile). */
  neutral() { this.target = { ...REST }; }

  /** Jump straight to the target (no easing), e.g. for screenshots. */
  snap() { this.cur = { ...this.target }; }

  /** Free-form expression for the debug panel. */
  show(params) { this.target = { ...REST, ...params }; }

  /** One speech pulse (a word or syllable). */
  pulse(strength = 0.7) { this.speech = Math.max(this.speech, strength); }

  /** Loudness 0..1 of the incoming voice, for the listening pulse. */
  setLevel(v) { this.level += (v - this.level) * 0.35; }

  setListening(on) {
    this.listening = on;
    this.gazeHold = on ? { x: 0, y: 0 } : null;
  }

  setThinking(on) { this.gazeHold = on ? { x: 0.55, y: -0.6 } : null; }

  // ---------------------------------------------------------------- drawing

  build() {
    const ns = "http://www.w3.org/2000/svg";
    const el = (tag, attrs, parent) => {
      const e = document.createElementNS(ns, tag);
      for (const k in attrs) e.setAttribute(k, attrs[k]);
      (parent || this.svg).appendChild(e);
      return e;
    };
    this.svg.setAttribute("viewBox", "-200 -150 400 300");
    const defs = el("defs", {});
    const glow = el("filter", { id: "glow", x: "-50%", y: "-50%", width: "200%", height: "200%" }, defs);
    el("feGaussianBlur", { stdDeviation: "4", result: "b" }, glow);
    const merge = el("feMerge", {}, glow);
    el("feMergeNode", { in: "b" }, merge);
    el("feMergeNode", { in: "SourceGraphic" }, merge);

    el("rect", { x: -200, y: -150, width: 400, height: 300, fill: BG });
    this.root = el("g", {});
    this.eyes = [-1, 1].map((side) => {
      const g = el("g", {}, this.root);
      return {
        side,
        body: el("rect", { fill: COLOR, filter: "url(#glow)" }, g),
        upper: el("polygon", { fill: BG }, g),   // upper lid: background-coloured mask
        lower: el("ellipse", { fill: BG }, g),   // lower lid: pushes up for happy arcs
      };
    });
    const m = el("g", { filter: "url(#glow)" }, this.root);
    this.mouthLips = el("path", { fill: COLOR, stroke: COLOR, "stroke-width": 6,
                                  "stroke-linecap": "round", "stroke-linejoin": "round" }, m);
    this.mouthO = el("ellipse", { fill: "none", stroke: COLOR, "stroke-width": 6 }, m);
  }

  frame(now) {
    const dt = Math.min(now - this.last, 100);
    this.last = now;
    const a = 1 - Math.exp(-dt / TAU_MS);
    for (const k in this.cur) this.cur[k] += (this.target[k] - this.cur[k]) * a;
    this.speech *= Math.exp(-dt / SPEECH_TAU_MS / 2.2);
    this.level *= Math.exp(-dt / 400);

    // Blinks every 3-6 s: close in ~70 ms, reopen in ~120 ms.
    if (now > this.nextBlink) {
      const t = now - this.nextBlink;
      this.blink = t < 70 ? 1 - t / 70 : t < 190 ? (t - 70) / 120 : 1;
      if (t >= 190) { this.blink = 1; this.nextBlink = now + 3000 + Math.random() * 3000; }
    }
    // Idle gaze: small saccades every 1.5-4 s, unless gaze is held.
    if (this.gazeHold) {
      this.gazeTarget = this.gazeHold;
    } else if (now > this.nextSaccade) {
      this.gazeTarget = { x: (Math.random() - 0.5) * 0.8, y: (Math.random() - 0.5) * 0.5 };
      this.nextSaccade = now + 1500 + Math.random() * 2500;
    }
    const g = 1 - Math.exp(-dt / 50);
    this.gaze.x += (this.gazeTarget.x - this.gaze.x) * g;
    this.gaze.y += (this.gazeTarget.y - this.gaze.y) * g;

    this.draw(now);
    requestAnimationFrame((t) => this.frame(t));
  }

  draw(now) {
    const p = this.cur;
    const breathe = Math.sin(now / 4000 * 2 * Math.PI) * 2;
    const listenBoost = this.listening ? 0.06 + this.level * 0.18 : 0;
    const scale = p.eye_scale + listenBoost;
    this.root.setAttribute("transform", `translate(${this.gaze.x * 16} ${this.gaze.y * 12 + breathe})`);

    for (const eye of this.eyes) {
      const W = 78 * scale, H = 96 * scale;
      const cx = eye.side * 78, cy = -22;
      const top = cy - H / 2, bottom = cy + H / 2;
      eye.body.setAttribute("x", cx - W / 2);
      eye.body.setAttribute("y", top);
      eye.body.setAttribute("width", W);
      eye.body.setAttribute("height", H);
      eye.body.setAttribute("rx", 24 * scale);

      // Disgust squints the left eye more than the right.
      const squint = eye.side < 0 ? 1 - p.asym * 0.55 : 1 - p.asym * 0.15;
      const open = Math.max(0, Math.min(1.05, p.eye_open * squint * this.blink));
      const lidY = top + (1 - open) * H;
      const tilt = p.lid_tilt * 0.38 * H;            // +: inner corner lower (angry)
      // Lid masks reach 20 px past the eye so they also cover its glow.
      const innerX = cx - eye.side * (W / 2 + 20);    // the side facing the nose
      const outerX = cx + eye.side * (W / 2 + 20);
      eye.upper.setAttribute("points",
        `${outerX},${top - 200} ${innerX},${top - 200} ${innerX},${lidY + tilt} ${outerX},${lidY - tilt}`);

      const ry = H * 0.6;
      eye.lower.setAttribute("cx", cx);
      eye.lower.setAttribute("cy", bottom + ry - p.happy * 0.6 * H);
      eye.lower.setAttribute("rx", W * 0.85);
      eye.lower.setAttribute("ry", ry);
    }

    // Mouth: a curved line that opens into a lens; narrower and taller as it opens.
    const open = Math.max(p.mouth_open * (1 - p.mouth_round), this.speech);
    const x0 = p.mouth_shift * 28, y = 78;
    const w = 74 * (1 - 0.35 * open);
    const bend = p.mouth_curve * 24;
    this.mouthLips.setAttribute("d",
      `M ${x0 - w / 2} ${y} Q ${x0} ${y + bend - open * 14} ${x0 + w / 2} ${y} ` +
      `Q ${x0} ${y + bend + open * 30} ${x0 - w / 2} ${y} Z`);
    // Crossfade line <-> "o" quickly, so a partly blended expression shows one mouth.
    const round = Math.min(1, Math.max(0, (p.mouth_round - 0.2) / 0.5));
    this.mouthLips.setAttribute("opacity", 1 - round);
    this.mouthO.setAttribute("cx", x0);
    this.mouthO.setAttribute("cy", y + 6);
    this.mouthO.setAttribute("rx", 9 + p.mouth_open * 8 + this.speech * 4);
    this.mouthO.setAttribute("ry", 10 + p.mouth_open * 14 + this.speech * 8);
    this.mouthO.setAttribute("opacity", round);
  }
}

window.Face = Face;
window.FACE_EMOTIONS = Object.keys(REACT);
