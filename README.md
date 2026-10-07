# Emotion-aware character: text + audio emotion recognition on MELD

A small, local, real-time character you can talk to. It listens to one conversational
utterance (your voice, or a typed message), recognizes one of seven emotions from **what
you said and how you said it**, shows that on an animated face, and answers in its own
voice with a short reply whose tone fits the emotion.

The emotion model is two frozen pretrained encoders (RoBERTa for words, WavLM for voice)
and a small trained fusion head, evaluated on the official MELD test split. Everything
runs on one laptop (Apple M5, 24 GB), offline, with 2.09B parameters in total against a
6B budget.

![the chat page](docs/images/face_chat.png)

| | |
| --- | --- |
| Emotion model, MELD test | **0.614 ± 0.006** weighted F1 (fused text + audio, 3 seeds); text alone 0.613, audio alone 0.469 |
| What audio adds | text without dialogue context 0.609 → **0.621** with audio |
| Emotion state after you stop talking | median **46 ms** with a transcript, 152 ms including Whisper |
| Character starts speaking | median **752 ms** after you stop (Whisper, LLM and TTS included) |
| Size | 2,086,294,769 parameters on the inference path (35% of the 6B limit), peak memory 5.49 GB |
| Checks | `evaluation/11_system_check.py`: requirements and live-system checks, all passing |

**Contents:** [Talk to the character](#talk-to-the-character) ·
[Results](#results) · [Where the model struggles](#where-the-model-struggles) ·
[Prior work: EMODE](#prior-work-emode-and-what-this-project-improves) ·
[Reproduce everything](#reproduce-everything) · [Repository layout](#repository-layout) ·
[Architecture](#architecture) · [Real-time definition](#real-time-definition) ·
[Reply quality](#reply-quality) · [Decisions](#decisions-and-trade-offs) ·
[External components](#external-components-and-ownership) · [Limitations](#limitations) ·
[Completed vs left out](#completed-vs-intentionally-left-out) · [Next steps](#next-steps-with-the-full-timebox)

## Talk to the character

### Set up once (about 15 minutes, mostly downloads)

```bash
# Python 3.11 and ffmpeg (brew install ffmpeg on macOS)
uv venv --python 3.11 .venv && source .venv/bin/activate    # or: python3.11 -m venv .venv
uv pip install -r requirements.txt                          # or: pip install -r requirements.txt

python training/01_check_environment.py   # records the hardware, downloads every model's weights (~10 GB)
```

The trained emotion heads are already in `models/`, so the character needs no training
and no dataset. (The MELD demo clips in the chat need the data step under
[Reproduce everything](#reproduce-everything); typing and talking do not.)

### Start it

```bash
python app/server.py          # loads the models (~10 s), then: face ready: http://localhost:8000
```

Open **http://localhost:8000** in Chrome, Safari or Firefox on the same machine. Nothing
leaves the laptop: the page talks only to that local server, which binds to 127.0.0.1.

### Interact

| To | Do |
| --- | --- |
| **Type** | Write in the box and press **Enter** (Shift+Enter for a new line). Typed messages are read from the words only. |
| **Talk** | Click the **mic**, speak, click it again to send. Or hold **Space** (outside the text box) while you speak. Allow microphone access the first time. Voice is read from your tone *and* your words (Whisper transcribes it). |
| **Mute** | The **speaker** button in the header. The character keeps talking on screen, word by word; the choice is remembered. |
| **Start over** | **New chat** clears the conversation, which otherwise gives the model the last two lines as context. |
| **Try without a mic** | Click a **MELD clip** under the empty chat (needs the dataset): the clip plays, the face listens, then reacts. |
| **Look inside** | **Debug** (under the face): preview every expression at any confidence, and the full JSON state of the last turn. |

What you will see on each turn:

1. **While you type or talk** the eyes widen a little and pulse with your voice (or keystrokes).
2. **Thinking:** a glance up and to the right while it transcribes and classifies.
3. **Reaction:** the emotion it heard, shown for about a second. The less sure it is, the
   closer the face stays to where it already was (the expression is blended by the model's
   calibrated confidence).
4. **Reply:** an *empathic* face, not a mirror (calm and steady when you are angry, gentle
   when you are sad), while it speaks. The mouth follows the loudness of its voice.
5. **Mood:** instead of snapping back to its resting smile, the face settles into a mood
   shaped by the conversation so far. Each reply pulls the mood toward that reply's face
   (by up to 60%, scaled by confidence, so an unsure reading barely moves it), and the mood
   fades back to the resting smile with a one-minute half-life. A sad conversation stays
   gently subdued between messages; a cheerful one stays bright. The line under the status
   shows it ("mood: gentle (after sadness), 41% and fading"). **New chat** resets it.
   Idle blinks and small glances continue throughout.

Every expression, at high and low confidence, and the reply faces:
[docs/images/face_expressions.png](docs/images/face_expressions.png).

Each of your messages gets chips showing what it heard (e.g. `anger 0.73 · voice · tone +
words · tone changed it from joy`), and each reply shows its timing. If it hears nothing
usable (a muted mic, an empty transcript) it says so and asks you to repeat, instead of
guessing an emotion.

Good to know: the mic is off while the character speaks (it never hears itself); the very
first turn after loading, or after a long idle spell, is a little slower (the page keeps the
GPU warm while you talk, see [Real-time definition](#real-time-definition)).

### Command-line demo

`demo.py` runs one utterance through the same pipeline and prints the JSON state before
streaming the reply (full output: `evals/system/demo_trace.txt`):

```bash
python demo.py --meld test dia113_utt10            # a MELD clip with its gold transcript
python demo.py --wav my_clip.wav                   # any clip, transcribed by Whisper
python demo.py --wav my_clip.wav --text "I'm fine" # any clip, your transcript
```

```
== input ==
audio: data/wav/test/dia113_utt10.wav
context:
  Chandler: I'm not a dropper!
  Ross: It's really a uh-uh three person game, y'know?
gold label (MELD annotation, for reference only): anger

== JSON state (emitted before the reply) ==
{ "transcript": "It's throwing and catching!", "emotion": "anger", "confidence": 0.7264,
  "probs": {"anger": 0.7264, "joy": 0.1951, "neutral": 0.0393, ...},
  "text_only_emotion": "joy", "audio_changed_prediction": true, "heard": true,
  "latency_ms": {"text_enc": 13.5, "audio_enc": 50.6, "heads": 1.6, "state_total": 66.7} }

== reply (streamed) ==
Gotcha, sounds like we need some practice. Let's work on it together.
```

The words alone read as joy; with the voice, the model says anger, which is the gold
label. The reply model never sees the word "anger", only tone guidance ("stay calm and
steady, acknowledge the problem plainly, and offer to help").

## Results

Full report: **[evals/classifiers/comparison.md](evals/classifiers/comparison.md)**
(`evaluation/01_compare_classifiers.py`). Seven models on the same 2,610 MELD test
utterances; mean ± std over 3 training seeds; every choice made on dev.

![overall metrics](evals/classifiers/charts/overall_metrics.png)

| Model | Input | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | utterance embedding | 0.573 ± 0.000 | 0.398 ± 0.000 | 0.397 ± 0.000 | 0.390 ± 0.000 | 0.564 ± 0.000 |
| Text: utterance | words | 0.616 ± 0.001 | 0.455 ± 0.010 | 0.436 ± 0.008 | 0.436 ± 0.007 | 0.609 ± 0.002 |
| Text: + context | words + previous two lines | 0.619 ± 0.005 | 0.450 ± 0.004 | 0.439 ± 0.001 | 0.433 ± 0.005 | 0.613 ± 0.005 |
| Audio | voice (WavLM) | 0.482 ± 0.007 | 0.307 ± 0.008 | 0.289 ± 0.004 | 0.291 ± 0.002 | 0.469 ± 0.002 |
| Late fusion | average of text + context and audio | 0.635 ± 0.003 | 0.482 ± 0.010 | 0.423 ± 0.010 | 0.436 ± 0.008 | 0.618 ± 0.005 |
| Fused, no context | words + voice | 0.619 ± 0.003 | 0.464 ± 0.006 | 0.465 ± 0.004 | 0.450 ± 0.004 | 0.621 ± 0.003 |
| **Fused (deployed)** | words + context + voice | **0.619 ± 0.007** | 0.449 ± 0.009 | 0.438 ± 0.008 | 0.441 ± 0.008 | **0.614 ± 0.006** |

- **Words carry most of the signal; the voice alone is weak** (0.469), as usual on MELD:
  sitcom audio has a laugh track, overlapping speakers and very short clips.
- **The voice helps when the text model has no dialogue context**: 0.609 → 0.621 weighted
  F1 and 0.436 → 0.450 macro F1, against a seed spread of 0.002 to 0.007.
- **With context, the voice adds little on test** (0.613 → 0.614), although it adds 0.015
  on dev, where the deployed model was chosen. I report that rather than switch models after
  seeing test. Context and tone seem to resolve the same ambiguous short lines.
- **Macro scores are far below weighted ones** because neutral is 48% of test and the rare
  emotions (fear 50 utterances, disgust 68) stay hard for every model.

**Accuracy per emotion** of the deployed fused model (mean over 3 seeds; recall is the
share of that emotion's test utterances it gets right):

| Emotion | Test utterances | Recall (accuracy on it) | Precision | F1 |
| --- | ---: | ---: | ---: | ---: |
| neutral | 1,256 | 0.787 | 0.757 | 0.772 |
| joy | 402 | 0.575 | 0.549 | 0.559 |
| surprise | 281 | 0.543 | 0.512 | 0.525 |
| anger | 345 | 0.446 | 0.502 | 0.472 |
| sadness | 208 | 0.324 | 0.416 | 0.364 |
| fear | 50 | 0.220 | 0.208 | 0.213 |
| disgust | 68 | 0.172 | 0.197 | 0.181 |
| **all** | **2,610** | accuracy **0.619 ± 0.007** | | weighted **0.614** |

No emotion is close to perfect, and 100% is not a realistic target on MELD: 29% of its
train utterances are misclassified by every model in cross-validation, many of them lines
whose label is debatable even for a person (see [Where the model struggles](#where-the-model-struggles)).
- **Confidence is honest**: temperature scaling (fit on dev) brings the fused model's test
  calibration error from 0.057 to 0.028, which is what the face's confidence blending relies on.

Per-emotion precision, recall and F1 for every model are in
[the comparison](evals/classifiers/comparison.md#per-emotion), and every model has its own
confusion matrix in [evals/classifiers/confusion_matrices/](evals/classifiers/confusion_matrices):

![confusion matrices](evals/classifiers/charts/confusion_matrices_all.png)

## Where the model struggles

Full analysis: **[evals/struggles/analytics.md](evals/struggles/analytics.md)**
(`evaluation/03_analyze_errors.py`). It uses the deployed model (one seed: accuracy 62.9%
on test) and every test utterance (`evals/struggles/test_predictions.csv`).

![class outcomes](evals/struggles/charts/class_outcomes.png)

1. **Rare emotions collapse into neutral.** Disgust, fear and sadness are right only 13%,
   24% and 32% of the time; 37% of disgust and 38% of sadness is called neutral. A real
   emotion called neutral is 33% of all errors.
2. **High-arousal emotions trade places.** Joy → anger and anger → surprise are two of the
   ten most common mistakes.
3. **Long lines are harder**: 72% accuracy on 1-2 word utterances, 53% on 21+ words.
4. **Live speech costs the most**: with Whisper's transcripts accuracy falls to 55.2%
   (joy recall −24 points, surprise −20, anger −17). Whisper's word error rate on MELD is 35.3%.
5. **The voice helps and hurts in equal measure** overall (165 text errors fixed, 147 right
   answers broken); it helps most on neutral and sadness, hurts most on surprise and joy.
6. **Part of the error is the labels.** 29% of train utterances are misclassified in every
   cross-validation fold of every seed, including 72% of disgust and 68% of fear; many read
   as ambiguous ("No!" labeled disgust). Training harder on those failures makes the model
   worse (below).
7. **But the model knows when it is unsure**: predictions with 0.9-1.0 confidence are right
   94% of the time, those under 0.4 only 34%.

### Would RL, or DAgger-style training on the failures, help?

I tested "find where the model fails and train harder there" in its legitimate form, with
failures found on *train* by cross-validation, never on test
([evals/experiments/train_on_failures.md](evals/experiments/train_on_failures.md)):

| Variant of the fused model (3 seeds) | Dev weighted F1 | Test weighted F1 |
| --- | ---: | ---: |
| Baseline | 0.612 ± 0.002 | 0.614 ± 0.006 |
| Misclassified train utterances weighted ×2 | 0.596 ± 0.007 | 0.608 ± 0.004 |
| Misclassified train utterances weighted ×3 | 0.582 ± 0.004 | 0.592 ± 0.004 |
| Focal loss, γ = 2 | 0.608 ± 0.003 | 0.617 ± 0.006 |
| Lower the neutral logit (offset tuned on dev) | best offset was 0.0 | same as baseline |

The more weight on failures, the worse: the failures are largely ambiguous labels.
**DAgger** fixes compounding errors in *sequential* decisions, where a learner's own actions
change what it sees next; classifying one utterance at a time has no such loop, and **RL**
would only optimize "match the label", which cross-entropy already does. Where they *would*
fit is committing to an emotion while the person is still talking (a sequential
wait-or-commit decision, with the full-utterance model as the expert), listed under next
steps. What would help more: a stronger ASR model, then fine-tuning the text encoder.

## Prior work: EMODE, and what this project improves

[EMODE](https://github.com/ShrishChou/EMODE) is my earlier speech emotion project. It pools
four acted-speech corpora (CREMA-D, ESD, RAVDESS, TESS) under one 5-emotion taxonomy
(angry, happy, sad, neutral, surprised) and trains a 4-stage CNN with GroupNorm, from
scratch, on 64-bin log-mel spectrograms of 2-second windows (three crops per clip). Its
goal is a teacher model for later distillation; the distillation step is not in that
repository yet, and its README reports no test metrics.

**Reused from EMODE:** the audio normalization in its `dataset.py` (downmix to mono, zero
NaNs, subtract the mean, divide by the standard deviation with a 1e-3 floor, clamp to ±5).
It is `emode_normalize` in [src/audio.py](src/audio.py) and is the only normalization the
voice gets here.

**What this project changes and improves:**

| | EMODE | This project |
| --- | --- | --- |
| Speech | Isolated emotional sentences, acted in studios (four corpora) | Scripted but conversational multi-party dialogue in context (MELD, *Friends*), with laugh track and overlaps |
| Emotions | 5 | 7 (adds fear and disgust, the hardest) |
| Modalities | Voice only | Voice **and** words, plus the previous two lines of dialogue |
| Audio model | CNN trained from scratch on log-mels | Self-supervised WavLM (pretrained on unlabeled speech), all 13 layers kept with a learned weighting |
| Input length | 2 s windows | The whole utterance (up to 15 s), padding masked out |
| Evaluation | Training and a report script; no published numbers | Official splits, 7-model ablation with 3 seeds, calibration, per-emotion and error analysis, Whisper-transcript evaluation |
| Output | A classifier | A real-time character: JSON emotion state, LLM reply, speech, animated face, measured latency and memory |

What EMODE does that this project does not: its cross-corpus pooling targets robustness
across recording conditions, and this project is trained on one sitcom. Using an EMODE-style
multi-corpus model as the voice branch is the first next step.

## Reproduce everything

Every number in this README comes from a file in `evals/`, produced by the scripts below
(run from the repository root; times on the Apple M5).

**Data** (MELD audio with the official splits, 1.5 GB; NOTES.md explains this mirror):

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('ajyy/MELD_audio', repo_type='dataset', local_dir='data/raw/meld_audio_hf')"
cd data/raw/meld_audio_hf && for s in train dev test; do tar -xzf archive/$s.tar.gz; done && cd -
```

**Training** (`training/`, run in order):

| Script | Time | Does |
| --- | --- | --- |
| `01_check_environment.py` | downloads | hardware report, all model weights |
| `02_prepare_meld.py` | ~1 min | clips → 16 kHz wav, manifests with dialogue context |
| `03_extract_features.py` | ~10 min | RoBERTa and WavLM embeddings, cached once |
| `04_train_classifiers.py` | ~1 min | 5 heads × 3 seeds → `models/`, predictions for evaluation |

**Evaluation** (`evaluation/`; 01 to 04 are the core, the rest are studies):

| Script | Time | Writes |
| --- | --- | --- |
| `01_compare_classifiers.py` | <1 min | `evals/classifiers/`: metrics, charts, confusion matrices |
| `02_asr_transcripts.py` | ~6 min | `evals/asr/`: Whisper transcripts and their cost (`--splits train dev test` for 09, ~30 min) |
| `03_analyze_errors.py` | <1 min | `evals/struggles/`: where the model fails |
| `04_count_parameters.py` | ~1 min | `evals/system/params.json` |
| `05_benchmark_latency.py` | ~3 min | `evals/system/latency.json` |
| `06_idle_latency.py` | ~5 min | `evals/system/idle_latency.md` |
| `07_compare_reply_models.py` | ~10 min | `evals/replies/`: reply rules across LLMs (needs Qwen2.5-3B and Qwen3-4B cached) |
| `08_blind_reply_review.py` | ~10 min, interactive | `evals/replies/blind_review.json` |
| `09_asr_matched_training.py` | ~1 min | `evals/asr/asr_matched.md` |
| `10_train_on_failures.py` | ~5 min | `evals/experiments/train_on_failures.md` |
| `11_system_check.py` | ~10 min | `evals/system/system_check.md` (`--server http://localhost:8000` also tests a running app) |

Training is deterministic: rerunning `04` reproduces the saved models bit for bit.

## Repository layout

```
app/                 the character: server.py (local web server) + web/ (chat page, SVG face)
demo.py              one utterance from the command line
config.py            every path, model name and hyperparameter
src/                 the library
  audio.py             loading + EMODE normalization
  encoders.py          RoBERTa, WavLM, Whisper wrappers
  fusion.py            the classifier head; loading the deployed heads
  training.py          training loop, metrics, calibration
  pipeline.py          utterance -> JSON state -> reply (+ speech)
  responder.py         reply prompt, decoding rules, streaming
  reply_checks.py      sentence and emotion-word checks
  tts.py               Kokoro speech
  data.py, plots.py    small shared helpers
models/              the trained heads (<name>.pt) and meta.json (which two the app uses)
training/            steps 01-04
evaluation/          scripts 01-11
evals/               every result: classifiers/, struggles/, asr/, replies/, system/, experiments/, logs/
docs/images/         screenshots
NOTES.md             every assumption, deviation and fallback, in the order they happened
```

`evals/README.md` lists what each file there is.

## Architecture

```
            utterance: 16 kHz audio [+ transcript, + previous 2 lines]           or a typed message
                         |                                                              |
          +--------------+------------------------------+                               |
          |                                             |                               |
   no transcript? -> Whisper-small (ASR)          EMODE normalization                   |
          |                                             |                               |
   RoBERTa-base (frozen)                         WavLM-base-plus (frozen)               |
   context </s></s> utterance,                   13 hidden layers, each                 |
   mean-pool utterance tokens -> 768             mean-pooled over time -> 13x768        |
          |                                             |                               |
          |                                  learned softmax layer weights -> 768       |
          +-------------------+-------------------------+                               |
                              |                                                         |
        concat 1536 -> LayerNorm -> Linear 256 -> GELU -> Dropout -> Linear 7      text-only head
                              |                                                         |
                       softmax(logits / T)  ------------------->  JSON state  <---------+
                                                                      |
                     emotion -> tone guidance (the label word never reaches the LLM)
                                                                      |
            Qwen2.5-1.5B-Instruct: 3 example exchanges + transcript, context, tone guidance;
            stops after 2 sentences; emotion words blocked                -> streamed reply
                                                                      |
            Kokoro-82M (CPU), one sentence at a time  -> speech + face in the browser
```

Parameters, counted in code (`evals/system/params.json`):

| Component | Model | Parameters | Runs |
| --- | --- | ---: | --- |
| Text encoder | `roberta-base` (no pooler) | 124,055,040 | always |
| Audio encoder | `microsoft/wavlm-base-plus` | 94,381,936 | voice input |
| Text-only head | own MLP (`models/text_ctx.pt`) | 200,199 | typed input |
| Fused head | own MLP + layer weights (`models/fused.pt`) | 398,356 | voice input |
| ASR | `openai/whisper-small` | 241,734,912 | voice without a transcript |
| Reply LLM | `Qwen/Qwen2.5-1.5B-Instruct` | 1,543,714,304 | always |
| Speech | `hexgrad/Kokoro-82M` | 81,810,022 | chat page, unless muted |
| **Total** | | **2,086,294,769** | 35% of the 6B limit |

## Real-time definition

Interaction is **turn-level**: the character reacts after you finish an utterance, as
people do. Latency runs from the moment the finished utterance reaches the pipeline.
Targets: the emotion state within 300 ms on a GPU and the first reply token within 1 s.
Because the state comes first, the face changes before any word is said.

50 MELD test clips (median 2.4 s), after 3 warm-ups, replies spoken (`evals/system/latency.json`):

| Stage | p50 (ms) | p95 (ms) |
| --- | ---: | ---: |
| ASR (Whisper-small) | 104.7 | 156.7 |
| Text encoder | 8.3 | 10.5 |
| Audio encoder | 34.0 | 61.1 |
| Heads | 0.6 | 0.8 |
| **Emotion state, transcript supplied** | **45.8** | **77.0** |
| **Emotion state, including ASR** | **151.8** | **235.8** |
| LLM first token (after the state) | 205.4 | 267.2 |
| First spoken sentence ready (after the state) | 586.3 | 964.5 |
| **End of speech → character's voice ready** | **752.4** | **1108.7** |

Both targets are met at p50 and p95. Speaking starts later than the first token because a
whole sentence must be written, then synthesized (Kokoro, on the CPU, while the GPU writes the
next sentence). Peak memory 4.87 GB process RSS, 5.49 GB on the GPU (shared on Apple Silicon).

**After idle gaps.** The benchmark runs clips back to back, but the character sits idle while
you talk, and the Apple GPU slows down when idle (`evals/system/idle_latency.md`, median of 3):

| Idle gap before the turn | State, no warm-up (ms) | State, warm-up ping 1 s before (ms) |
| ---: | ---: | ---: |
| 2 s | 134 | 57 |
| 5 s | 147 | 76 |
| 10 s | 150 | 70 |
| 20 s | 312 | 132 |

So the page pings the server every second while you type or talk (`Pipeline.ping()`).

## Reply quality

The first reply prompt put the emotion label in the prompt and asked the model not to name
it; a quarter of its replies ran past two sentences. The final prompt turns each emotion into
**tone guidance**, shows **three short example exchanges**, and **enforces** two rules in code:
generation stops after the second sentence, and emotion words (angry, upset, sad, …) are
blocked. Compliance on 50 test clips, with the pass bar fixed before running
(`evals/replies/reply_compliance.md`):

| Reply model | ≤ 2 sentences | Emotion word | Tells you how you feel | First token p50 / p95 (ms) | Passes |
| --- | ---: | ---: | ---: | ---: | --- |
| Qwen2.5-1.5B, first prompt | 76% | 4% | 2% | 182 / 234 | no |
| **Qwen2.5-1.5B, final prompt** | **100%** | **0%** | **0%** | **308 / 382** | **yes** |
| Qwen2.5-3B, final prompt | 100% | 0% | 0% | 598 / 702 | yes |
| Qwen3-4B-Instruct-2507, final prompt | 100% | 0% | 6% | 838 / 981 | yes |

The decision rule (smallest model that passes) kept the 1.5B. The sentence limit and the word
list are compliant *by construction*; "tells you how you feel" is not enforced. My own read
(one reader, not a rating) is that the 3B writes noticeably more natural replies. Whether the
emotion state makes replies better at all needs the blind check: TODO(Shrish): run
`python evaluation/08_blind_reply_review.py` (about 10 minutes) and report "preferred the
emotion-conditioned reply in X/20 (single rater)". (That comparison ran before speech output
was added; its "pipeline total" column excludes the 82M speech model.)

## Decisions and trade-offs

- **Text + audio rather than text + vision.** MELD frames often show several faces and no
  active-speaker label; the audio is always the speaker's own.
- **Frozen encoders, cached features.** Encoding the corpus takes about 10 minutes once;
  then every head and seed trains in about a minute, which made a real ablation possible.
  Fine-tuning RoBERTa would likely score higher (not measured; hours of training).
- **General-purpose encoders only.** Popular "emotion" text checkpoints list MELD in their
  training data, which would leak test labels. RoBERTa, WavLM and Whisper never saw emotion labels.
- **WavLM layer weighting.** Paralinguistic cues tend to sit in middle layers, and the best
  layer varies by task, so all 13 are kept with a learned softmax weighting (13 parameters).
  The learned weights are fairly flat (`models/meta.json`).
- **Concatenation MLP**: the simplest fusion that can learn interactions, and a fair
  comparison (the same head for every input). Late fusion is reported as a cheaper baseline.
- **Dialogue context**: the previous two lines go in as the first segment of a RoBERTa
  sentence pair, and only the utterance's own tokens are pooled. Pooling every token let the
  context swamp the utterance (dev 0.496, `evals/experiments/table_v1_ctx_pool_all_tokens.md`).
- **Calibrated confidence**, because the face and the reply both use it.
- **Tone guidance instead of labels** for the LLM, and rules enforced in decoding rather than
  requested, because a 1.5B model follows examples better than instructions.
- **Speech on the CPU** so it overlaps with the LLM on the GPU.
- **Turn-level real time** matches how people take turns and keeps each stage measurable;
  reacting *during* speech is a next step.

## External components and ownership

| Component | Source | License | Use |
| --- | --- | --- | --- |
| MELD (Poria et al., 2019) | github.com/declare-lab/MELD | GPL-3.0 | data, official splits |
| MELD audio mirror | huggingface.co/datasets/ajyy/MELD_audio | GPL-3.0 (card) | 16 kHz audio of the official clips |
| EMODE (my prior work) | github.com/ShrishChou/EMODE, `dataset.py` | own | audio normalization |
| RoBERTa-base | huggingface.co/roberta-base | MIT | text encoder |
| WavLM-base-plus | huggingface.co/microsoft/wavlm-base-plus | no license field on the card. TODO(Shrish): confirm (the microsoft/unilm code is MIT) | audio encoder |
| Whisper-small | huggingface.co/openai/whisper-small | Apache-2.0 | ASR |
| Qwen2.5-1.5B-Instruct | huggingface.co/Qwen/Qwen2.5-1.5B-Instruct | Apache-2.0 | replies |
| Kokoro-82M | huggingface.co/hexgrad/Kokoro-82M (voice `af_heart`) | Apache-2.0 | speech |
| misaki, spaCy `en_core_web_sm` | Kokoro's English front end | Apache-2.0, MIT | used by Kokoro |
| espeakng-loader | bundles espeak-ng, Kokoro's fallback for unknown words | no license field; espeak-ng is GPL-3.0. TODO(Shrish): confirm acceptable | used by Kokoro |
| Qwen2.5-3B-Instruct, Qwen3-4B-Instruct-2507 | huggingface.co/Qwen | Qwen license; Apache-2.0 | compared only, not used |
| PyTorch, transformers, scikit-learn, ffmpeg, … | `requirements.txt` | open source | |

Written for this project: everything in `app/`, `src/`, `training/`, `evaluation/`,
`demo.py` and `config.py`, and the trained heads in `models/`. The face borrows its idea (a
few procedural parameters, a one-line mouth driven by an open ratio) from Cozmo/Vector-style
procedural eyes and the M5Stack-Avatar face of Stack-chan; no code or assets were copied.

Verified rather than assumed: every parameter count, that no pretrained model is an emotion
classifier trained on MELD, and that inference makes no network call (offline mode is forced
in every script that loads a model, and the app server binds to localhost). Two result files
contain MELD test transcripts (GPL-3.0 permits this; MELD is credited above).

## Limitations

- **Domain.** *Friends* has a laugh track, studio acoustics and exaggerated delivery; a
  robot's microphone hears rooms, distance and ordinary speech.
- **Labels.** MELD labels are noisy and imbalanced (neutral 47% of train, fear and disgust
  under 3% each); many short lines are ambiguous even to people.
- **Live speech.** Whisper-small's 35.3% word error rate on MELD costs about 0.10 weighted F1
  against gold transcripts. No voice activity detection: noise that is not near-silent still
  reaches Whisper, which can invent text.
- **Reply quality** is checked for rules, not rated; the word ban can be dodged by spelling
  ("angr-y" in an adversarial test).
- **One machine.** Latency was measured on an Apple M5 (MPS); CUDA and CPU paths exist but were not run.
- **Coarse labels**: seven categories per utterance, no intensity, no mixed emotions.

## Completed vs intentionally left out

Completed:

- Frozen RoBERTa and WavLM features for all of MELD; 7 classifiers compared with 3 seeds,
  calibration, per-emotion scores and confusion matrices
- Error analysis, Whisper evaluation, Whisper-matched training, train-on-failures experiment
- `demo.py` and the chat character: type or talk, emotion chips, spoken replies with mute,
  animated face with a conversation mood that persists and fades, empathic reply faces,
  a "nothing heard" guard
- Parameter, latency, idle-latency and memory reports; reply-rule comparison across three
  LLMs; a blind review tool; an end-to-end system check

Intentionally left out:

- Vision of any kind; fine-tuning the encoders or the LLM; RL and DAgger (assessed above)
- Retraining or extending EMODE (only its normalization is reused)
- Voice activity detection (push-to-talk instead), interrupting the character, phoneme-level lip sync
- Reacting to emotion while the person is still talking
- A human rating of the replies (the blind review is ready to run)

## Next steps with the full timebox

- An EMODE-style multi-corpus voice model as the audio branch, compared with WavLM.
- A stronger ASR model for live use, measured on the same test set.
- Fine-tune RoBERTa on MELD train.
- Emotion updates during speech with an early-commit policy learned by DAgger (the
  full-utterance model as the expert).
- Silero VAD for hands-free turns, and letting the person interrupt.
- Distill the audio branch for latency; a tri-modal demo with an active-speaker face model.

## Time spent

The core prototype was built in about three hours of agent time and scoped to match; the
chat character, speech, evaluations and this reorganization came in later sessions. NOTES.md
logs every assumption, fallback and deviation in order.
TODO(Shrish): add your own review time and anything you changed.
