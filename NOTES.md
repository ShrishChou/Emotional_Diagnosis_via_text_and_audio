# Build notes: assumptions, deviations, fallbacks

Running log kept while building, in the order things happened. Every fallback
the plan allows and every place I deviated from it is listed here.

## Environment

- Hardware: Apple M5, 10 cores, 24 GB unified memory, no CUDA. The plan's table
  puts Apple MPS in the "Qwen2.5-1.5B-Instruct, fp16, full train split" row,
  so that is the configuration (chosen automatically in `config.py`).
- Python 3.11 venv created with `uv` (the system Python is 3.14, which is outside
  the plan's 3.10/3.11 range). Install with `uv pip install -r requirements.txt`
  or plain `pip install -r requirements.txt`.
- Library versions are newer than many tutorials assume: transformers 5.18 and
  torch 2.14. Pinned in `requirements.txt`. The transformers 5 API uses `dtype=`
  rather than `torch_dtype=` when loading models.
- The repo root *is* the project folder (the plan calls it `meld-emotion-prototype/`).

## Data (Phase 1): fallback 1 taken

- The official MELD.Raw archive (10.9 GB, web.eecs.umich.edu) ran at a few MB/s,
  then stalled at about 1.5 GB; at that rate it would have taken hours.
- **Fallback 1 (audio-only mirror on Hugging Face):** `ajyy/MELD_audio`
  (huggingface.co/datasets/ajyy/MELD_audio, 1.5 GB). It contains:
  - `train.csv`, `dev.csv` and `test.csv`, byte-for-byte the same size as the
    official `*_sent_emo.csv` files (1,105,502 / 120,071 / 290,841 bytes), with
    the same columns, official splits and Dialogue_ID / Utterance_ID.
  - One FLAC per utterance, already extracted from the MELD mp4s at 16 kHz mono
    and named `dia{D}_utt{U}.flac`, in `train/`, `dev/` and `test/` folders.
  - Extra `final_videos_test*` files in `test/`, which are not in the CSV and are ignored.
- I did not verify the audio sample by sample against the mp4s (no mp4s are
  available). TODO(Shrish): if the official archive becomes reachable,
  `01_prepare_meld.py` handles it unchanged (it looks for mp4 or flac).
- `01_prepare_meld.py` still runs every clip through ffmpeg (flac -> 16 kHz mono wav)
  as the plan describes, so both sources produce identical outputs.
- Missing clips: 2 rows have no audio (`train/dia125_utt3`, `dev/dia110_utt7`),
  listed in `results/missing_clips.txt`. Both are known-missing in MELD. Final
  sizes: 9,988 train / 1,108 dev / 2,610 test (all test rows present).
- Transcript text has cp1252 mojibake (e.g. U+0092 for the apostrophe).
  `ftfy.fix_text` repairs every case; no stray C1 characters remain afterwards.
- Long clips: 25 train / 6 dev / 6 test clips exceed 15 s and are cropped to their
  first 15 s at load time (one test clip is 305 s long, a known MELD artifact).
- Empty context (the first utterance of a dialogue) is an empty string.

## Features (Phase 2)

- RoBERTa is loaded with `add_pooling_layer=False`: the checkpoint has no trained
  pooler, and we mean-pool the last hidden state instead. This also keeps the
  parameter count to the weights that are actually used.
- `text_ctx` uses RoBERTa's sentence-pair format: `<s> context </s></s> utterance </s>`
  with left-truncation of the context to 256 tokens total. Context lines are
  joined with newlines.
- WavLM-base-plus's feature extractor has `do_normalize=False`; the EMODE
  normalization (per-utterance mean/std, clamp +-5) is the only normalization applied.
- WavLM-base-plus uses group-norm in its conv feature extractor, so zero padding
  slightly changes the embeddings of shorter clips in a batch. Checked: the
  cosine similarity between a clip encoded alone and the same clip padded in a
  batch is >= 0.996 at every layer tested. Batches are sorted by duration to
  keep padding small, as the plan says.

## Training and evaluation (Phase 3)

- Heads train on CPU (they are tiny and all features fit in memory); about 45 s
  for every configuration and seed together. Reruns reproduce the same metrics
  (the script was run three times; every configuration present in two runs
  printed identical dev and test numbers).
- **Deviation: `text_ctx` pooling.** The first run mean-pooled over *all* tokens
  (context + utterance), as the plan literally says ("same pooling"). That made
  `text_ctx` much *worse* than `text` on dev (0.496 vs 0.593 weighted F1): the
  two context lines are usually longer than the utterance and swamp the average.
  I switched to pooling over the current utterance's tokens only (context still
  shapes them through self-attention). Dev then rose to 0.597. The decision was
  made on dev numbers. The first run's full table is kept for transparency in
  `results/table_v1_ctx_pool_all_tokens.md`. I did look at that run's test column
  too; nothing was tuned on it.
- With the fix, `text_ctx` beats `text` on dev, so the main `fused` model uses
  `text_ctx`. On test, `fused` (0.614) is no better than `text_ctx` (0.613).
  Reported as is (the plan: "report that plainly").
- **Addition: `fused_utt` configuration** (utterance-only text + audio). Because
  the dev-selected text variant changed, I added this explicitly so the table
  shows what audio adds without dialogue context (0.609 -> 0.621 on test). It is
  never used for any selection. Reading: context and tone overlap. Both
  disambiguate the same short utterances ("Nothing!", "What?").
- Class weights are 1/sqrt(frequency), rescaled to mean 1.
- Early stopping restores the best-dev-epoch weights. The saved checkpoint for
  each of text-only and fused is the best-on-dev seed of three.
- Temperature scaling: one scalar T fit by LBFGS on dev NLL. T = 1.27 (the head
  is overconfident). Test ECE is reported too, because dev ECE is measured on
  the data T was fit on.
- The logistic-regression sanity baseline uses standardized utterance embeddings,
  default C, no class weights.
- `examples.md` picks the 3 + 3 test utterances with the largest change in the
  probability of the gold label, among those where exactly one of text-only and
  fused is right. Selection is mechanical, not hand-picked.
- One transcript in examples.md, "I'm trppd... in an ATM vstbl... wth", is
  spelled that way in the MELD CSV itself (checked the raw bytes), not a cleaning bug.

## Demo, parameters, latency (Phase 4)

- Inference scripts set `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` before
  importing anything, so the inference path cannot touch the network. Weights are
  downloaded once by `scripts/00_check_env.py`.
- RoBERTa without its pooler: 124,055,040 parameters (the checkpoint's pooler is
  untrained and unused; with it the count would be ~124.6M).
- Qwen2.5-1.5B-Instruct ties input and output embeddings, and
  `model.parameters()` counts the shared tensor once: 1,543,714,304.
- The demo runs one untimed warm-up pass (1 s of noise plus a short reply) before
  the timed utterance. On MPS the first call of each kernel shape is slow
  (first token about 3 s cold vs 0.13 s warm). `--no-warmup` disables it. The
  benchmark uses 3 warm-up clips as the plan says.
- Latency timing starts when the finished utterance is handed to the pipeline
  (audio file already written), i.e. end of speech. MPS work is synchronized
  before reading the clock.
- Memory on Apple Silicon: GPU memory is unified with system RAM, so "VRAM" is
  reported as `torch.mps.driver_allocated_memory`, sampled every 20 ms. Process
  RSS is reported from both psutil sampling and getrusage.
- Reply prompt was iterated once on a single demo clip: the first version made
  Qwen-1.5B describe the scene in the third person ("The conversation seems
  lively..."). The second version tells it to speak directly to the person. No
  further prompt tuning. The 1.5B model still sometimes paraphrases the emotion
  ("you're frustrated") despite the instruction not to name it.
- LLM sampling: temperature 0.7 (plan), plus top_p 0.9 (my addition, the Qwen default).

## Reply prompt rework and LLM comparison (after the first review)

- First prompt: label + confidence in the prompt, "never name the emotion". Measured
  on 50 clips: 76% of replies within two sentences, 4% used an emotion word, 2%
  announced a feeling. (My earlier impression from 20 replies, "often names the
  emotion", overstated it; length was the main problem.)
- Final prompt ("behavior" style in `src/responder.py`): label → tone guidance, three
  invented few-shot exchanges, a stopping criterion after two sentences, a blocked
  emotion-word list, T 0.4, 40 tokens. Both styles stay in the code so the comparison
  reproduces.
- Judgment call: the rule became "don't tell them how they feel" rather than "never
  reflect a feeling". A gentle reflection ("that sounds annoying") is fine. The metric
  `announces_feeling` only catches attributions to the listener ("you seem upset").
- transformers' built-in `bad_words_ids` processor doubled generation time on MPS
  (many small GPU ops per step). Replaced with `BannedWords`, which copies the last few
  tokens to the CPU once per step and applies one mask. First token went from ~0.48 s
  to ~0.25 s on the profiling clip. The ban holds, but can be dodged by spelling
  ("angr-y" in an adversarial test).
- Pass bar and decision rule were written into `06_response_samples.py` before any run.
  Qwen2.5-1.5B passes, so it stays, even though my own read prefers the 3B's replies.
- The 1.5B sometimes falls back to assistant phrasing ("please provide more details so
  I can assist"). Not addressed.

## Error analysis and improvement experiments

- `09_error_analysis.py` uses the saved single-seed checkpoints, so its numbers differ
  slightly from the 3-seed means in table.md (fused 0.622 vs 0.614 ± 0.006).
- `10_improvement_experiments.py`: hard-example upweighting uses out-of-fold
  predictions on train; early stopping inside each fold uses dev, as everywhere else.
  Test is reported for every variant, but no choice was made on it. The neutral-offset
  grid search on dev picked 0.0 (no change).
- `11_asr_matched_training.py`: first run crashed after transcribing train (duplicate
  `asr_text` column on merge); fixed and rerun, train transcripts reused from cache.
  Gain on Whisper test transcripts: 0.521 → 0.535. The main model stays gold-trained
  to keep the results table comparable.
- `07_asr_eval.py` first crashed on a runaway Whisper transcript longer than the
  256-token window (`truncation="only_first"` cannot cut the utterance). The text
  encoder now caps the utterance at its last 192 tokens; the longest gold utterance is
  90 tokens, and re-encoding test reproduced the cached features exactly (max abs diff 0.0).
- Two copies of 07 briefly ran at once (my error chaining background commands); I
  killed the older one before either wrote results.
- Latency was re-measured after the prompt change. The first run's large encoder p95s
  (128 ms text, 192 ms audio) did not recur (16 ms and 80 ms).

## Chat face, typed input, system check

- `face_server.py` uses only the standard library (`http.server`): one POST per turn,
  streamed back as newline-delimited JSON, so no new dependencies. It binds to
  127.0.0.1 and serves only `face/` plus MELD wavs matching `dia\d+_utt\d+`.
- Typed messages use the text-only head. Its temperature, fit on dev in
  `03_train_eval.py`, came out at 1.013 (dev ECE 0.030 → 0.034, so it was already
  calibrated). Re-running 03 with this addition reproduced every existing metric exactly.
- The chat history (last two lines, "Person: …" / "Robot: …") is the classifier's
  context for the next turn. MELD context lines use character names, so this is a
  mild train/serve mismatch; not measured.
- Face expression numbers were tuned by eye on headless-Chrome screenshots. Fear
  first looked almost identical to surprise; it now has worried inner lids and a small
  open frown instead of the round mouth.
- Idle slowdown: found when the first browser turn took 1.1 s for the state that
  took 85 ms back to back. One manual browser trial after 60 s idle: state 1,091 ms
  without pings, 93 ms with them (server log, not saved to results/, so not in the README). `12_idle_latency.py` quantifies it; the page now pings
  `/api/ping` every second while the person talks or types. The ping is skipped when
  a turn is running, so it never delays one.
- System check, first run: an empty transcript gave anger 0.92 (RoBERTa on "" is out
  of distribution). Added the "nothing heard" state (uniform probabilities, fixed reply,
  no LLM call) and an RMS gate before Whisper (5e-4; the quietest real MELD clip is 7e-4).
- I changed one system-check criterion after seeing its result: "silence or noise gives
  low confidence" became "silence or noise does not trigger a strong emotion". Both came
  out neutral at 0.99, which is safe for the face, so the original rule was the wrong test.
- The README-number check counts a README number as traced when it appears in a
  results file, as a percentage of one, or in billions. Two are untraced by design:
  0.014 (a difference of two table values) and 2507 (part of a model name).

## Speech output and resting face

- Kokoro-82M runs on the CPU: about 0.26 to 0.42 s per sentence for 3 to 5 s of
  audio, and it leaves the GPU to the LLM, which keeps generating the next sentence
  while one is synthesized (the LLM runs in its own thread). The first synthesis after
  loading is about twice as slow, so `Pipeline.warmup()` includes one.
- Kokoro's English front end (misaki) downloaded the spaCy model `en_core_web_sm`
  from GitHub on first use. That would be network access at inference time, so the
  wheel is pinned in requirements.txt, and 00_check_env.py fetches Kokoro's weights.
  Verified that synthesis works with HF_HUB_OFFLINE=1.
- Sentences are cut with the same `sentence_ends` function that enforces the two-
  sentence limit, so the spoken text always equals the written reply (checked in 13).
- `llm_total` in latency.json now includes time the token loop waited on synthesis.
  `llm_first_token` is unaffected (it comes before any synthesis).
- Muting skips synthesis on the server (the page leaves `?tts=1` off), so a muted
  page costs nothing extra. Muting mid-sentence stops the audio at once; the rest of
  the reply is shown text-paced.
- The mic is disabled while the robot speaks (half-duplex), so it never transcribes
  its own voice.
- Resting face: `mouth_curve 0.32, happy 0.1` (a slight smile with a hint of raised
  lower lids). Low-confidence expressions blend toward it rather than toward a flat face.
- The latency run with speech came out faster overall than the previous run (state
  median 46 vs 73 ms, first token 205 vs 319 ms) on the same code path, which says more
  about MPS run-to-run variance than about any change. The README reports the latest run.
- `results/test_predictions.csv` and `results/asr_transcripts_test.csv` contain MELD test
  transcripts alongside the predictions. MELD is GPL-3.0, which permits redistribution;
  it is credited in the README. Raw audio and features are not committed.
