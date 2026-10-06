# Error breakdown: deployed models on MELD test

2,610 test utterances. Saved checkpoints: text-only (`text_ctx`, seed 1) and fused (`text_ctx` + audio, seed 2, T = 1.27). "ASR" = the same fused model fed Whisper-small transcripts instead of gold. Per-utterance rows: `results/test_predictions.csv`.

| Model | Accuracy | Weighted F1 | Macro F1 |
| --- | ---: | ---: | ---: |
| text-only, gold transcript | 0.622 | 0.617 | 0.431 |
| fused, gold transcript | 0.629 | 0.622 | 0.449 |
| text-only, Whisper transcript | 0.540 | 0.514 | 0.352 |
| fused, Whisper transcript | 0.552 | 0.525 | 0.358 |

## Per class

Recall is the accuracy on utterances of that gold class. "→ neutral" is the share of that class the fused model calls neutral.

| gold class | support | text recall | fused recall | fused precision | fused F1 | fused recall (ASR) | → neutral |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| anger | 345 | 0.507 | 0.496 | 0.494 | 0.495 | 0.322 | 20% |
| disgust | 68 | 0.103 | 0.132 | 0.188 | 0.155 | 0.118 | 37% |
| fear | 50 | 0.280 | 0.240 | 0.273 | 0.255 | 0.140 | 24% |
| joy | 402 | 0.542 | 0.525 | 0.593 | 0.557 | 0.281 | 24% |
| neutral | 1,256 | 0.788 | 0.803 | 0.759 | 0.780 | 0.823 | – |
| sadness | 208 | 0.231 | 0.322 | 0.435 | 0.370 | 0.298 | 38% |
| surprise | 281 | 0.612 | 0.580 | 0.491 | 0.532 | 0.381 | 14% |

## Most common errors (fused, gold transcripts)

| Gold | Predicted | Count | Share of all errors |
| --- | --- | ---: | ---: |
| joy | neutral | 97 | 10% |
| sadness | neutral | 79 | 8% |
| anger | neutral | 69 | 7% |
| neutral | joy | 68 | 7% |
| neutral | anger | 57 | 6% |
| neutral | surprise | 56 | 6% |
| joy | anger | 47 | 5% |
| anger | surprise | 44 | 5% |
| neutral | sadness | 43 | 4% |
| surprise | neutral | 39 | 4% |

## Accuracy by slice

`text_acc` / `fused_acc`: accuracy with gold transcripts; `fused_asr_acc`: fused with Whisper transcripts; `fused_wF1`: weighted F1 of the fused model within the slice.

| Fused confidence (temperature-scaled) | n | share | text_acc | fused_acc | fused_asr_acc | fused_wF1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| <0.4 | 476 | 18% | 0.342 | 0.342 | 0.370 | 0.357 |
| 0.4-0.5 | 436 | 17% | 0.459 | 0.477 | 0.374 | 0.466 |
| 0.5-0.6 | 394 | 15% | 0.579 | 0.591 | 0.503 | 0.576 |
| 0.6-0.7 | 357 | 14% | 0.678 | 0.692 | 0.583 | 0.679 |
| 0.7-0.8 | 336 | 13% | 0.732 | 0.726 | 0.607 | 0.709 |
| 0.8-0.9 | 327 | 13% | 0.850 | 0.856 | 0.761 | 0.848 |
| 0.9-1.0 | 284 | 11% | 0.940 | 0.940 | 0.859 | 0.933 |

| Utterance length (words) | n | share | text_acc | fused_acc | fused_asr_acc | fused_wF1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-2 | 493 | 19% | 0.708 | 0.724 | 0.527 | 0.722 |
| 3-5 | 610 | 23% | 0.675 | 0.664 | 0.574 | 0.654 |
| 6-10 | 708 | 27% | 0.607 | 0.614 | 0.578 | 0.610 |
| 11-20 | 648 | 25% | 0.556 | 0.563 | 0.545 | 0.555 |
| 21+ | 151 | 6% | 0.483 | 0.530 | 0.464 | 0.508 |

| Clip duration (s) | n | share | text_acc | fused_acc | fused_asr_acc | fused_wF1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| <1 | 223 | 9% | 0.682 | 0.664 | 0.502 | 0.663 |
| 1-2 | 684 | 26% | 0.706 | 0.724 | 0.601 | 0.711 |
| 2-4 | 1,010 | 39% | 0.600 | 0.610 | 0.543 | 0.603 |
| 4-8 | 575 | 22% | 0.560 | 0.557 | 0.544 | 0.549 |
| 8+ | 118 | 5% | 0.517 | 0.534 | 0.492 | 0.548 |

| Dialogue context | n | share | text_acc | fused_acc | fused_asr_acc | fused_wF1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| first line of dialogue | 280 | 11% | 0.664 | 0.675 | 0.575 | 0.660 |
| has previous lines | 2,330 | 89% | 0.617 | 0.624 | 0.550 | 0.618 |

| Speaker | n | share | text_acc | fused_acc | fused_asr_acc | fused_wF1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Chandler | 379 | 15% | 0.599 | 0.586 | 0.528 | 0.576 |
| Joey | 411 | 16% | 0.625 | 0.657 | 0.574 | 0.656 |
| Monica | 346 | 13% | 0.575 | 0.584 | 0.506 | 0.577 |
| Phoebe | 291 | 11% | 0.612 | 0.612 | 0.502 | 0.602 |
| Rachel | 356 | 14% | 0.632 | 0.618 | 0.517 | 0.611 |
| Ross | 373 | 14% | 0.654 | 0.651 | 0.568 | 0.639 |
| other | 454 | 17% | 0.648 | 0.676 | 0.637 | 0.673 |

| Whisper word error rate on the clip | n | share | text_acc | fused_acc | fused_asr_acc | fused_wF1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (exact) | 816 | 31% | 0.653 | 0.665 | 0.594 | 0.661 |
| 0-25% | 709 | 27% | 0.577 | 0.598 | 0.581 | 0.591 |
| 25-50% | 426 | 16% | 0.592 | 0.580 | 0.519 | 0.566 |
| >50% | 659 | 25% | 0.653 | 0.649 | 0.492 | 0.642 |

| Did audio change the text-only prediction? | n | share | text_acc | fused_acc | fused_asr_acc | fused_wF1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| audio changed prediction | 452 | 17% | 0.325 | 0.365 | 0.372 | 0.372 |
| same as text-only | 2,158 | 83% | 0.684 | 0.684 | 0.590 | 0.673 |
