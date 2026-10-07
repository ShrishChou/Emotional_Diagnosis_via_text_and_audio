| Config | LLM params | Pipeline total | ≤ 2 sentences | Ends cleanly | Emotion word | Announces feeling | First token p50 / p95 (ms) | Total p50 (ms) | Meets bar |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qwen2.5-1.5b_baseline | 1.54B | 2.00B | 76% | 100% | 4% | 2% | 182 / 234 | 736 | no |
| qwen2.5-1.5b_behavior | 1.54B | 2.00B | 100% | 100% | 0% | 0% | 308 / 382 | 770 | yes |
| qwen2.5-3b_behavior | 3.09B | 3.55B | 100% | 100% | 0% | 0% | 598 / 702 | 1490 | yes |
| qwen3-4b-2507_behavior | 4.02B | 4.48B | 100% | 96% | 0% | 6% | 838 / 981 | 2133 | yes |

50 MELD test clips, gold transcripts, emotion-conditioned replies, device mps. Bar (fixed before running): within_2_sentences >= 0.95, ends_cleanly >= 0.9, emotion_word <= 0.05, announces_feeling <= 0.1, first_token_p95_ms < 1000.0. In the behavior style the sentence limit and the emotion-word list are enforced in decoding, so those two columns are compliant by construction there; 'announces feeling' is not enforced.
