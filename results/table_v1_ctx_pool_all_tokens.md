| Configuration | Input | Test weighted F1 | Test macro F1 | Test accuracy | Dev weighted F1 |
| --- | --- | --- | --- | --- | --- |
| text | utterance (RoBERTa) | 0.609 ± 0.002 | 0.436 ± 0.007 | 0.616 ± 0.001 | 0.593 ± 0.003 |
| text_ctx | context + utterance (RoBERTa) | 0.516 ± 0.007 | 0.342 ± 0.002 | 0.518 ± 0.011 | 0.496 ± 0.009 |
| audio | WavLM, layer-weighted | 0.469 ± 0.002 | 0.291 ± 0.002 | 0.482 ± 0.007 | 0.475 ± 0.006 |
| fused | text + audio, concat MLP | 0.621 ± 0.003 | 0.450 ± 0.004 | 0.619 ± 0.003 | 0.600 ± 0.002 |
| late_fusion | mean of text and audio probabilities | 0.615 ± 0.003 | 0.438 ± 0.006 | 0.632 ± 0.003 | 0.596 ± 0.004 |
| logreg (sanity) | utterance (RoBERTa), sklearn | 0.564 | 0.390 | 0.573 | – |

Mean ± std over seeds [0, 1, 2]. MELD official test split, 2610 utterances.

Per-class test F1 (mean over seeds):

| Configuration | anger | disgust | fear | joy | neutral | sadness | surprise |
| --- | --- | --- | --- | --- | --- | --- | --- |
| text | 0.443 | 0.232 | 0.199 | 0.555 | 0.776 | 0.308 | 0.537 |
| text_ctx | 0.417 | 0.105 | 0.142 | 0.438 | 0.683 | 0.241 | 0.368 |
| audio | 0.381 | 0.066 | 0.043 | 0.304 | 0.649 | 0.250 | 0.343 |
| fused | 0.463 | 0.215 | 0.204 | 0.578 | 0.776 | 0.366 | 0.546 |
| late_fusion | 0.475 | 0.193 | 0.196 | 0.550 | 0.778 | 0.331 | 0.540 |
