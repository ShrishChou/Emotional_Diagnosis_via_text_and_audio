| Configuration | Input | Test weighted F1 | Test macro F1 | Test accuracy | Dev weighted F1 |
| --- | --- | --- | --- | --- | --- |
| text | utterance (RoBERTa) | 0.609 ± 0.002 | 0.436 ± 0.007 | 0.616 ± 0.001 | 0.593 ± 0.003 |
| text_ctx | context + utterance (RoBERTa) | 0.613 ± 0.005 | 0.433 ± 0.005 | 0.619 ± 0.005 | 0.597 ± 0.004 |
| audio | WavLM, layer-weighted | 0.469 ± 0.002 | 0.291 ± 0.002 | 0.482 ± 0.007 | 0.475 ± 0.006 |
| fused | text_ctx + audio, concat MLP (main model) | 0.614 ± 0.006 | 0.441 ± 0.008 | 0.619 ± 0.007 | 0.612 ± 0.002 |
| fused_utt | text + audio, concat MLP (no context) | 0.621 ± 0.003 | 0.450 ± 0.004 | 0.619 ± 0.003 | 0.600 ± 0.002 |
| late_fusion | mean of text_ctx and audio probabilities | 0.618 ± 0.005 | 0.436 ± 0.008 | 0.635 ± 0.003 | 0.595 ± 0.003 |
| logreg (sanity) | utterance (RoBERTa), sklearn | 0.564 | 0.390 | 0.573 | – |

Mean ± std over seeds [0, 1, 2]. MELD official test split, 2610 utterances.

Per-class test F1 (mean over seeds):

| Configuration | anger | disgust | fear | joy | neutral | sadness | surprise |
| --- | --- | --- | --- | --- | --- | --- | --- |
| text | 0.443 | 0.232 | 0.199 | 0.555 | 0.776 | 0.308 | 0.537 |
| text_ctx | 0.460 | 0.178 | 0.192 | 0.551 | 0.779 | 0.326 | 0.545 |
| audio | 0.381 | 0.066 | 0.043 | 0.304 | 0.649 | 0.250 | 0.343 |
| fused | 0.472 | 0.181 | 0.213 | 0.559 | 0.772 | 0.364 | 0.525 |
| fused_utt | 0.463 | 0.215 | 0.204 | 0.578 | 0.776 | 0.366 | 0.546 |
| late_fusion | 0.493 | 0.130 | 0.203 | 0.540 | 0.780 | 0.353 | 0.550 |
