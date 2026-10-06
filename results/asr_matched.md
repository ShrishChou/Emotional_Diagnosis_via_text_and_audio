| Fused head trained on | Dev weighted F1 (selection set) | Test wF1, gold transcripts | Test wF1, Whisper transcripts | Test macro F1, Whisper |
| --- | ---: | ---: | ---: | ---: |
| gold | 0.612 ± 0.002 | 0.614 ± 0.006 | 0.521 ± 0.003 | 0.355 ± 0.003 |
| asr | 0.531 ± 0.003 | 0.596 ± 0.015 | 0.531 ± 0.005 | 0.369 ± 0.003 |
| gold+asr | 0.522 ± 0.000 | 0.608 ± 0.012 | 0.535 ± 0.006 | 0.375 ± 0.004 |

Mean ± std over 3 seeds. `gold` selects on gold dev; `asr` and `gold+asr` on Whisper dev.
