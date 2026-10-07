| Idle gap (s) | Condition | State (ms) | Audio encoder (ms) | First token (ms) |
| ---: | --- | ---: | ---: | ---: |
| 0 | cold | 52 | 39 | 411 |
| 0 | ping | 49 | 37 | 371 |
| 2 | cold | 134 | 69 | 349 |
| 2 | ping | 57 | 35 | 344 |
| 5 | cold | 147 | 92 | 359 |
| 5 | ping | 76 | 43 | 353 |
| 10 | cold | 150 | 91 | 357 |
| 10 | ping | 70 | 44 | 366 |
| 20 | cold | 312 | 228 | 399 |
| 20 | ping | 132 | 69 | 378 |

Median of 3 turns per cell, gold transcripts, mps. `ping`: Pipeline.ping() 1 s before the turn (at gap 0 there is no ping).
