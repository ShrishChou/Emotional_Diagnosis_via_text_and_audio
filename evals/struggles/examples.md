# Where audio changed the prediction (MELD test)

Out of 2,610 test utterances, adding the voice turned a wrong text-only answer into a right one 165 times and a right one into a wrong one 147 times (the deployed models). Rows below are picked mechanically: the largest change in the probability of the true emotion.

## Audio helped (fused right, text-only wrong)

| Clip | Transcript | True | Text-only | Fused |
| --- | --- | --- | --- | --- |
| dia2_utt1 | Ross, didn't you say that there was an elevator in here? | neutral | surprise (0.65) | neutral (0.82) |
| dia232_utt1 | Uhh, yeah. She uh, she uh, she uh might've mentioned him. | neutral | fear (0.48) | neutral (0.83) |
| dia113_utt10 | It's throwing and catching! | anger | joy (0.61) | anger (0.73) |

## Audio hurt (text-only right, fused wrong)

| Clip | Transcript | True | Text-only | Fused |
| --- | --- | --- | --- | --- |
| dia170_utt0 | Oh my God, you're back! | surprise | surprise (0.82) | joy (0.67) |
| dia84_utt1 | Nothing! | anger | anger (0.72) | surprise (0.34) |
| dia271_utt3 | I'm trppd... in an ATM vstbl... wth | fear | fear (0.50) | sadness (0.47) |
