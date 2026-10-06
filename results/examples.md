# Where audio changed the prediction (MELD test)

Out of 2610 test utterances, audio flipped a wrong text-only prediction to correct 165 times and a correct one to wrong 147 times (saved text-only and fused checkpoints).

## Audio helped (fused correct, text-only wrong)

| Clip | Transcript | Gold | Text-only | Fused |
| --- | --- | --- | --- | --- |
| dia2_utt1 | Ross, didn't you say that there was an elevator in here? | neutral | surprise (0.65) | neutral (0.82) |
| dia232_utt1 | Uhh, yeah. She uh, she uh, she uh might've mentioned him. | neutral | fear (0.48) | neutral (0.83) |
| dia113_utt10 | It's throwing and catching! | anger | joy (0.61) | anger (0.73) |

## Audio hurt (text-only correct, fused wrong)

| Clip | Transcript | Gold | Text-only | Fused |
| --- | --- | --- | --- | --- |
| dia170_utt0 | Oh my God, you're back! | surprise | surprise (0.82) | joy (0.67) |
| dia84_utt1 | Nothing! | anger | anger (0.72) | surprise (0.34) |
| dia271_utt3 | I'm trppd... in an ATM vstbl... wth | fear | fear (0.51) | sadness (0.47) |
