# Released Who&When scorer audit

The released `Automated_FA/evaluate.py` counts an agent as correct when the actual agent name is a substring of the predicted string and counts a step when the actual step number is a substring of the predicted string. We applied those two comparisons to our saved step-model outputs.

| Subset | Strict agent | Released-style agent | Strict raw step | Released-style step |
| --- | ---: | ---: | ---: | ---: |
| Algorithm-Generated | 62/126 | 62/126 | 36/126 | 36/126 |
| Hand-Crafted | 37/58 | 37/58 | 10/58 | 12/58 |

This reproduces only the released scorer's final string comparisons on our saved predictions. It does not reproduce the paper's prompt, model run, all filtering decisions, or statistical protocol. Strict exact matching remains the project's main reported metric.
