# Soft agreement experiment

The improved agent prediction might help find the step. We tested agent-marginal blending weights 0, 0.25, 0.5, 0.75 and 1, and step-agreement weights 0, 0.25, 0.5 and 1. External grouped predictions alone selected agent weight 0.5 and step weight 1, maximizing the average of Algorithm-Generated agent and step accuracy, with joint accuracy as a tie breaker.

| Evaluation | Before agent | After agent | Before exact step | After exact step | Before joint | After joint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| External Algorithm-Generated development | 781/989 | 807/989 | 681/989 | 692/989 | See selection.json | 668/989 |
| Original Algorithm-Generated | 73/126 | 70/126 | 42/126 | 42/126 | 36/126 | 40/126 |
| Original Hand-Crafted | 37/58 | 37/58 | 8/58 | 9/58 | 7/58 | 8/58 |

This improved agreement between the two outputs and joint correctness, but reduced original Algorithm-Generated agent accuracy. It is saved as an experiment and is not applied by the new prediction command. The 57.9% agent and 33.3% exact-step result remains the recommended Algorithm-Generated candidate. This is further evidence that external development improvements do not always transfer to the original logs. Original results remain exploratory, and no original labels selected these weights.

Reproduce with `python -m agentdiag_v2.contextual_consistency`. The full grid, selected settings and every original prediction are saved alongside this report.
