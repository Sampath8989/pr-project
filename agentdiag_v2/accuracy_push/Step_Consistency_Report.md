# Agent-aware step selection experiment

The weakest reported area is exact step attribution. This experiment asks whether a step is easier to locate when its author's agent probability is included in the score.
The setting was chosen on saved StepFinder question-group validation predictions. No Who&When labels selected it.

Selected agent weight: **0**. Exact paired test on the 184 inspected traces: 0 gains, 0 losses, p=1.000.

| Subset | Agent (unchanged) | Original exact step | Candidate exact step | Original joint | Candidate joint |
| --- | ---: | ---: | ---: | ---: | ---: |
| Algorithm-Generated | 62/126 | 36/126 | 36/126 | 33/126 | 33/126 |
| Hand-Crafted | 37/58 | 10/58 | 10/58 | 8/58 | 8/58 |

The saved benchmark was inspected before this experiment, so this table does not establish an independent paper-matched win.

## Error categories

- Algorithm-Generated: 61 both wrong; 29 agent right/step wrong; 3 agent wrong/step right; 33 both right. 3 labels name a different speaker at the labeled step.
  Step errors by trace length: {'6-10': 89, '1-5': 1}. Examples where the agent was correct but the step was not: Algorithm-Generated/101.json, Algorithm-Generated/11.json, Algorithm-Generated/111.json.
- Hand-Crafted: 19 both wrong; 29 agent right/step wrong; 2 agent wrong/step right; 8 both right. 16 labels name a different speaker at the labeled step.
  Step errors by trace length: {'11+': 33, '6-10': 11, '1-5': 4}. Examples where the agent was correct but the step was not: Hand-Crafted/1.json, Hand-Crafted/11.json, Hand-Crafted/13.json.
