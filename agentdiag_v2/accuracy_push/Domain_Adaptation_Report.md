# Frozen-split domain adaptation experiment

External training: 2004 filtered traces. Original training: 118 label-consistent traces. Validation: 27. Test: 23.
Validation selected original-training multiplier **0** from (0, 4, 12). Selection maximized exact-step accuracy, with agent accuracy as a tie breaker.

| Method | Validation agent | Validation step | Test agent | Test step | Test joint |
| --- | ---: | ---: | ---: | ---: | ---: |
| Multiplier 0 | 14/27 | 5/27 | 11/23 | 7/23 | 6/23 |
| Multiplier 4 | 14/27 | 4/27 | not tested | not tested | not tested |
| Multiplier 12 | 14/27 | 5/27 | not tested | not tested | not tested |

The original test cases were inspected during earlier work. This result is exploratory and cannot establish a fair win over published systems.
