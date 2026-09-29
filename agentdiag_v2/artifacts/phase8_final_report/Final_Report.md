# AgentDiag v2 — Final report

This is an independent implementation of the multi-agent scope in the PPT. It parses 184 Who&When traces and uses a question-group-safe split.

## Root-cause attribution (held-out four-agent traces)

- **Bayesian network, calibrated:** 5/13 correct (38.5%); Brier 0.689; ECE 0.146; log loss 1.238.
- **Position-prior baseline:** 4/13 correct (30.8%); Brier 0.741; ECE 0.144; log loss 1.340.
- Selected features: ('error', 'uncertain', 'correction'); temperature: 2.645.
- Accuracy 95% bootstrap interval: [0.15384615384615385, 0.6153846153846154].
- Categorical posteriors are checked to sum to one. VE is checked against joint enumeration.

- Evidence-backed two-agent shortlist contains the true agent in 10/13 held-out traces. This is review coverage, not automatic accuracy.

- Exploratory uncalibrated test breakdown: 2 agents 3/3, 3 agents 3/7, 4 agents 5/13. These smaller groups are not the primary calibrated result.

## Missing logs

- **complete:** 5/13 correct (38.5%); Brier 0.689; ECE 0.146; log loss 1.238.
- **middle_A3_missing:** 5/13 correct (38.5%); Brier 0.699; ECE 0.097; log loss 1.262.
- **one_random_missing:** 6/13 correct (46.2%); Brier 0.711; ECE 0.177; log loss 1.295.
- **two_random_missing:** 4/13 correct (30.8%); Brier 0.729; ECE 0.099; log loss 1.333.

## Early warning

- Correct culprit warned before the mistake: 2/10 eligible held-out traces.
- Wrong-agent warning traces: 11/12.

- Keyword baseline: 1/10 correct early, 6/12 wrong-agent warning traces.

## Scope and claims

- This is multi-agent failure analysis. A separate multimodal performance claim is not supported by the current evaluation.
- Comparison to AgenTracer is not asserted: a matched evaluation protocol/predictions were not available locally.
- The source benchmark contains failed runs, so final-failure prediction from successful/failed examples is not evaluated.
- The PPT should use these held-out results and explicitly identify any goal not achieved.
