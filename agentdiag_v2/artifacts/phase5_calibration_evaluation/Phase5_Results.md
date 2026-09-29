# Phase 5 — Held-out calibration and attribution

- Temperature chosen on validation only: 2.645.
- **BN uncalibrated:** 5/13 correct (38.5%); Brier 0.814; ECE 0.438; log loss 1.424.
- **BN calibrated:** 5/13 correct (38.5%); Brier 0.689; ECE 0.146; log loss 1.238.
- **Position-prior baseline:** 4/13 correct (30.8%); Brier 0.741; ECE 0.144; log loss 1.340.
- Accuracy 95% trace-bootstrap interval: [0.15384615384615385, 0.6153846153846154].
- Exploratory uncalibrated test breakdown by agent count: 2 agents 3/3, 3 agents 3/7, 4 agents 5/13.
- This is a small held-out subset; literature scores are not directly comparable without a matched protocol.
