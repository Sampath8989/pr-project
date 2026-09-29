# Phase 2 — Evidence and baselines

- Trace-level features: error, uncertain, correction. Counts of positive agent features: {'error': 190, 'uncertain': 61, 'correction': 296}.
- Four-agent validation n=13; test n=13.
- **position_prior (test):** 4/13 correct (30.8%); Brier 0.741; ECE 0.144; log loss 1.340.
- **first_error_keyword (test):** 3/13 correct (23.1%); Brier 1.183; ECE 0.535; log loss 2.574.
- **regularized_logistic (test):** 4/13 correct (30.8%); Brier 0.715; ECE 0.284; log loss 1.251.
