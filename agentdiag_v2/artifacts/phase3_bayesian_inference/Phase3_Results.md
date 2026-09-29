# Phase 3 — Bayesian network and custom VE

- Selected evidence features on validation: ('error', 'uncertain', 'correction').
- Categorical culprit model trained on 65 four-agent traces.
- Validation: 6/13 correct (46.2%); Brier 0.711.
- VE versus independent joint enumeration: maximum difference 0.
- All posterior vectors sum to 1. Small agent-count groups use prior-only fallback.
