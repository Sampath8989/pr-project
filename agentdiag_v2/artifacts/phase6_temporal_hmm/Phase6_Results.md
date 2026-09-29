# Phase 6 — Temporal HMM and early warning

- HMM fitted on train traces with annotation-derived proxy states; threshold 0.25 chosen on validation.
- Ambiguous mistake-step labels excluded: train 16, four-agent validation 0, four-agent test 1.
- Held-out four-agent traces: 12; eligible for pre-mistake warning: 10.
- Correct-culprit early warnings: 2/10 eligible.
- Traces with a warning for another agent: 11/12.
- Keyword baseline correct early: 1/10; wrong-agent warning traces: 6/12.
- Forward versus brute-force max difference: 0.
- A fitted HMM does not establish early-warning usefulness unless these held-out results support it.
