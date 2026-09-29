# Current PPT claims versus verified AgentDiag v2 results

- **Bayesian network and data-driven CPTs:** Implemented using training-only counts.
- **Hand-implemented Variable Elimination:** Implemented; compared with exact joint enumeration.
- **Probability distribution over root causes:** Implemented as one categorical culprit variable; every posterior sums to one.
- **Missing observations:** Implemented by marginalizing absent evidence; evaluated on held-out masks.
- **Calibration metrics and reliability diagram:** Implemented on held-out data after validation-only temperature fitting. Test ECE is 0.146 versus 0.144 for the positional baseline, so superior calibration is **not established**.
- **HMM reliability tracking:** Implemented and tested. Correct early warnings are 2/10 eligible traces, with wrong-agent warnings on 11/12 traces; early identification is **not reliable**.
- **Competitive performance versus AgenTracer:** **Not established**; no matched runnable evaluation is available.
- **Multimodal agents:** Outside the current PPT's multi-agent scope and without a separate modality-specific evaluation.
