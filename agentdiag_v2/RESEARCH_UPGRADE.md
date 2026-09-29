# Research and accuracy audit (29 September 2026)

## What the supplied papers suggest

The supplied PDF collection was screened for its problem, method, and evaluation. The closest papers received a focused method comparison:

- [Who&When](https://arxiv.org/abs/2505.00212) supplies the agent and step attribution benchmark. Its published numbers use its own model and protocol, so they are not a score for this code.
- [AgenTracer](https://arxiv.org/abs/2509.03312) learns a failure tracer using replay and reinforcement learning. This project has no matched replay environment or runnable matched baseline.
- [StepFinder](https://arxiv.org/abs/2606.03467) combines semantic and temporal evidence for step localization. Its public augmented training traces were considered here, with near-duplicate held-out questions removed.
- [CAGE-CAL](https://arxiv.org/abs/2605.30653) calibrates graph-based multi-agent consensus. It concerns reliability/calibration, not the same categorical culprit task.
- The supplied *Ensemble Bayesian Network root cause analysis of product defects* studies BN ensembles in manufacturing. It prevents a broad claim that using a Bayesian network for root-cause analysis is itself new.
- The supplied *Graph-based Confidence Calibration* paper likewise prevents claiming that graph confidence calibration alone is new.

The defensible **project contribution** is an auditable failure-review output that combines a categorical Bayesian posterior, explicit missing-log marginalization, a two-agent shortlist, exact leave-one-agent-evidence-out model sensitivity, and message-indexed evidence excerpts. The sensitivity is an ablation of a predictive model, **not** a causal counterfactual. We have not established that this combination is publication-level novelty or state-of-the-art performance.

## Accuracy experiments

All model choices below were exploratory. The primary frozen split contains 65 four-agent training traces, 13 validation traces, and 13 test traces. The small test set has now been inspected during research, so it must not be treated as a pristine future holdout for further tuning.

- Existing calibrated Bayesian network: **5/13 top-one** on the four-agent test; Brier **0.689**. Its position-prior baseline is **4/13**.
- Hashed lexical plus observed-handoff ranker trained on the 65 base training traces: selected using validation, then **6/13 top-one** on test, but worse Brier (**0.721**) and ECE (**0.320**). It is not a general quality improvement.
- Same ranker trained additionally on 1,316 filtered public StepFinder generated traces: **3/13 top-one** on test for the selected setting. The generated data did not transfer reliably.
- Local BGE-small semantic embeddings trained with the same filtered augmented data: best validation configuration **9/13**, but only **3/13** on test (Brier **0.994**). This is a clear example of validation overfitting; it was not put into the prediction path.
- Removing task-instruction messages before keyword extraction: still **5/13 top-one** on test, with worse Brier (**0.722**). The review packets instead identify excerpts from task instructions so readers can judge this weak signal.
- A ranker trained on all original training traces (including two- and three-agent examples) reached **7/13** on the already-inspected four-agent test set. However, [five question-group folds](accuracy_study/Accuracy_Study.md) covering all 91 four-agent traces gave it **33/91**, below the BN's **40/91**. It is not deployed as an accuracy improvement.
- Frozen BN two-agent shortlist: the true culprit appears in the top two for **10/13** test cases (**76.9%**). This is a review metric, not automatic top-one accuracy.

The augmented data was fetched from [StepFinder's public repository](https://github.com/taiyu-zhu/StepFinder), kept outside this project for the experiments, and filtered by question-token Jaccard similarity of at least 0.85 to any validation or test question. This is a partial leakage guard, not a guarantee against semantic duplicates. No augmented traces or embedding model are required for the delivered prediction command.

## What remains unproved

- High top-one attribution accuracy and a matched comparison with AgenTracer or StepFinder.
- A robust improvement across new question groups. This needs a fresh untouched holdout or external benchmark after model choices are frozen.
- Reliable early warning: the existing HMM generated many wrong-agent warnings.
- Multimodal performance: the supplied labeled data does not support a separate image/audio/video evaluation.

The [Phase 9 result](artifacts/phase9_evidence_triage/Phase9_Results.md) and [per-trace packets](artifacts/phase9_evidence_triage/test_review_packets.jsonl) are the implemented, checked output improvement. They deliberately preserve uncertainty and trace provenance.

## Subsequent benchmark upgrade

A separate [step ranker](benchmark_step/Benchmark_Report.md) was trained on 2,004 StepFinder traces after semantic question-overlap filtering. It identifies both a message and an agent, but its exact full-benchmark results remain below the newer AgenTracer agent/step results. A [four-agent hybrid](hybrid_upgrade/Hybrid_Report.md) blends its agent scores with the BN and improves nested grouped accuracy from 40/91 to 48/91. The paired p-value is 0.096 and the old 13-case test was already inspected, so this is a measured project improvement, not proof of benchmark leadership.
