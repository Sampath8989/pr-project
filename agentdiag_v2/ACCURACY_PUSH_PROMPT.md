# Prompt: Continue the AgentDiag accuracy push

Work in the `agentdiag_v2` folder from the repository root. Continue the existing project; do not overwrite or delete prior files, reports, models, presentations, or predictions. Read this prompt and the linked project reports before changing code.

## Goal

Improve responsible-agent and decisive-step attribution for multi-agent failures. Compare results fairly with the original Who&When work and AgenTracer. Evaluate against Who&When Pro only if its official labeled evaluation data and scorer are available. Do not claim to beat a paper from incomparable metrics, a reused test set, or a small exploratory result.

## Existing project state

- Original model and pipeline: `run_pipeline.py`, `predict.py`, `artifacts/phase0_protocol/` through `phase9_evidence_triage/`.
- Current baseline: categorical Bayesian network. It scored 5/13 (38.5%) top-choice accuracy on the original four-agent test split.
- Tuned step model: `src/step_ranker.py`, trained on filtered StepFinder training traces. Its full Who&When evaluation is in `benchmark_step/Benchmark_Report.md` and its saved model is `benchmark_step/step_model.json` (L2=0.45, selected using external question-group CV). The former L2=1.0 model and predictions are preserved in `benchmark_step/previous_l2_1/`.
- Hybrid four-agent predictor: `predict_hybrid.py`; training/evaluation code is `hybrid_upgrade.py`; saved model is `hybrid_upgrade/hybrid_model.json`.
- With the tuned step model, the hybrid scored 48/91 (52.7%) versus 40/91 (44.0%) for the BN in nested question-group folds. The paired exact test gave p = 0.077 (12 hybrid-only correct, 4 BN-only correct), so the improvement is promising but inconclusive. The previous hybrid model, metrics, and predictions are preserved in `hybrid_upgrade/previous_step_model_l2_1/`.
- On the already-inspected 13-case test, the hybrid scored 6/13 versus 5/13 for a BN refit on the same train-plus-validation cases. This test result is exploratory because prior experiments already inspected the test examples.
- The tuned external step model scored 62/126 agent and 36/126 step on Algorithm-Generated traces, and 37/58 agent and 10/58 step on Hand-Crafted traces. These use strict exact matching after filtering near-duplicate training questions. Compared with archived L2=1.0, it gains two correct steps per subset but does not improve agent accuracy overall.
- AgenTracer's reported no-ground-truth scores are 63.73% agent / 37.30% step on Algorithm-Generated, and 63.82% agent / 20.68% step on Hand-Crafted. The current project does not beat these scores.
- Who&When Pro's official repository lists its full benchmark data release as pending. Do not invent a Pro score or substitute an unofficial dataset without documenting that distinction.
- `BENCHMARK_STATUS.md`, `RESEARCH_UPGRADE.md`, and `ALL_ACCURACY_RESULTS.md` contain the current conclusions and references.
- Most recent checks passed: 13 upgrade checks, 17 original artifact checks, and 6 tests.
- In the prior session, no NVIDIA GPU was available and no model API key was found in the environment. Ask the user if a GPU or model API is now available; continue with the available local resources while waiting.

## Required work

1. Recheck the current workspace state and read the reports above. Preserve unrelated or pre-existing files and changes.
2. Confirm whether GPU or model API access is available. If available, use it only through an explicit, documented model configuration; never print or save secrets. If no access is available, continue with reproducible local experiments.
3. Fix the benchmark protocol before tuning. Keep question groups together, remove exact and semantic near-duplicate train/evaluation questions, report dataset subset counts, and freeze a fresh evaluation split or use nested question-group cross-validation. The original 13-case test is no longer an untouched holdout.
4. Evaluate relevant model options with ablations: evidence-only BN; external step ranker; hybrid fusion; a stronger text/semantic model if compute permits; and an abstaining top-k mode. Choose settings only on training/inner-validation folds.
5. Compare exact agent, exact raw step, and joint accuracy on the same subset and scorer as each cited benchmark. Also report confidence intervals and a paired comparison against the current hybrid. Use strict equality unless the official benchmark scorer requires otherwise; disclose any discrepancy.
6. Make the strongest *validated* model available through a clear CLI. Retain the existing BN and hybrid as baselines. Prediction code must not read `mistake_agent` or `mistake_step` from an input trace.
7. Save model provenance, training-data hashes/filtering, fold assignments, per-trace predictions, aggregate metrics, ablations, and a plain-language report in a new accuracy-study subfolder. Update `BENCHMARK_STATUS.md`, `ALL_ACCURACY_RESULTS.md`, and README so claims match actual measured results.
8. Run the project verification scripts and tests after implementation. If the model does not beat a benchmark, say so plainly and state what resource or data is missing. Do not manufacture, tune on, or overstate results to satisfy the target.

## Primary references

- Who&When: https://arxiv.org/abs/2505.00212
- AgenTracer: https://arxiv.org/abs/2509.03312
- Who&When Pro code and data-release status: https://github.com/ag2ai/whowhen_pro
- StepFinder training source: https://github.com/taiyu-zhu/StepFinder
- StepFinder snapshot used in this project: commit `48d71c7090adca3667a31b043a78484f407b4a2f`

## Completion standard

Finish only after the experiments, outputs, saved model, reproducible commands, and checks are complete. State separately whether implementation succeeded, whether the measured accuracy improved over the current hybrid, and whether either published benchmark was actually beaten under a comparable protocol.
