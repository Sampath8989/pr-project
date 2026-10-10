# Prompt: Continue the AgentDiag accuracy push

## Latest measured version (10 October 2026)

Use the contextual model as the Algorithm-Generated starting point: `contextual_upgrade/Contextual_Report.md`, `selection.json`, `metrics.json`, `verification.json` and `benchmark_predictions.jsonl`. It scores **73/126 (57.9%) agent** and **42/126 (33.3%) exact step**, up from 62/126 and 36/126. The 60%/35% goals require 76/126 and 45/126, three more correct predictions each. AgenTracer's reported no-ground-truth scores remain 63.73%/37.30%.

The new architecture blends the previous model with a contextual lexical model and a task-aware bidirectional-GRU/attention ensemble. All settings used external question-group validation. Thirty neural fold fits were completed. The soft agreement follow-up improved external development metrics but lowered original agent accuracy to 70/126, so it was not adopted. The original corpus is inspected and exploratory; a clean paper-matched victory is not established.

Saved checkpoints: `model_sparse_0.5.json`, `model_temporal_exact_16_13.pt`, `model_temporal_exact_16_29.pt` under `contextual_upgrade/`. Command: `python -m agentdiag_v2.predict_contextual TRACE`. Keep the old `predict_step.py` for Hand-Crafted: the new combined candidate's exact step score fell from 10/58 to 8/58. The original BN, missing-evidence and HMM features remain separate, with their previous limitations.

Local runtime: `S:/pr/agentdiag-mlenv/Scripts/python.exe`; encoder/cache: `S:/pr/agentdiag-model-cache`; GPU: RTX 2050 4 GB. Embeddings are packed to avoid slow small-file reads; a redundant numeric-feature loop was fixed with identical output values. Run `python -m agentdiag_v2.verify_contextual`, both existing verifiers and the tests. Latest checks: 16 tests, 17 artifact checks, 13 prior upgrade checks, and all 184 contextual predictions reproduced.

Future research should address transfer from synthetic external training logs to original failures, using more representative independent data or a residual contextual correction model with correctly nested prior predictions. Do not tune blends on original test answers, relax overlap filters to manufacture a score, or claim novelty beyond the evidence. Only local resources are authorized. Stop work when remaining usage falls below 25%.

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
- Who&When Pro now provides an evaluation harness and dataset download. No Pro score has been measured for this project; do not invent one or mix its benchmark with the older Who&When scores.
- `BENCHMARK_STATUS.md`, `RESEARCH_UPGRADE.md`, and `ALL_ACCURACY_RESULTS.md` contain the current conclusions and references.
- Most recent checks passed: 13 upgrade checks, 17 original artifact checks, and 6 tests.
- On the Windows migration machine, an RTX 2050 with 4 GB VRAM is available. The user authorized local resources only. Do not assume that the GPU can run a large 8B tracer model.

## Required work

1. Recheck the current workspace state and read the reports above. Preserve unrelated or pre-existing files and changes.
2. Use only local resources unless the user changes that instruction. Document the actual hardware and runtime used; never print or save secrets.
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
