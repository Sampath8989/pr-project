# Benchmark status — 29 September 2026

## Current measured results

- The tuned external StepFinder ranker (L2=0.45, selected by five-fold question-group validation on external training data) scores **62/126 (49.2%) agent** and **36/126 (28.6%) exact raw step** on Algorithm-Generated traces; **37/58 (63.8%) agent** and **10/58 (17.2%) exact raw step** on Hand-Crafted traces. The joint correct counts are 33/126 and 8/58.
- Against the archived L2=1.0 model, agent accuracy is 63/126 and 36/58; step accuracy was 34/126 and 9/58. The new setting gains two exact steps on each subset, while agent accuracy changes by -1 on Algorithm-Generated and +1 on Hand-Crafted. This is a small improvement, not a broad accuracy breakthrough.
- The tuned hybrid still scores **48/91 (52.7%)** in nested question-group evaluation, versus **40/91 (44.0%)** for the Bayesian network. Its paired exact test is p=0.077 (12 hybrid-only correct, 4 BN-only correct), so this small-sample improvement is not conclusive. The already-inspected 13-case test gives 6/13 hybrid versus 5/13 BN and is exploratory.

## Published benchmark comparison

- The older [Who&When paper](https://arxiv.org/abs/2505.00212) reports GPT-4o all-at-once without ground-truth trajectory access at **51.12% agent / 13.53% step** on Algorithm-Generated and **53.44% / 3.51%** on Hand-Crafted. Our tuned model is close on the first agent score (49.2%, 1.9 percentage points lower), higher on Hand-Crafted agent accuracy (63.8%, 10.4 points higher), and above both reported step scores. These are not fully matched protocols: our scoring uses strict exact agent names and raw step indices, while the paper's released scorer uses permissive substring comparisons.
- [AgenTracer](https://arxiv.org/abs/2509.03312) reports no-ground-truth results of **63.73% agent / 37.30% step** on Algorithm-Generated and **63.82% / 20.68%** on Hand-Crafted. Our scores are below those results on all four measures (49.2% / 28.6%; 63.8% / 17.2%).
- [Who&When Pro](https://github.com/ag2ai/whowhen_pro) has no verified score here because its official full benchmark release and matching evaluation data are unavailable in this workspace.

## Interpretation and reproducibility

Implementation and local verification succeeded. Accuracy improved modestly on exact step attribution relative to the archived prior setting, and the hybrid improves on the BN in grouped folds. The project has **not** beaten AgenTracer. It is close to the older Who&When reported scores under only a partly comparable protocol; this is not evidence of a clean benchmark win. The Who&When corpus has been inspected during development, so its scores are exploratory rather than an untouched holdout.

Re-run the external experiment with `python3 agentdiag_v2/benchmark_step.py --stepfinder-data /path/to/StepFinder/data`, then `python3 -m agentdiag_v2.hybrid_upgrade`, `python3 -m agentdiag_v2.verify_upgrade`, `python3 agentdiag_v2/verify.py`, and `python3 -m pytest agentdiag_v2/tests -q`. The previous L2=1.0 model and predictions are preserved in `benchmark_step/previous_l2_1/`; the previous hybrid snapshot is in `hybrid_upgrade/previous_step_model_l2_1/`.

Independent proof of benchmark leadership requires a frozen method, official benchmark data and scorer, and ideally additional local compute or authorized model access. No GPU or model API was available for this run.
