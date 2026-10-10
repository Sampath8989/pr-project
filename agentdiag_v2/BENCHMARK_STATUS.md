# Benchmark status — 10 October 2026

## Latest contextual experiment

| Algorithm-Generated method | Exact agent | Exact step |
| --- | ---: | ---: |
| Older PR ranker, L2=1.0 | 63/126 (50.0%) | 34/126 (27.0%) |
| Previous tuned PR ranker | 62/126 (49.2%) | 36/126 (28.6%) |
| New contextual ensemble | **73/126 (57.9%)** | **42/126 (33.3%)** |
| Who&When GPT-4o all-at-once, no ground truth, reported | 51.12% | 13.53% |
| AgenTracer, no ground truth, reported | 63.73% | 37.30% |

The new architecture adds task-aware semantic and temporal context, word pairs, broader message context and a directly supervised agent head. External validation froze the chosen model and blend before original scoring. It gains 16 and loses 5 agent matches, and gains 11 and loses 5 step matches, relative to the previous tuned model. Exact paired p-values are 0.0266 for agent and 0.2101 for step; these are exploratory. Both outputs are correct on 36/126 cases versus 33/126 previously.

The targets remain unmet: 60% agent needs 76/126 and 35% step needs 45/126, three additional correct predictions each. The reported AgenTracer gaps are 5.8 and 4.0 percentage points. The new Hand-Crafted agent score stays 37/58, but steps fall to 8/58; keep the old predictor for that subset. The follow-up soft agreement experiment achieved 70/126 agents and 42/126 steps and was not adopted because agent accuracy regressed.

All 184 saved checkpoint predictions were reproduced with label-free parsing and their strict counts recomputed. Read [Contextual_Report.md](contextual_upgrade/Contextual_Report.md), [selection](contextual_upgrade/selection.json), [predictions](contextual_upgrade/benchmark_predictions.jsonl) and [verification](contextual_upgrade/verification.json). These exceed the older Who&When reported percentages numerically; the inspected corpus and protocol differences prevent a clean benchmark-victory claim. The historical results below are preserved.

## Current measured results

- The tuned external StepFinder ranker (L2=0.45, selected by five-fold question-group validation on external training data) scores **62/126 (49.2%) agent** and **36/126 (28.6%) exact raw step** on Algorithm-Generated traces; **37/58 (63.8%) agent** and **10/58 (17.2%) exact raw step** on Hand-Crafted traces. The joint correct counts are 33/126 and 8/58.
- Against the archived L2=1.0 model, agent accuracy is 63/126 and 36/58; step accuracy was 34/126 and 9/58. The new setting gains two exact steps on each subset, while agent accuracy changes by -1 on Algorithm-Generated and +1 on Hand-Crafted. This is a small improvement, not a broad accuracy breakthrough.
- The tuned hybrid still scores **48/91 (52.7%)** in nested question-group evaluation, versus **40/91 (44.0%)** for the Bayesian network. Its paired exact test is p=0.077 (12 hybrid-only correct, 4 BN-only correct), so this small-sample improvement is not conclusive. The already-inspected 13-case test gives 6/13 hybrid versus 5/13 BN and is exploratory.

## Windows migration and new accuracy attempts

- The Windows checkout at `S:/pr/pr-project` contains the complete GitHub `agentdiag_v2` code. All 184 original Who&When records are present and parse to the same data as the local upstream copy. The saved phase manifest contained Linux absolute paths; it now uses portable relative paths. The pipeline, 17 artifact checks, upgrade checks, and tests are rerun locally.
- [Agent-aware step selection](accuracy_push/Step_Consistency_Report.md) was selected on external grouped validation. Validation chose weight **0**, so the original step model remains unchanged. The original-corpus scores remain 36/126 and 10/58 exact steps.
- [Frozen-split domain adaptation](accuracy_push/Domain_Adaptation_Report.md) tried zero, four, and twelve copies of 118 label-consistent original training traces alongside the 2,004 external traces. Validation selected **zero**. The 23-case frozen test therefore stayed at 11/23 agent and 7/23 exact step.
- [Subset-specific training](accuracy_push/Subset_Specialist_Report.md) lowered external grouped validation scores on both subsets, so the combined model remained selected. No new model from this round was deployed as an accuracy improvement.
- The [error audit](accuracy_push/Step_Consistency_Report.md) counts 29 agent-correct/step-wrong cases in each original subset. In 19 of 184 records, the annotated step belongs to a different speaker than the annotated culprit; that needs a labeling-convention review before enforcing agent-step consistency.
- The weakest **internal feature** remains early warning: the HMM warned about the correct culprit early on only 2/10 eligible held-out traces and warned about another agent on 11/12 traces. Its validation threshold sweep shows a strong recall/false-alarm tradeoff; a higher threshold stops almost all warnings but also loses the observed early successes. Do not present this as a useful early-warning system yet.

## Published benchmark comparison

- The older [Who&When paper](https://arxiv.org/html/2505.00212) reports GPT-4o all-at-once without ground-truth trajectory access. [AgenTracer](https://arxiv.org/html/2509.03312) reports its no-ground-truth results on the same named subsets. The table puts the numbers together, but **it is not a matched head-to-head test**.

| Method and scoring | Algorithm-Generated agent | Algorithm-Generated step | Hand-Crafted agent | Hand-Crafted step |
| --- | ---: | ---: | ---: | ---: |
| Our external step ranker, strict exact match | 62/126 (49.2%) | 36/126 (28.6%) | 37/58 (63.8%) | 10/58 (17.2%) |
| Our saved outputs through released Who&When string comparison | 62/126 (49.2%) | 36/126 (28.6%) | 37/58 (63.8%) | 12/58 (20.7%) |
| Who&When GPT-4o all-at-once, no ground truth (paper) | 51.12% | 13.53% | 53.44% | 3.51% |
| AgenTracer, no ground truth (paper) | 63.73% | 37.30% | 63.82% | 20.68% |

The [released scorer audit](accuracy_push/Paper_Scorer_Audit.md) found two extra Hand-Crafted step matches caused by substring comparisons of numbers. That gives a numerical 20.7%, but it is **not** evidence that our step model matches AgenTracer: its implementation and full evaluation protocol are different. The strict result of 17.2% is the project's primary score. Our method still trails AgenTracer on Algorithm-Generated agent and step accuracy and Hand-Crafted agent accuracy.
- [Who&When Pro](https://github.com/ag2ai/whowhen_pro) now provides an evaluation harness and a dataset download. This project has **no Pro evaluation**: the newer benchmark has different tasks, modalities, and scoring, and its data was not used here.

## Interpretation and reproducibility

The Windows migration and local checks succeeded. The new accuracy attempts did **not** improve the selected model. The older tuned step model improved modestly relative to its archived L2=1.0 setting, and the hybrid remains better than the BN in exploratory grouped folds. The project has **not** beaten AgenTracer. It is close to some older Who&When reported numbers under only a partly comparable protocol; this is not evidence of a clean benchmark win. The Who&When corpus has been inspected during development, so its scores are exploratory rather than an untouched holdout.

Re-run the external experiment with `python3 agentdiag_v2/benchmark_step.py --stepfinder-data /path/to/StepFinder/data`, then `python3 -m agentdiag_v2.hybrid_upgrade`, `python3 -m agentdiag_v2.verify_upgrade`, `python3 agentdiag_v2/verify.py`, and `python3 -m pytest agentdiag_v2/tests -q`. The previous L2=1.0 model and predictions are preserved in `benchmark_step/previous_l2_1/`; the previous hybrid snapshot is in `hybrid_upgrade/previous_step_model_l2_1/`.

Independent proof of benchmark leadership requires a frozen method, matched scorer, and new untouched evaluation data. The Windows machine has an RTX 2050 with 4 GB VRAM; only local resources were authorized for this run. A faculty grade of 18–19/20 cannot be guaranteed from code checks or these benchmark numbers, particularly while multimodal performance is untested.
