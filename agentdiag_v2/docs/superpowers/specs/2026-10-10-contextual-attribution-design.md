# Contextual agent and step attribution

The user approved continuing the accuracy work after the diagnosis of the current sparse ranker. Targets are at least the older Who&When results and, if achievable, approximately 60% Algorithm-Generated agent accuracy and 35% exact step accuracy. These are research targets, not promised outcomes.

## Method and alternatives

Who&When prompts a judge with a question and an entire log, or searches the log incrementally. AgenTracer trains Qwen3-8B with GRPO on replay and fault-injection examples, rewarding both agent correctness and distance to the decisive step. StepFinder uses Qwen3 embeddings, bidirectional temporal modeling, agent interaction and step scoring. Sources: https://arxiv.org/html/2505.00212, https://arxiv.org/html/2509.03312, https://github.com/taiyu-zhu/StepFinder.

1. Extend sparse contextual features and directly supervise agent attribution. Low compute; limited semantic reasoning.
2. Frozen semantic encoder plus a compact temporal model with separate agent and step heads. Recommended locally feasible architecture; context can span the whole saved trace.
3. Train an 8B reasoning model with replay-based reinforcement learning. Too large for the available 4 GB GPU and no replay environment is supplied.

Implement option 2 and a sparse contextual comparator. The semantic encoder is pinned Qwen3-Embedding-0.6B. A compact bidirectional GRU and attention layer process message embeddings with question interactions and observed flags. A step head predicts a distribution over actual raw message indices. An agent head pools each agent's contextual messages using mean/max pooling and predicts a distribution over agents. Both are trained explicitly. A small optional distance-smoothed training loss is evaluated while exact step matching remains the evaluation metric.

## Protocol

Use only the previously audited 2,004 external StepFinder training traces from 58 question groups for the primary model. Keep the existing question-overlap filter. Use the frozen five-fold external question-group assignments. Candidate selection and blending use external out-of-fold predictions, with Algorithm-Generated agent/step accuracy as the primary target. Never select an epoch, seed, architecture or blend on Who&When labels. Freeze the selected configuration before one evaluation on the 184 original traces.

The original benchmark has already been inspected. Its results are exploratory and cannot establish an independent paper-matched victory. The old predictions and models remain available. A separately trained model's result must not be silently attributed to the Bayesian network.

Before neural training and original scoring, also predeclare an equal lexical/semantic probability mixture as a candidate. Its blend with the old model is selected on the same external folds; this permits complementary evidence from all three views.

## Files and outputs

- `src/semantic_encoder.py`: pinned local encoding and content-addressed, label-free cache.
- `src/contextual_ranker.py`: temporal model, masked probability heads and supervised losses.
- `contextual_upgrade.py`: filtered training, fold selection, ensemble, frozen benchmark and provenance.
- `predict_contextual.py`: label-free prediction command.
- `contextual_upgrade/`: fold metrics, selection manifest, chosen checkpoints, per-trace predictions, comparison report and verification.
- `tests/test_contextual.py`: label independence, padding, agent/step probability validity and save/load consistency.

## Acceptance

Run the actual training and evaluation locally; preserve errors rather than hiding them. Predict without reading target labels or ground-truth answers. Verify raw step indices, model serialization and probability normalization. Report old/new counts on the same records and paired uncertainty. Integrate only a measured improvement, otherwise preserve the old default and report the failed candidate. Retain the project's Bayesian inference, calibration and temporal analysis features.
