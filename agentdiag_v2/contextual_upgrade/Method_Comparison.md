# Why this architecture was tested

| Method | Information and architecture | Training objective | Local feasibility |
| --- | --- | --- | --- |
| Who&When | GPT-4o judges a question and complete log, a growing prefix, or halves of a log | Prompted attribution; no dedicated training in the cited GPT-4o comparison | Needs model access or a different local judge; results cannot be copied as ours |
| AgenTracer | Qwen3-8B reasons over the trajectory; data comes from counterfactual replay and fault injection | GRPO, agent correctness, Gaussian step-distance reward, output format | The paper uses eight H100 80 GB GPUs; this project has an RTX 2050 4 GB and no agent replay environment |
| StepFinder | Qwen3 embeddings, bidirectional LSTM, agent interaction attention, step score and temporal differences | Step localization and temporal consistency | Compact trainable model; pretrained embedding stage runs locally |
| Previous AgentDiag step ranker | Words of a message and its successor, role and positional features; sparse linear score | Exact step likelihood; agent derived by summing step probabilities | Small and fast, but limited task and conversation context |
| New lexical comparator | Word pairs, previous/subsequent messages, pooled authored content, separate agent classifier | Separate agent and exact-step likelihood | Small, runs on CPU; useful complementary errors |
| New semantic candidates | Frozen Qwen3-Embedding-0.6B; task interaction; bidirectional GRU and global attention; mean/max agent pooling | Direct agent and exact-step cross entropy, with an optional small distance loss | Encoder uses GPU; small trainable model uses cached vectors |

Sources: [Who&When](https://arxiv.org/html/2505.00212), [AgenTracer](https://arxiv.org/html/2509.03312), [StepFinder authors' code](https://github.com/taiyu-zhu/StepFinder), [Qwen model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).

The new model is a locally feasible candidate, not a reproduction of AgenTracer's replay or reinforcement learning. Its architectural differences are a research contribution candidate; this comparison does not establish worldwide novelty or better performance. Only the frozen selected model's measured results justify an improvement claim.

## Selection and accuracy

The same audited 2,004 external traces and 58 question groups are used. The existing five folds keep a question's related traces together. The contextual candidates compare temporal modeling, global attention, a step-distance loss, two fixed seeds, and three epoch counts. Agent and step blends are selected using external Algorithm-Generated accuracy, with Hand-Crafted accuracy as a tie breaker. Every setting is chosen before the new original-corpus predictions are scored.

The original Who&When data has already been inspected during development. It is an exploratory evaluation, even though this round selects its settings using external data. Neither the reported paper percentages nor our validation scores guarantee performance on new questions.

## Simple explanation

The old model looked mainly for useful words near a message. The new candidates can compare messages across the conversation, relate them to the original task, and learn which agent is responsible directly. We also test a combination of models because different models can make different mistakes. We keep the measured results of all attempted settings and use exact message numbers when checking the answers.
