# External-data step attribution benchmark

Training: 2,004 filtered StepFinder traces from 58 questions. No original Who&When labels were used for training. Five-fold question-group CV selected L2=0.45. Exact agent names and exact raw step indices are scored.

- Algorithm-generated (126 traces): agent 62/126 (49.2%, 95% bootstrap CI 40.5%–57.9%), step 36/126 (28.6%, 95% bootstrap CI 20.6%–36.5%), both 33/126.
- Hand-crafted (58 traces): agent 37/58 (63.8%, 95% bootstrap CI 51.7%–75.9%), step 10/58 (17.2%, 95% bootstrap CI 8.6%–27.6%), both 8/58.

The newer [AgenTracer paper](https://arxiv.org/abs/2509.03312) reports, without ground-truth trajectory access, agent/step accuracy of 63.73%/37.30% on algorithm-generated and 63.82%/20.68% on handcrafted Who&When. This candidate does not beat those benchmarks. The older [Who&When paper](https://arxiv.org/abs/2505.00212) used different judge protocols and its released scorer has permissive substring comparisons, so a direct victory claim would be unsafe.

The filter removed external questions with token Jaccard similarity ≥0.85 or BGE-small cosine similarity ≥0.75 to any of the 184 benchmark questions. The selection metric averaged agent and step accuracy equally over both external subsets. Even this cannot exclude all task-family overlap. Model parameters were explored during this project's development; the benchmark is no longer an untouched holdout. The official [Who&When Pro repository](https://github.com/ag2ai/whowhen_pro) lists its full benchmark release as pending, so no Pro score is claimed.
