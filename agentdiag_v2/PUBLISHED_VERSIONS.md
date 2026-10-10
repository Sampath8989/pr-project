# Published versions and result evidence

This upload contains runnable source code, dependencies, tests, original phase outputs, previous and current predictor checkpoints, benchmark predictions, result summaries, 26 PNG graphs and the updated review PDF.

## Version links

- [Current project](https://github.com/Sampath8989/pr-project/tree/main/agentdiag_v2)
- [Previous project snapshot](https://github.com/Sampath8989/pr-project/tree/69b1eb5cbc7a5ea6e440a027d8c258734edca925/agentdiag_v2)
- [Older predictor and results](benchmark_step/)
- [Updated predictor and results](contextual_upgrade/)
- [All result graphs and CSV evidence](results_graphs/)
- [Updated review PDF](output/pdf/AgentDiag_v2_Updated_Project_Review_Guide.pdf)

## Actual version comparison

| Predictor | Algorithm agent | Algorithm exact step | Hand-Crafted agent | Hand-Crafted exact step |
| --- | ---: | ---: | ---: | ---: |
| Previous ranker, L2=0.45 | 62/126 (49.2%) | 36/126 (28.6%) | 37/58 (63.8%) | 10/58 (17.2%) |
| Updated contextual model | 73/126 (57.9%) | 42/126 (33.3%) | 37/58 (63.8%) | 8/58 (13.8%) |

The updated model improves Algorithm-Generated diagnosis. Keep the older predictor for Hand-Crafted steps. These scores are exploratory original-corpus evaluations, not an independent matched paper win.

## Essential files versus local research outputs

The selected external validation predictions, aggregate candidate scores, selection hashes, original benchmark predictions and small trained checkpoints are included. The 20 intermediate neural/lexical cross-validation prediction files and training progress log are omitted from this publication to reduce download size. They remain preserved on the local `feature/contextual-attribution` research branch and can be regenerated with the documented training phases. The selection manifest records their hashes; reproducing selection from all candidate predictions requires regeneration or the full local research outputs.

No virtual environment, dependency installation, embedding cache, temporary render files or large frozen foundation-model weights are uploaded. The pinned public encoder is downloaded and cached when needed. The already-published original data and historical model outputs remain available.

The PDF was prepared before publication. Its statements that the upgrade was local and not yet pushed describe its preparation time; this page records the subsequent upload. Its complete inventory describes the full local research folder, including the intermediate files omitted here.
