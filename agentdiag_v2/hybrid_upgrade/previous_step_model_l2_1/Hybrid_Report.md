# Hybrid four-agent attribution

The step-model weight was selected inside each training fold. Across 91 question-group outer-fold predictions, the hybrid got **48/91** agents correct (52.7%), compared with **40/91** (44.0%) for the BN. Brier improved from 0.696 to 0.628.

The hybrid alone fixed 13 cases and the BN alone fixed 5; paired exact p = 0.096. This small-sample improvement is not conclusive.

With weight **0.4** chosen from train+validation question folds, the saved model got **6/13** on the original four-agent test, compared with **5/13** for the BN refit on the same train+validation cases. This test had already been inspected during earlier experiments, so its result is exploratory.

The hybrid requires the saved external step model. It remains below the published AgenTracer agent-accuracy results and has not been evaluated on Who&When Pro.
