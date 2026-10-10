# Saved result graphs and evidence

These figures visualize saved experiments; no new model training was performed.

## Use in your report

Graphs support the reported results. The prediction CSV, source JSON files, source hashes, and reproduction checks provide the underlying evidence. A graph alone does not prove generalization.

Original-corpus results are exploratory. The contextual model has not been evaluated on live or image/audio/video inputs. Bayesian and contextual results belong to different models and evaluation sets. Published numbers in reference charts are taken from the saved Contextual_Report.md: [Who&When](https://arxiv.org/html/2505.00212) and [AgenTracer](https://arxiv.org/html/2509.03312). They are no-answer-supplied reported results, not independently rerun or matched comparisons.

## Figures

- [01_agent_accuracy.png](01_agent_accuracy.png): Saved original-corpus evaluation; exploratory. Overall combines different subsets.
- [01_step_accuracy.png](01_step_accuracy.png): Saved original-corpus evaluation; exploratory. Overall combines different subsets.
- [01_joint_accuracy.png](01_joint_accuracy.png): Saved original-corpus evaluation; exploratory. Overall combines different subsets.
- [02_dataset_sizes.png](02_dataset_sizes.png): 184 saved Who&When conversations; this figure is an inventory, not an accuracy result.
- [03_algorithm_correct_counts.png](03_algorithm_correct_counts.png): Same 126 conversations for both versions.
- [04_algorithm_gains_losses.png](04_algorithm_gains_losses.png): Show gains and regressions together. Net improvement = gains minus losses.
- [05_algorithm_uncertainty.png](05_algorithm_uncertainty.png): Saved 95% trace-bootstrap intervals; they do not remove development-set reuse or question dependence.
- [03_handcrafted_correct_counts.png](03_handcrafted_correct_counts.png): Same 58 conversations for both versions.
- [04_handcrafted_gains_losses.png](04_handcrafted_gains_losses.png): Show gains and regressions together. Net improvement = gains minus losses.
- [05_handcrafted_uncertainty.png](05_handcrafted_uncertainty.png): Saved 95% trace-bootstrap intervals; they do not remove development-set reuse or question dependence.
- [06_algorithm_outcomes.png](06_algorithm_outcomes.png): Computed directly from individual saved predictions.
- [07_algorithm_step_distance.png](07_algorithm_step_distance.png): Absolute raw-message index distance. Near misses do not count toward exact step accuracy.
- [06_handcrafted_outcomes.png](06_handcrafted_outcomes.png): Computed directly from individual saved predictions.
- [07_handcrafted_step_distance.png](07_handcrafted_step_distance.png): Absolute raw-message index distance. Near misses do not count toward exact step accuracy.
- [08_validation_vs_original.png](08_validation_vs_original.png): Different datasets. Validation was used for selection; it is not an independent final benchmark.
- [09_bn_calibration_scores.png](09_bn_calibration_scores.png): Separate frozen Bayesian model, n=13. Lower is better. Top-choice accuracy stays 5/13.
- [10_bn_reliability.png](10_bn_reliability.png): Separate frozen Bayesian model, n=13. Tiny bins make this estimate noisy; not the contextual model.
- [11_missing_evidence.png](11_missing_evidence.png): Separate Bayesian model; 13 examples per condition. A higher small-sample score does not prove missing evidence helps.
- [12_early_warning.png](12_early_warning.png): Higher early recall is better; lower wrong-agent warning rate is better. HMM remains unreliable.
- [13_shortlist.png](13_shortlist.png): Separate Bayesian model, n=13. Shortlist coverage is not top-choice accuracy.
- [14_hybrid_grouped_validation.png](14_hybrid_grouped_validation.png): Separate 91-example grouped evaluation; paired p=0.0768. Improvement is not conclusive.
- [15_baselines_validation_4agent.png](15_baselines_validation_4agent.png): Original four-agent split, n=13. Separate from the 184-example contextual evaluation.
- [15_baselines_test_4agent.png](15_baselines_test_4agent.png): Original four-agent split, n=13. Separate from the 184-example contextual evaluation.
- [16_paper_reference_algorithm.png](16_paper_reference_algorithm.png): No answer supplied. Paper numbers are references, not a matched head-to-head or a proven paper win.
- [16_paper_reference_handcrafted.png](16_paper_reference_handcrafted.png): No answer supplied. Paper numbers are references, not a matched head-to-head or a proven paper win.
- [17_algorithm_targets.png](17_algorithm_targets.png): Targets remain unmet. At n=126, reaching them requires 76 correct agents and 45 correct steps.

## Evidence files

- prediction_evidence.csv: all 184 actual and predicted answers.
- graph_values.csv: values plotted by the bar charts.
- saved_phase_metrics.csv: numeric scalar metrics from saved phase reports.
- source_manifest.json: source hashes, figure captions, and checked prediction count.

## Reproduce

Run `python agentdiag_v2/export_result_graphs.py` from the repository parent with matplotlib, numpy and Pillow installed. All PNGs are checked for valid image encoding. The generator asserts that prediction counts match the saved accuracy metrics.
