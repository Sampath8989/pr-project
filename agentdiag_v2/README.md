# AgentDiag v2

An independent implementation of the **multi-agent LLM failure attribution** scope in the PR presentation. The previous `phase*.py` scripts and outputs are not imported or reused.

This version analyzes saved Who&When execution traces. It does not install or launch the original benchmark agents; the upstream benchmark authors ran systems to collect failures and then evaluated attribution methods on the saved logs. AgentDiag v2 has not been evaluated on live or multimodal agent inputs.

## Run

From the repository root:

```bash
python3 agentdiag_v2/run_pipeline.py
python3 agentdiag_v2/verify.py
python3 -m pytest agentdiag_v2/tests -q
```

Dependencies are listed in `agentdiag_v2/requirements.txt`. The bundled environment used for this run has NumPy, SciPy, Matplotlib, and pytest installed. The pipeline reads the existing Who&When data at `Agents_Failure_Attribution/Who&When` and writes only under `agentdiag_v2/artifacts/`. Use `--dataset PATH --output PATH --seed NUMBER` to override these defaults.

To diagnose one JSON trace after running the pipeline:

```bash
python3 agentdiag_v2/predict.py 'Agents_Failure_Attribution/Who&When/Algorithm-Generated/1.json'
python3 agentdiag_v2/predict.py 'Agents_Failure_Attribution/Who&When/Algorithm-Generated/1.json' --mask-agent-index 2
```

The single-trace command does **not** read `mistake_agent` or `mistake_step`, even when they are present in the input JSON. It accepts an unlabeled trace with a `history` list. Agent indices for `--mask-agent-index` are zero-based.

## Outputs by phase

| Phase | Main output | What it proves |
| --- | --- | --- |
| 0 | `artifacts/phase0_protocol/Phase0_Results.md` | Data audit and frozen question-group-safe splits. |
| 1 | `artifacts/phase1_parsing/Phase1_Results.md` | Parsed traces, exclusions, and label checks. |
| 2 | `artifacts/phase2_evidence_baselines/Phase2_Results.md` | Evidence features and simple/regularized baselines. |
| 3 | `artifacts/phase3_bayesian_inference/Phase3_Results.md` | Categorical BN, custom VE, brute-force cross-check. |
| 4 | `artifacts/phase4_missing_evidence/Phase4_Results.md` | Held-out missing-log scenarios. |
| 5 | `artifacts/phase5_calibration_evaluation/Phase5_Results.md` | Validation-fitted calibration and held-out attribution metrics. |
| 6 | `artifacts/phase6_temporal_hmm/Phase6_Results.md` | HMM warning results and false alarms. |
| 7 | `artifacts/phase7_multimodal_scope/Phase7_Results.md` | Audit of whether supplied data supports a multimodal claim. |
| 8 | `artifacts/phase8_final_report/Final_Report.md` | Overall result, limits, and PPT claim check. |
| 9 | `artifacts/phase9_evidence_triage/Phase9_Results.md` | Two-candidate review packets with source excerpts and model-sensitivity checks. |

Machine-readable files and the reliability diagram sit beside each report. A fresh, result-aligned `AgentDiag_v2_Final_Review.pptx` is generated in `artifacts/phase8_final_report/`; the previous PPT files are untouched. `artifacts/phase_manifest.json` lists all phase reports.
`verify.py` rechecks the saved predictions, metrics, splits, and phase files; it writes `artifacts/phase8_final_report/Verification_Report.md`.

The single-trace command now includes a `review_packet`: ranked agents, calibrated probabilities, the two-agent shortlist, message-indexed excerpts for model-selected flags, and the change in each candidate's probability when its own evidence is masked. This change measures **model sensitivity**, not a real-world causal effect. A warning accompanies the HMM timeline because its held-out early-warning performance was poor.

See [RESEARCH_UPGRADE.md](RESEARCH_UPGRADE.md) for the paper comparison, failed accuracy experiments, measured shortlist coverage, and remaining gaps. The new semantic and lexical rankers are not used for prediction because they did not reliably outperform the Bayesian network.

Run `python3 agentdiag_v2/accuracy_study.py` to regenerate the [91-trace grouped accuracy comparison](accuracy_study/Accuracy_Study.md). Its per-trace predictions are saved beside that report.

## External-data step attribution experiment

The [step benchmark report](benchmark_step/Benchmark_Report.md) evaluates a separate step-level model trained on filtered StepFinder data. It predicts both a raw message index and a responsible agent, using only observed trace text at inference. Its tuned saved model is at `benchmark_step/step_model.json` (L2=0.45); the previous L2=1.0 model and predictions are preserved in `benchmark_step/previous_l2_1/`. The original Bayesian predictor remains the default because the external model has not demonstrated a reliable agent-accuracy improvement on the primary four-agent split or beaten AgenTracer.

To diagnose one trace with the step model:

```bash
python3 agentdiag_v2/predict_step.py 'Agents_Failure_Attribution/Who&When/Algorithm-Generated/1.json'
```

To retrain and reevaluate, obtain the StepFinder repository at commit `48d71c7090adca3667a31b043a78484f407b4a2f`, including its `data` directory, then run:

```bash
python3 agentdiag_v2/benchmark_step.py --stepfinder-data /path/to/StepFinder/data
```

The audited question hashes and filtering method are in `data/stepfinder_safe_questions.json`. The final evaluation uses all 184 original Who&When traces with strict agent-name and raw-step matching. This is a research comparison, not a clean untouched holdout: model choices and the benchmark have been inspected during development. The official Who&When Pro repository lists its full benchmark release as pending, so this project does not claim a Who&When Pro score.

## Improved four-agent predictor

The [hybrid report](hybrid_upgrade/Hybrid_Report.md) combines the BN with the external step model. In nested question-group cross-validation it achieved **48/91** correct versus **40/91** for the BN. On the previously inspected 13-case test, the train-only selected hybrid got **6/13** versus **5/13** for a BN refit on the same data. The paired cross-validation improvement has an exact p-value of 0.077, so it is promising but not conclusive. This still does **not** beat AgenTracer.

Use it for a four-agent trace with:

```bash
python3 agentdiag_v2/predict_hybrid.py 'Agents_Failure_Attribution/Who&When/Algorithm-Generated/1.json'
```

`python3 agentdiag_v2/hybrid_upgrade.py` regenerates the nested evaluation and saved hybrid model. The original `predict.py` remains available for BN-only diagnoses and other agent counts.

Read [BENCHMARK_STATUS.md](BENCHMARK_STATUS.md) for a direct comparison with the older Who&When paper, AgenTracer, and Who&When Pro. Run `python3 agentdiag_v2/verify_upgrade.py` to recheck both upgrade result files.

To continue the research in another session, start with [ACCURACY_PUSH_PROMPT.md](ACCURACY_PUSH_PROMPT.md). It records the current models, scores, benchmark gaps, and the next evaluation steps.

For a simple-English project-review guide with a file-by-file explanation, result comparisons, limitations, commands, and sample viva answers, open [AgentDiag_v2_Project_Review_Guide.pdf](AgentDiag_v2_Project_Review_Guide.pdf). Rebuild it with `python3 agentdiag_v2/create_project_review_pdf.py` from the repository root.

## Model and evaluation protocol

- The root cause is **one categorical variable**, so every prediction is a probability distribution that sums to one. Candidate evidence variables include binary text signals and coarse agent-role flags in a small Bayesian dependency chain. The feature set is chosen using validation data.
- Variable Elimination is implemented directly in `src/bayes.py`; sampled outputs are checked against full joint enumeration.
- Related traces with the same question ID stay in the same train/validation/test partition. Feature selection, calibration temperature, and HMM threshold use validation data. The final reported scores use held-out test traces.
- Four-agent traces are the primary comparable experiment. Other available agent counts can be diagnosed with separately fitted models; tiny groups use prior-only fallbacks. The benchmark contains failed traces, so it cannot train a meaningful success-versus-failure output node.
- Temporal analysis excludes traces whose annotated mistake step does not point to the labeled culprit's message. HMM states are proxies defined from training annotations, not directly observed health states.

## What the outputs do not establish

A working inference engine does not guarantee useful accuracy, calibration, or early warning. Read the actual held-out results in `Final_Report.md`. The supplied Who&When traces are multi-agent, not a separately labeled image/audio/video benchmark; multimodal performance is **not** claimed. Published AgenTracer numbers are not treated as a direct comparison unless its predictions can be evaluated on the identical split and subset.
