# AgentDiag v2 Recovery and Accuracy Implementation Plan

**Goal:** Restore a verifiable Windows copy, then test focused ways to improve the weakest attribution result using reproducible evaluation without overstating comparisons with published papers.

**Architecture:** Keep `agentdiag_v2` as the independent project. First repair portable artifact references and verify the existing pipeline. Then analyze saved per-trace mistakes, train any candidate only on allowed training data, choose settings inside grouped validation, and keep a new method only if measured results justify it. Preserve the Bayesian and step-model baselines for direct comparison.

**Tech Stack:** Python 3.12, NumPy, SciPy, pytest, JSON/Markdown artifacts. Local machine only.

---

## Evidence already established

- GitHub `main` at `69b1eb5` was cloned into `S:/pr/pr-project`. No `agentdiag_v2` folder existed in the supplied `S:/pr/project` tree.
- All 184 original Who&When JSON records exist in the clone and parse to the same objects as the local upstream copy.
- Baseline tests: 6 passed. Upgrade verification: 13 checks passed. Phase verifier: 16 of 17 checks passed; the saved phase manifest contains absolute Pop!_OS paths.
- Weakest published-paper comparison is step attribution on Hand-Crafted traces: 10/58 (17.2%) versus AgenTracer's reported 20.68%. The Algorithm-Generated agent result is 62/126 (49.2%) versus the older Who&When 51.12% and AgenTracer 63.73%. These are indicative comparisons because protocols differ.
- The four-agent nested grouped evaluation improved from Bayesian 40/91 to hybrid 48/91, but the paired p-value is 0.077 and the 13-case test was inspected during development.

## Task 1: Repair the migrated artifact paths

**Files:** `agentdiag_v2/artifacts/phase_manifest.json`, `agentdiag_v2/run_pipeline.py`, `agentdiag_v2/verify.py`, `agentdiag_v2/tests/test_pipeline.py`.

- [x] Add a regression check that every saved phase path is relative, uses forward slashes, and resolves to a real file.
- [x] Run the check against the committed manifest and confirm it fails on its Linux paths.
- [x] Run `python agentdiag_v2/run_pipeline.py` to regenerate the manifest. Change the writer to use `Path.as_posix()` for cross-platform paths; keep the source path portable as well.
- [x] Run `python agentdiag_v2/verify.py` and `python -m pytest agentdiag_v2/tests -q` and record exact results: 17/17 verifier checks and 8/8 tests on the latest run.
- [x] Inspect `git diff` and restore unrelated numerical/figure rendering noise. `.agent/config.yml` is absent; no commit or push has been made.

## Task 2: Diagnose the accuracy failures

**Files:** `agentdiag_v2/benchmark_step/full_benchmark_predictions.jsonl`, `agentdiag_v2/benchmark_step/metrics.json`, `agentdiag_v2/hybrid_upgrade/nested_fold_predictions.jsonl`, and new analysis artifacts under `agentdiag_v2/accuracy_push/`.

- [x] Identify the existing prediction file names and inspect their schema.
- [x] Count errors separately for Algorithm-Generated and Hand-Crafted, by agent error, step error, trace length, and label-step speaker mismatch.
- [x] Save a machine-readable error audit and a short Markdown report with example trace IDs.
- [x] Test limited candidate changes for step localization while leaving saved baseline predictions untouched.

## Task 3: Test a focused candidate

**Files:** new experiment scripts and results under `agentdiag_v2/accuracy_push/`, and focused tests under `agentdiag_v2/tests/`. The existing model code remains unchanged because none of the candidates passed selection.

- [x] Recheck the pinned StepFinder commit and audited question-overlap filter before training.
- [x] Run separate experiments: externally grouped agent-aware step selection; frozen validation selection for domain adaptation; externally grouped subset specialists. Do not select on Who&When test labels.
- [x] Measure exact agent, raw-step, and joint scores for all 184 original traces and each subset where applicable. Keep the old model because every candidate selection returned to it.
- [x] Run the paired step comparison and retain the previously saved bootstrap intervals. Label the inspected original benchmark as exploratory.
- [x] Check `.agent/config.yml` before any commit; it is absent. No commit or push was made.

## Task 4: Make the paper comparison honest and review-ready

**Files:** `agentdiag_v2/BENCHMARK_STATUS.md`, `agentdiag_v2/RESEARCH_UPGRADE.md`, `agentdiag_v2/README.md`, and new result artifact if the candidate is retained.

- [x] Verify the published numbers and evaluator definitions from the original papers and released code.
- [x] Present one table with our scores, older Who&When, and AgenTracer, including denominators and protocol differences. Audit the effect of the released substring scorer.
- [x] State what the first-review PPT asked for and what is demonstrated or unproved in `accuracy_push/Review_Readiness.md`.
- [x] Re-run the full pipeline, phase verifier, upgrade verifier, and tests after edits. Do not claim a grade or benchmark victory without matching evidence.

## Completion rule

The Windows copy is operational, and all new accuracy attempts were recorded. The selected model did not improve. The target of 18–19/20 or beating published systems remains an aspiration that cannot be guaranteed from the present data and local resources. The next research step is a fresh, independent, paper-matched evaluation plus a better early-warning method and real multimodal traces if the PPT's multimodal claim is required.
