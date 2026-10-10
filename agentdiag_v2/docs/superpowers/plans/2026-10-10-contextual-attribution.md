# Contextual Attribution Implementation Plan

> **For agentic workers:** Execute this plan inline, task by task, with verification checkpoints. The user has authorized implementation.

**Goal:** Improve Algorithm-Generated agent and exact-step attribution with locally trained contextual models and a reproducible comparison.

**Architecture:** Use pinned frozen Qwen3 semantic embeddings, bidirectional temporal context, task interaction, and separately supervised agent/step probability heads. Compare against the saved sparse ranker and select only on external question-group validation.

**Tech Stack:** Python 3.12, PyTorch 2.8 CUDA 12.6, Transformers 4.57, NumPy, SciPy; RTX 2050 4 GB.

## Tasks

- [x] Establish isolated dependency environment; confirm CUDA and pinned encoder load.
- [x] Preserve `question` during unlabeled parsing and test independence from answer/attribution labels.
- [x] Implement content-addressed semantic caching; verify cache identity is based only on observed text and encoder configuration.
- [x] Implement temporal contextual model, explicit agent head, masked softmax and step distance loss.
- [x] Run focused tests on variable-length traces, altered labels, valid step indices and serialization. Five focused tests passed with PyTorch installed.
- [x] Reuse audited external training snapshot and fold assignments; train candidates and record external out-of-fold scores.
- [x] Freeze selected architecture, epoch count and blend; save selection provenance before benchmark access.
- [x] Train final checkpoints and run label-free original-corpus inference once; save every prediction, exact counts, bootstrap intervals and paired changes.
- [x] Integrate the selected improvement for Algorithm-Generated traces; document any failed target and all protocol limitations. Result: 73/126 agent and 42/126 step; 60%/35% remain unmet. Keep the old Hand-Crafted predictor.
- [x] Run existing tests, phase verifier, upgrade verifier and new model verification; review diff and commit verified changes. Evidence: 16 tests passed; 17 artifact checks; 13 previous upgrade checks; all 184 contextual checkpoint predictions reproduced.

## Exact verification commands

Use `S:/pr/agentdiag-mlenv/Scripts/python.exe` after installing `requirements-contextual.txt`.

```powershell
python -m pytest agentdiag_v2/tests -q
python agentdiag_v2/verify.py
python agentdiag_v2/verify_upgrade.py
python -m agentdiag_v2.contextual_upgrade --phase sparse --stepfinder-data S:/pr/StepFinder/data --cache S:/pr/agentdiag-model-cache
python -m agentdiag_v2.contextual_upgrade --phase encode --stepfinder-data S:/pr/StepFinder/data --cache S:/pr/agentdiag-model-cache
python -m agentdiag_v2.contextual_upgrade --phase neural --stepfinder-data S:/pr/StepFinder/data --cache S:/pr/agentdiag-model-cache
python -m agentdiag_v2.contextual_upgrade --phase select --stepfinder-data S:/pr/StepFinder/data --cache S:/pr/agentdiag-model-cache
python -m agentdiag_v2.contextual_upgrade --phase train --stepfinder-data S:/pr/StepFinder/data --cache S:/pr/agentdiag-model-cache
python -m agentdiag_v2.contextual_upgrade --phase evaluate --stepfinder-data S:/pr/StepFinder/data --cache S:/pr/agentdiag-model-cache
python -m agentdiag_v2.predict_contextual --trace <unlabeled-json> --cache S:/pr/agentdiag-model-cache
```

Report actual measured outcomes. Stop further work if the remaining usage allowance falls below the user's 25% threshold.
