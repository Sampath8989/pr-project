# Phase 0 — Frozen protocol

- Source: `data/Who&When`. Parsed 184 of 184 traces.
- Question-group-safe split with seed 20260929: train 134, validation 27, test 23.
- Primary experiment: four-agent traces; counts by split: {'train': {4: 65, 2: 25, 3: 40, 1: 2, 5: 2}, 'validation': {4: 13, 3: 7, 1: 1, 2: 6}, 'test': {4: 13, 3: 7, 2: 3}}.
- Post-mortem attribution uses full logs. Early warning uses only messages observed so far.
- Model/calibration/threshold selection never uses test labels.
