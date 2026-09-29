#!/usr/bin/env python3
"""Reproduce the group-fold attribution comparison without changing deployment."""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentdiag_v2.src.bayes import fit_model, predict as bn_predict
from agentdiag_v2.src.data import feature_rows, load_records, write_json, write_jsonl
from agentdiag_v2.src.ranker import fit as fit_ranker, predict as rank_predict


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data" / "Who&When"
OUTPUT = ROOT / "accuracy_study"
FEATURES = ("error", "uncertain", "correction")


def scores(rows: list[dict], key: str) -> dict:
    labels = np.array([r["actual"] for r in rows])
    probabilities = np.array([r[key] for r in rows])
    one_hot = np.eye(4)[labels]
    return {"n": len(rows),
            "correct": int((probabilities.argmax(axis=1) == labels).sum()),
            "accuracy": float((probabilities.argmax(axis=1) == labels).mean()),
            "brier": float(((probabilities - one_hot) ** 2).sum(axis=1).mean()),
            "log_loss": float(-np.log(np.maximum(probabilities[np.arange(len(labels)), labels], 1e-12)).mean())}


def main() -> None:
    records, exclusions = load_records(DATASET)
    if exclusions:
        raise RuntimeError(f"Unexpected source exclusions: {exclusions}")
    rows_by_id = {r["id"]: r for r in feature_rows(records)}
    groups = defaultdict(list)
    for record in records:
        groups[record["question_id"]].append(record)
    keys = sorted(groups)
    random.Random(4929).shuffle(keys)
    fold_by_question = {key: index % 5 for index, key in enumerate(keys)}
    predictions = []
    for fold in range(5):
        train = [r for r in records if fold_by_question[r["question_id"]] != fold]
        evaluation = [r for r in records if fold_by_question[r["question_id"]] == fold
                      and len(r["agents"]) == 4]
        bn = fit_model([rows_by_id[r["id"]] for r in train], 4, FEATURES)
        ranker = fit_ranker(train, l2=1, components="text")
        ranked = rank_predict(ranker, evaluation)
        for record, rank_probabilities in zip(evaluation, ranked):
            predictions.append({"id": record["id"], "question_id": record["question_id"],
                                "fold": fold, "actual": record["culprit"],
                                "bn": bn_predict(bn, rows_by_id[record["id"]]["evidence"]),
                                "text_ranker": rank_probabilities})
    if len(predictions) != sum(len(r["agents"]) == 4 for r in records):
        raise AssertionError("Each four-agent trace must have one out-of-fold prediction")
    if len({p["id"] for p in predictions}) != len(predictions):
        raise AssertionError("Duplicate evaluation trace")
    metrics = {"protocol": "Five question-group folds on all supplied traces; train on four-agent and other agent-count traces, evaluate only four-agent traces.",
               "seed": 4929, "fold_count": 5, "model_selection_note":
               "This is an exploratory comparison. Ranker settings were investigated after the original test set had been inspected.",
               "bn": scores(predictions, "bn"), "text_ranker": scores(predictions, "text_ranker")}
    OUTPUT.mkdir(exist_ok=True)
    write_jsonl(OUTPUT / "group_fold_predictions.jsonl", predictions)
    write_json(OUTPUT / "results.json", metrics)
    (OUTPUT / "Accuracy_Study.md").write_text(
        "# Can top-choice accuracy be increased?\n\n"
        "A text ranker scored **7/13** on the previously inspected test set, compared with the BN's **5/13**. "
        "This was exploratory and is not a clean estimate of improvement.\n\n"
        "For a broader check, five question-group folds cover all 91 four-agent source traces. "
        f"The BN got **{metrics['bn']['correct']}/91** correct; the text ranker got "
        f"**{metrics['text_ranker']['correct']}/91**. The ranker therefore did not improve "
        "top-choice accuracy across groups. Its Brier score was "
        f"{metrics['text_ranker']['brier']:.3f} versus {metrics['bn']['brier']:.3f} for the BN; "
        "that is a different metric.\n\n"
        "The deployed BN stays in place. A convincing accuracy increase needs more representative labeled traces, "
        "a method selected without looking at a new test set, and an untouched external evaluation.\n",
        encoding="utf-8")
    print(f"BN {metrics['bn']['correct']}/91; text ranker {metrics['text_ranker']['correct']}/91")


if __name__ == "__main__":
    main()
