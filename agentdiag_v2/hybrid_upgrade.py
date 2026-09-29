#!/usr/bin/env python3
"""Train and evaluate the four-agent BN plus external step-model blend."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentdiag_v2.src.bayes import fit_model, predict as bn_predict
from agentdiag_v2.src.data import feature_rows, load_records, write_json, write_jsonl
from agentdiag_v2.src.step_ranker import predict as step_predict


HERE = Path(__file__).resolve().parent
DATASET = HERE.parent / "Agents_Failure_Attribution" / "Who&When"
STEP_MODEL = HERE / "benchmark_step" / "step_model.json"
SPLITS = HERE / "artifacts" / "phase0_protocol" / "splits.json"
OUTPUT = HERE / "hybrid_upgrade"
FEATURES = ("error", "uncertain", "correction")
WEIGHTS = [i / 10 for i in range(11)]


def metrics(rows: list[dict], key: str) -> dict:
    actual = np.asarray([r["actual"] for r in rows])
    probabilities = np.asarray([r[key] for r in rows])
    one_hot = np.eye(4)[actual]
    return {"n": len(rows), "correct": int((probabilities.argmax(axis=1) == actual).sum()),
            "accuracy": float((probabilities.argmax(axis=1) == actual).mean()),
            "brier": float(((probabilities - one_hot) ** 2).sum(axis=1).mean())}


def choose_weight(records: list[dict], feature_by_id: dict, step_by_id: dict,
                  folds: int, seed: int) -> tuple[float, list[dict]]:
    question_ids = sorted({record["question_id"] for record in records})
    random.Random(seed).shuffle(question_ids)
    fold_by_question = {question_id: index % folds for index, question_id in enumerate(question_ids)}
    predictions = []
    for fold in range(folds):
        train = [r for r in records if fold_by_question[r["question_id"]] != fold]
        evaluation = [r for r in records if fold_by_question[r["question_id"]] == fold
                      and len(r["agents"]) == 4]
        bn = fit_model([feature_by_id[r["id"]] for r in train], 4, FEATURES)
        for record in evaluation:
            predictions.append({"id": record["id"], "actual": record["culprit"],
                                "bn": bn_predict(bn, feature_by_id[record["id"]]["evidence"]),
                                "step": step_by_id[record["id"]]["agent_probabilities"]})
    actual = np.asarray([r["actual"] for r in predictions])
    bn_probabilities = np.asarray([r["bn"] for r in predictions])
    step_probabilities = np.asarray([r["step"] for r in predictions])
    one_hot = np.eye(4)[actual]
    choices = []
    for weight in WEIGHTS:
        probabilities = (1 - weight) * bn_probabilities + weight * step_probabilities
        accuracy = float((probabilities.argmax(axis=1) == actual).mean())
        brier = float(((probabilities - one_hot) ** 2).sum(axis=1).mean())
        choices.append((accuracy, -brier, -weight, weight))
    # Accuracy is primary; Brier and lower step weight break ties.
    return max(choices)[3], predictions


def run() -> dict:
    records, exclusions = load_records(DATASET)
    if exclusions:
        raise RuntimeError("Unexpected source exclusions")
    by_id = {r["id"]: r for r in records}
    features = {r["id"]: r for r in feature_rows(records)}
    split = json.loads(SPLITS.read_text())["splits"]
    step_model = json.loads(STEP_MODEL.read_text())
    # Predictions use unlabeled records: step_predict reads messages only.
    from agentdiag_v2.src.data import parse_unlabeled
    step_outputs = step_predict(step_model, [parse_unlabeled(DATASET / r["id"]) for r in records])
    step_by_id = dict(zip((r["id"] for r in records), step_outputs))

    question_ids = sorted({r["question_id"] for r in records})
    random.Random(4929).shuffle(question_ids)
    outer_fold = {question_id: index % 5 for index, question_id in enumerate(question_ids)}
    outer_rows = []
    selected_weights = []
    for fold in range(5):
        train = [r for r in records if outer_fold[r["question_id"]] != fold]
        evaluation = [r for r in records if outer_fold[r["question_id"]] == fold
                      and len(r["agents"]) == 4]
        weight, _ = choose_weight(train, features, step_by_id, 4, 1000 + fold)
        selected_weights.append(weight)
        bn = fit_model([features[r["id"]] for r in train], 4, FEATURES)
        for record in evaluation:
            bn_probabilities = bn_predict(bn, features[record["id"]]["evidence"])
            step_probabilities = step_by_id[record["id"]]["agent_probabilities"]
            blend = [(1 - weight) * a + weight * b for a, b in zip(bn_probabilities, step_probabilities)]
            outer_rows.append({"id": record["id"], "fold": fold, "actual": record["culprit"],
                               "selected_weight": weight, "bn": bn_probabilities, "hybrid": blend})
    if len(outer_rows) != 91 or len({r["id"] for r in outer_rows}) != 91:
        raise AssertionError("Nested evaluation must cover every four-agent trace exactly once")
    actual_outer = np.asarray([r["actual"] for r in outer_rows])
    bn_correct = np.asarray([np.argmax(r["bn"]) for r in outer_rows]) == actual_outer
    hybrid_correct = np.asarray([np.argmax(r["hybrid"]) for r in outer_rows]) == actual_outer
    hybrid_only = int((hybrid_correct & ~bn_correct).sum())
    bn_only = int((bn_correct & ~hybrid_correct).sum())
    paired_p = float(binomtest(hybrid_only, hybrid_only + bn_only, 0.5).pvalue)

    train_ids = split["train"] + split["validation"]
    train = [by_id[i] for i in train_ids]
    test = [by_id[i] for i in split["test"] if len(by_id[i]["agents"]) == 4]
    weight, inner_rows = choose_weight(train, features, step_by_id, 5, 1900)
    final_bn = fit_model([features[r["id"]] for r in train], 4, FEATURES)
    test_rows = []
    for record in test:
        bn_probabilities = bn_predict(final_bn, features[record["id"]]["evidence"])
        step_probabilities = step_by_id[record["id"]]["agent_probabilities"]
        blend = [(1 - weight) * a + weight * b for a, b in zip(bn_probabilities, step_probabilities)]
        test_rows.append({"id": record["id"], "actual": record["culprit"], "agents": record["agents"],
                          "bn": bn_probabilities, "step": step_probabilities, "hybrid": blend})
    result = {"method": "Four-agent BN and externally trained step-ranker probability blend",
              "selected_inner_weight": weight, "outer_fold_weights": selected_weights,
              "nested_group_cv": {"bn": metrics(outer_rows, "bn"), "hybrid": metrics(outer_rows, "hybrid")},
              "paired_comparison": {"hybrid_only_correct": hybrid_only, "bn_only_correct": bn_only,
                                    "exact_mcnemar_p_value": paired_p},
              "original_test_after_refit": {"bn": metrics(test_rows, "bn"), "hybrid": metrics(test_rows, "hybrid")},
              "caveat": "Original 13-case test was inspected during previous experiments; nested group CV is stronger evidence but also exploratory because this method was developed after benchmark inspection."}
    OUTPUT.mkdir(exist_ok=True)
    write_json(OUTPUT / "hybrid_model.json", {"bn": final_bn, "step_weight": weight,
                                               "step_model_path": "../benchmark_step/step_model.json",
                                               "four_agent_only": True})
    write_jsonl(OUTPUT / "nested_fold_predictions.jsonl", outer_rows)
    write_jsonl(OUTPUT / "test_predictions.jsonl", test_rows)
    write_json(OUTPUT / "metrics.json", result)
    ncv, tst = result["nested_group_cv"], result["original_test_after_refit"]
    (OUTPUT / "Hybrid_Report.md").write_text(
        "# Hybrid four-agent attribution\n\n"
        f"The step-model weight was selected inside each training fold. Across 91 question-group "
        f"outer-fold predictions, the hybrid got **{ncv['hybrid']['correct']}/91** agents correct "
        f"({ncv['hybrid']['accuracy']:.1%}), compared with **{ncv['bn']['correct']}/91** "
        f"({ncv['bn']['accuracy']:.1%}) for the BN. Brier improved from "
        f"{ncv['bn']['brier']:.3f} to {ncv['hybrid']['brier']:.3f}.\n\n"
        f"The hybrid alone fixed {hybrid_only} cases and the BN alone fixed {bn_only}; "
        f"paired exact p = {paired_p:.3f}. This small-sample improvement is not conclusive.\n\n"
        f"With weight **{weight:.1f}** chosen from train+validation question folds, the saved "
        f"model got **{tst['hybrid']['correct']}/13** on the original four-agent test, compared "
        f"with **{tst['bn']['correct']}/13** for the BN refit on the same train+validation cases. "
        "This test had already been inspected during earlier experiments, so its result is exploratory.\n\n"
        "The hybrid requires the saved external step model. It remains below the published "
        "AgenTracer agent-accuracy results and has not been evaluated on Who&When Pro.\n",
        encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
