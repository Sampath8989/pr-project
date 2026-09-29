#!/usr/bin/env python3
"""Train an external-data step ranker and evaluate exact Who&When labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentdiag_v2.src.data import load_records, parse_unlabeled, write_json, write_jsonl
from agentdiag_v2.src.step_ranker import fit, predict


HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE.parent / "Agents_Failure_Attribution" / "Who&When"
SAFE = HERE / "data" / "stepfinder_safe_questions.json"
OUTPUT = HERE / "benchmark_step"
L2_CANDIDATES = (0.25, 0.35, 0.45)
FOLD_SEED = 8229


def evaluation(records: list[dict], predictions: list[dict]) -> dict:
    by_subset = {}
    for subset in sorted({benchmark_subset(r) for r in records}):
        group = [(r, p) for r, p in zip(records, predictions) if benchmark_subset(r) == subset]
        agent = sum(p["predicted_agent"] == r["culprit_name"] for r, p in group)
        step = sum(p["predicted_step"] == r["mistake_step"] for r, p in group)
        both = sum(p["predicted_agent"] == r["culprit_name"] and p["predicted_step"] == r["mistake_step"]
                   for r, p in group)
        by_subset[subset] = {"traces": len(group), "correct_agent": agent, "agent_accuracy": agent / len(group),
                             "correct_step": step, "step_accuracy": step / len(group),
                             "correct_both": both, "joint_accuracy": both / len(group)}
    return by_subset


def exact_metrics(rows: list[dict], predictions: list[dict]) -> dict:
    result = evaluation(rows, predictions)
    result["Overall"] = {
        "traces": len(rows),
        "correct_agent": sum(p["predicted_agent"] == r["culprit_name"] for r, p in zip(rows, predictions)),
        "agent_accuracy": sum(p["predicted_agent"] == r["culprit_name"] for r, p in zip(rows, predictions)) / len(rows),
        "correct_step": sum(p["predicted_step"] == r["mistake_step"] for r, p in zip(rows, predictions)),
        "step_accuracy": sum(p["predicted_step"] == r["mistake_step"] for r, p in zip(rows, predictions)) / len(rows),
        "correct_both": sum(p["predicted_agent"] == r["culprit_name"]
                            and p["predicted_step"] == r["mistake_step"]
                            for r, p in zip(rows, predictions)),
        "joint_accuracy": sum(p["predicted_agent"] == r["culprit_name"]
                              and p["predicted_step"] == r["mistake_step"]
                              for r, p in zip(rows, predictions)) / len(rows),
    }
    return result


def benchmark_subset(record: dict) -> str:
    return record["id"].split("/", 1)[0]


def bootstrap_interval(rows: list[dict], predictions: list[dict], metric: str,
                       seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    values = np.asarray([p["predicted_agent"] == r["culprit_name"] if metric == "agent"
                         else p["predicted_step"] == r["mistake_step"]
                         for r, p in zip(rows, predictions)], dtype=float)
    draws = rng.integers(0, len(values), size=(10000, len(values)))
    scores = values[draws].mean(axis=1)
    return [float(v) for v in np.quantile(scores, [0.025, 0.975])]


def external_group_cv(train: list[dict]) -> tuple[float, dict, dict, dict]:
    """Choose L2 on five folds, keeping each external question group intact."""
    groups = sorted({r["question_id"] for r in train})
    random.Random(FOLD_SEED).shuffle(groups)
    fold_by_question = {question_id: index % 5 for index, question_id in enumerate(groups)}
    predictions_by_l2 = {l2: [] for l2 in L2_CANDIDATES}
    for fold in range(5):
        fitting = [r for r in train if fold_by_question[r["question_id"]] != fold]
        validation = [r for r in train if fold_by_question[r["question_id"]] == fold]
        for l2 in L2_CANDIDATES:
            model = fit(fitting, l2=l2)
            predictions = predict(model, validation)
            predictions_by_l2[l2].extend((r, p) for r, p in zip(validation, predictions))
    scores = {}
    serialized = {}
    for l2, pairs in predictions_by_l2.items():
        source_rows = [r for r, _ in pairs]
        outputs = [p for _, p in pairs]
        summary = exact_metrics(source_rows, outputs)
        subset_scores = []
        for subset in ("Algorithm-Generated", "Hand-Crafted"):
            group = [(r, p) for r, p in pairs if benchmark_subset(r) == subset]
            group_metrics = exact_metrics([r for r, _ in group], [p for _, p in group])
            subset_scores.extend(group_metrics[subset][key] for key in ("agent_accuracy", "step_accuracy"))
        scores[str(l2)] = {"metrics": summary, "selection_score_macro_subset_agent_step": float(np.mean(subset_scores))}
        serialized[str(l2)] = [
            {"id": r["id"], "benchmark_subset": benchmark_subset(r),
             "question_id": r["question_id"], "outer_fold": fold_by_question[r["question_id"]],
             "actual_agent": r["culprit_name"], "actual_step": r["mistake_step"], **p}
            for r, p in pairs]
    selected = max(L2_CANDIDATES, key=lambda value: (
        scores[str(value)]["selection_score_macro_subset_agent_step"],
        scores[str(value)]["metrics"]["Overall"]["agent_accuracy"],
        scores[str(value)]["metrics"]["Overall"]["step_accuracy"], -value))
    fold_manifest = {"seed": FOLD_SEED, "fold_count": 5,
                     "question_to_fold": fold_by_question,
                     "train_ids_by_fold": {str(fold): sorted(r["id"] for r in train
                                                  if fold_by_question[r["question_id"]] != fold)
                                           for fold in range(5)},
                     "validation_ids_by_fold": {str(fold): sorted(r["id"] for r in train
                                                       if fold_by_question[r["question_id"]] == fold)
                                                for fold in range(5)}}
    return float(selected), scores, serialized, fold_manifest


def run(stepfinder_data: Path, output: Path) -> dict:
    allowed = json.loads(SAFE.read_text())
    expected = set(allowed["safe_question_sha256"])
    external, external_exclusions = load_records(stepfinder_data)
    train = [r for r in external if "/train/" in r["id"] and r["label_message_matches"]
             and 2 <= len(r["agents"]) <= 4
             and hashlib.sha256(r["question"].strip().encode()).hexdigest() in expected]
    if len(train) != 2004 or len({r["question"].strip() for r in train}) != 58:
        raise RuntimeError("External data does not match the audited StepFinder snapshot and question filter")
    selected_l2, tuning, cv_predictions, fold_manifest = external_group_cv(train)
    model = fit(train, l2=selected_l2)
    original, original_exclusions = load_records(DEFAULT_DATA)
    training_exclusions = [r for r in external_exclusions if "/train/" in r["id"]]
    if len(training_exclusions) != 1 or original_exclusions:
        raise RuntimeError("Unexpected JSON exclusions in benchmark data")
    # Label-free parse at inference time to catch accidental target leakage.
    unlabeled = [parse_unlabeled(DEFAULT_DATA / r["id"]) for r in original]
    predictions = predict(model, unlabeled)
    for record, result in zip(original, predictions):
        if len(result["step_probabilities"]) != len(record["messages"]):
            raise AssertionError("Step probability vector does not match observed messages")
        if abs(sum(result["agent_probabilities"]) - 1) > 1e-9:
            raise AssertionError("Agent probabilities do not sum to one")
    metrics = exact_metrics(original, predictions)
    for subset_index, subset in enumerate(("Algorithm-Generated", "Hand-Crafted")):
        group = [(r, p) for r, p in zip(original, predictions) if r["subset"] == subset]
        metrics[subset]["agent_accuracy_95pct_bootstrap_interval"] = bootstrap_interval(
            [r for r, _ in group], [p for _, p in group], "agent", 53 + subset_index)
        metrics[subset]["step_accuracy_95pct_bootstrap_interval"] = bootstrap_interval(
            [r for r, _ in group], [p for _, p in group], "step", 71 + subset_index)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "step_model.json", model | {"training_source": allowed["source"],
                                                   "source_commit": allowed["commit"],
                                                   "safe_question_count": 58,
                                                   "l2_selected_by_external_question_group_cv": selected_l2})
    write_jsonl(output / "full_benchmark_predictions.jsonl", [
        {"id": r["id"], "subset": r["subset"], "actual_agent": r["culprit_name"],
         "actual_step": r["mistake_step"], **p}
        for r, p in zip(original, predictions)])
    write_json(output / "metrics.json", {"training_traces": len(train),
                                         "training_questions": 58, "metrics": metrics,
                                         "selected_l2": selected_l2,
                                         "external_group_cv_tuning": tuning,
                                         "protocol": "External StepFinder training only; exact and semantic near-duplicate question filters; five-fold question-group CV selects L2 on external data; strict exact agent and raw-step scoring on all 184 original Who&When traces.",
                                         "limitations": "The benchmark corpus and external training corpus derive from related task sources. The original Who&When has been inspected during project development; this is not an independently collected test set; paper evaluator details may differ."})
    write_json(output / "external_training_folds.json", fold_manifest)
    write_json(output / "external_training_cv_metrics.json", tuning)
    for l2, rows in cv_predictions.items():
        write_jsonl(output / f"external_cv_predictions_l2_{str(l2).replace('.', '_')}.jsonl", rows)
    source_manifest = {"external_commit": allowed["commit"],
                       "training_trace_ids_sha256": hashlib.sha256(
                           "\n".join(sorted(r["id"] for r in train)).encode()).hexdigest(),
                       "training_question_hashes": allowed["safe_question_sha256"],
                       "benchmark_trace_ids_sha256": hashlib.sha256(
                           "\n".join(sorted(r["id"] for r in original)).encode()).hexdigest(),
                       "question_filter": allowed["question_filter"],
                       "selection_metric": "Mean of agent and exact-step accuracy over Algorithm-Generated and Hand-Crafted external question-group CV folds.",
                       "selected_l2": selected_l2}
    write_json(output / "provenance.json", source_manifest)
    a, h = metrics["Algorithm-Generated"], metrics["Hand-Crafted"]
    (output / "Benchmark_Report.md").write_text(
        "# External-data step attribution benchmark\n\n"
        f"Training: {len(train):,} filtered StepFinder traces from 58 questions. No original Who&When labels "
        f"were used for training. Five-fold question-group CV selected L2={selected_l2:g}. "
        "Exact agent names and exact raw step indices are scored.\n\n"
        f"- Algorithm-generated (126 traces): agent {a['correct_agent']}/126 "
        f"({a['agent_accuracy']:.1%}, 95% bootstrap CI {a['agent_accuracy_95pct_bootstrap_interval'][0]:.1%}–{a['agent_accuracy_95pct_bootstrap_interval'][1]:.1%}), "
        f"step {a['correct_step']}/126 ({a['step_accuracy']:.1%}, 95% bootstrap CI {a['step_accuracy_95pct_bootstrap_interval'][0]:.1%}–{a['step_accuracy_95pct_bootstrap_interval'][1]:.1%}), "
        f"both {a['correct_both']}/126.\n"
        f"- Hand-crafted (58 traces): agent {h['correct_agent']}/58 "
        f"({h['agent_accuracy']:.1%}, 95% bootstrap CI {h['agent_accuracy_95pct_bootstrap_interval'][0]:.1%}–{h['agent_accuracy_95pct_bootstrap_interval'][1]:.1%}), "
        f"step {h['correct_step']}/58 ({h['step_accuracy']:.1%}, 95% bootstrap CI {h['step_accuracy_95pct_bootstrap_interval'][0]:.1%}–{h['step_accuracy_95pct_bootstrap_interval'][1]:.1%}), "
        f"both {h['correct_both']}/58.\n\n"
        "The newer [AgenTracer paper](https://arxiv.org/abs/2509.03312) reports, without ground-truth trajectory access, agent/step "
        "accuracy of 63.73%/37.30% on algorithm-generated and 63.82%/20.68% on handcrafted Who&When. "
        "This candidate does not beat those benchmarks. The older [Who&When paper](https://arxiv.org/abs/2505.00212) used different "
        "judge protocols and its released scorer has permissive substring comparisons, so a direct "
        "victory claim would be unsafe.\n\n"
        "The filter removed external questions with token Jaccard similarity ≥0.85 or BGE-small "
        "cosine similarity ≥0.75 to any of the 184 benchmark questions. The selection metric averaged "
        "agent and step accuracy equally over both external subsets. Even this cannot exclude "
        "all task-family overlap. Model parameters were explored during this project's development; "
        "the benchmark is no longer an untouched holdout. The official "
        "[Who&When Pro repository](https://github.com/ag2ai/whowhen_pro) lists its full benchmark "
        "release as pending, so no Pro score is claimed.\n",
        encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stepfinder-data", type=Path, required=True,
                        help="Path to the data directory at the pinned StepFinder commit in data/stepfinder_safe_questions.json")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    metrics = run(args.stepfinder_data, args.output)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
