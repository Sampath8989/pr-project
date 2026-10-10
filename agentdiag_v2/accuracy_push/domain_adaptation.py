"""Test limited domain adaptation on the frozen Who&When train/validation/test split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from agentdiag_v2.benchmark_step import exact_metrics
from agentdiag_v2.src.data import load_records, parse_unlabeled, write_json, write_jsonl
from agentdiag_v2.src.step_ranker import fit, predict


HERE = Path(__file__).resolve().parents[1]
MULTIPLIERS = (0, 4, 12)


def score(records: list[dict], predictions: list[dict]) -> dict:
    return exact_metrics(records, predictions)


def run(stepfinder_data: Path, output: Path) -> dict:
    allowed = json.loads((HERE / "data" / "stepfinder_safe_questions.json").read_text())
    safe_hashes = set(allowed["safe_question_sha256"])
    external, _ = load_records(stepfinder_data)
    train_external = [r for r in external if "/train/" in r["id"] and r["label_message_matches"]
                      and 2 <= len(r["agents"]) <= 4
                      and hashlib.sha256(r["question"].strip().encode()).hexdigest() in safe_hashes]
    if len(train_external) != 2004:
        raise ValueError("StepFinder filtered training data does not match the prior experiment")
    original, exclusions = load_records(HERE / "data" / "Who&When")
    if len(original) != 184 or exclusions:
        raise ValueError("Original benchmark changed")
    splits = json.loads((HERE / "artifacts/phase0_protocol/splits.json").read_text())["splits"]
    by_id = {r["id"]: r for r in original}
    if len({rid for part in splits.values() for rid in part}) != 184:
        raise ValueError("Frozen split changed")
    train = [by_id[rid] for rid in splits["train"] if by_id[rid]["label_message_matches"]]
    validation = [by_id[rid] for rid in splits["validation"]]
    test = [by_id[rid] for rid in splits["test"]]
    unlabeled_val = [parse_unlabeled(HERE / "data/Who&When" / r["id"]) for r in validation]
    unlabeled_test = [parse_unlabeled(HERE / "data/Who&When" / r["id"]) for r in test]
    baseline_model = json.loads((HERE / "benchmark_step/step_model.json").read_text())
    candidates = {}
    models = {}
    for multiplier in MULTIPLIERS:
        model = baseline_model if multiplier == 0 else fit(train_external + train * multiplier, l2=0.45)
        models[multiplier] = model
        val_predictions = predict(model, unlabeled_val)
        metrics = score(validation, val_predictions)
        # Primary target is exact-step accuracy; agent is a tie breaker.
        selection = (metrics["Overall"]["step_accuracy"], metrics["Overall"]["agent_accuracy"])
        candidates[multiplier] = {"validation": metrics, "selection": selection}
    chosen = max(MULTIPLIERS, key=lambda n: (*candidates[n]["selection"], -n))
    test_predictions = predict(models[chosen], unlabeled_test)
    baseline_test = predict(baseline_model, unlabeled_test)
    metrics = {"protocol": "Frozen question-group train/validation/test split. 2,004 filtered external traces plus matched-label original training traces. Multiplier chosen on validation; test predictions use unlabeled parsing.",
               "original_train_traces_used": len(train), "validation_traces": len(validation),
               "test_traces": len(test), "multipliers": candidates, "selected_multiplier": chosen,
               "baseline_test": score(test, baseline_test), "candidate_test": score(test, test_predictions),
               "caveat": "The test set has been inspected during prior project development. It is a frozen split but not an untouched final holdout; paper protocols are not matched."}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "domain_adaptation_metrics.json", metrics)
    if chosen:
        write_json(output / "domain_adapted_step_model.json", models[chosen] | {
            "training_source": "Filtered StepFinder plus frozen Who&When training split",
            "selected_multiplier": chosen, "selection_source": "Who&When frozen validation split"})
    write_jsonl(output / "domain_adaptation_test_predictions.jsonl", [
        {"id": r["id"], "subset": r["subset"], "actual_agent": r["culprit_name"],
         "actual_step": r["mistake_step"], "baseline_agent": b["predicted_agent"],
         "baseline_step": b["predicted_step"], "candidate_agent": p["predicted_agent"],
         "candidate_step": p["predicted_step"]}
        for r, b, p in zip(test, baseline_test, test_predictions)])
    lines = ["# Frozen-split domain adaptation experiment", "",
             f"External training: {len(train_external)} filtered traces. Original training: {len(train)} label-consistent traces. Validation: {len(validation)}. Test: {len(test)}.",
             f"Validation selected original-training multiplier **{chosen}** from {MULTIPLIERS}. Selection maximized exact-step accuracy, with agent accuracy as a tie breaker.", "",
             "| Method | Validation agent | Validation step | Test agent | Test step | Test joint |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for multiplier in MULTIPLIERS:
        v = candidates[multiplier]["validation"]["Overall"]
        t = metrics["baseline_test"]["Overall"] if multiplier == 0 else (metrics["candidate_test"]["Overall"] if multiplier == chosen else None)
        suffix = (f"{t['correct_agent']}/{t['traces']} | {t['correct_step']}/{t['traces']} | {t['correct_both']}/{t['traces']}" if t else "not tested | not tested | not tested")
        lines.append(f"| Multiplier {multiplier} | {v['correct_agent']}/{v['traces']} | {v['correct_step']}/{v['traces']} | {suffix} |")
    lines += ["", "The original test cases were inspected during earlier work. This result is exploratory and cannot establish a fair win over published systems."]
    (output / "Domain_Adaptation_Report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stepfinder-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "accuracy_push")
    args = parser.parse_args()
    result = run(args.stepfinder_data, args.output)
    print(json.dumps({"selected": result["selected_multiplier"],
                      "baseline_test": result["baseline_test"],
                      "candidate_test": result["candidate_test"]}, indent=2))


if __name__ == "__main__":
    main()
