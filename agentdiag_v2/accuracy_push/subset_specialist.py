"""Evaluate subset-specific step rankers selected by external question-group CV."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from agentdiag_v2.benchmark_step import exact_metrics
from agentdiag_v2.src.data import load_records, parse_unlabeled, write_json, write_jsonl
from agentdiag_v2.src.step_ranker import fit, predict


HERE = Path(__file__).resolve().parents[1]
SUBSETS = ("Algorithm-Generated", "Hand-Crafted")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run(stepfinder_data: Path, output: Path) -> dict:
    allowed = json.loads((HERE / "data/stepfinder_safe_questions.json").read_text())
    hashes = set(allowed["safe_question_sha256"])
    external, _ = load_records(stepfinder_data)
    train = [r for r in external if "/train/" in r["id"] and r["label_message_matches"]
             and 2 <= len(r["agents"]) <= 4
             and hashlib.sha256(r["question"].strip().encode()).hexdigest() in hashes]
    if len(train) != 2004:
        raise ValueError("External corpus differs from the audited 2,004 records")
    fold_map = json.loads((HERE / "benchmark_step/external_training_folds.json").read_text())["question_to_fold"]
    combined_cv = read_jsonl(HERE / "benchmark_step/external_cv_predictions_l2_0_45.jsonl")
    original, exclusions = load_records(HERE / "data/Who&When")
    if exclusions or len(original) != 184:
        raise ValueError("Original benchmark differs from the audited 184 records")
    saved_model = json.loads((HERE / "benchmark_step/step_model.json").read_text())
    saved_predictions = read_jsonl(HERE / "benchmark_step/full_benchmark_predictions.jsonl")
    by_original = {r["id"]: r for r in original}
    by_combined_cv = {p["id"]: p for p in combined_cv}
    by_saved = {p["id"]: p for p in saved_predictions}
    if set(by_saved) != set(by_original) or len(by_combined_cv) != len(train):
        raise ValueError("Saved predictions do not align with source records")
    choices = {}
    all_test_rows = []
    all_test_predictions = []
    for subset in SUBSETS:
        subset_train = [r for r in train if r["id"].split("/", 1)[0] == subset]
        fold_pairs = []
        for fold in range(5):
            fitting = [r for r in subset_train if fold_map[r["question_id"]] != fold]
            validation = [r for r in subset_train if fold_map[r["question_id"]] == fold]
            if not fitting or not validation:
                raise ValueError(f"Empty group fold for {subset}")
            model = fit(fitting, l2=0.45)
            fold_pairs.extend(zip(validation, predict(model, validation)))
        cv_records = [r for r, _ in fold_pairs]
        specialist_cv = [p for _, p in fold_pairs]
        combined_cv_predictions = [by_combined_cv[r["id"]] for r in cv_records]
        specialist_metric = exact_metrics(cv_records, specialist_cv)[subset]
        combined_metric = exact_metrics(cv_records, combined_cv_predictions)[subset]
        # The subset decision uses external CV only. Prefer the existing model on ties.
        specialist_score = specialist_metric["agent_accuracy"] + specialist_metric["step_accuracy"]
        combined_score = combined_metric["agent_accuracy"] + combined_metric["step_accuracy"]
        use_specialist = specialist_score > combined_score
        candidate_model = fit(subset_train, l2=0.45) if use_specialist else saved_model
        subset_test = [r for r in original if r["subset"] == subset]
        unlabeled = [parse_unlabeled(HERE / "data/Who&When" / r["id"]) for r in subset_test]
        candidate_predictions = predict(candidate_model, unlabeled)
        baseline_predictions = [by_saved[r["id"]] for r in subset_test]
        choices[subset] = {"selected": "specialist" if use_specialist else "combined",
                           "external_cv_combined": combined_metric,
                           "external_cv_specialist": specialist_metric,
                           "original_combined": exact_metrics(subset_test, baseline_predictions)[subset],
                           "original_selected": exact_metrics(subset_test, candidate_predictions)[subset],
                           "external_training_traces": len(subset_train)}
        if use_specialist:
            output.mkdir(parents=True, exist_ok=True)
            write_json(output / f"{subset.lower()}_specialist_model.json", candidate_model | {
                "training_source": "Filtered StepFinder subset", "source_commit": allowed["commit"]})
        all_test_rows.extend(subset_test)
        all_test_predictions.extend(candidate_predictions)
    result = {"protocol": "Five question-group folds of filtered StepFinder data choose combined versus subset-specific ranker separately per subset; original labels are used only for post-selection scoring.",
              "choices": choices,
              "original_selected_overall": exact_metrics(all_test_rows, all_test_predictions)["Overall"],
              "caveat": "Who&When cases were previously inspected; original-corpus scores remain exploratory and are not directly comparable to paper numbers."}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "subset_specialist_metrics.json", result)
    write_jsonl(output / "subset_specialist_predictions.jsonl", [
        {"id": r["id"], "subset": r["subset"], "actual_agent": r["culprit_name"],
         "actual_step": r["mistake_step"], "predicted_agent": p["predicted_agent"],
         "predicted_step": p["predicted_step"]}
        for r, p in zip(all_test_rows, all_test_predictions)])
    lines = ["# Subset-specific ranker experiment", "",
             "The model choice for each subset was made using the filtered external StepFinder question-group folds. The original Who&When labels were scored afterward.", "",
             "| Subset | External CV combined agent/step | External CV specialist agent/step | Chosen model | Original combined agent/step | Original chosen agent/step |",
             "| --- | ---: | ---: | --- | ---: | ---: |"]
    for subset in SUBSETS:
        c = choices[subset]
        def fmt(m: dict) -> str:
            return f"{m['correct_agent']}/{m['traces']} / {m['correct_step']}/{m['traces']}"
        lines.append(f"| {subset} | {fmt(c['external_cv_combined'])} | {fmt(c['external_cv_specialist'])} | {c['selected']} | {fmt(c['original_combined'])} | {fmt(c['original_selected'])} |")
    lines += ["", "The original corpus was previously inspected; these are exploratory comparisons, not a paper-matched benchmark win."]
    (output / "Subset_Specialist_Report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stepfinder-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "accuracy_push")
    args = parser.parse_args()
    result = run(args.stepfinder_data, args.output)
    print(json.dumps(result["choices"], indent=2))


if __name__ == "__main__":
    main()
