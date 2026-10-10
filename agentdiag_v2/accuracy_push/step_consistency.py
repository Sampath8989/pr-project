"""Evaluate agent-aware step selection using saved external grouped predictions."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from scipy.stats import binomtest

from agentdiag_v2.src.data import load_records, write_json, write_jsonl


HERE = Path(__file__).resolve().parents[1]
ALPHAS = (0.0, 0.5, 1.0, 2.0, 4.0)
SUBSETS = ("Algorithm-Generated", "Hand-Crafted")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def chosen_step(record: dict, prediction: dict, alpha: float) -> int:
    step_probs = prediction["step_probabilities"]
    agent_probs = prediction["agent_probabilities"]
    if len(step_probs) != len(record["messages"]) or len(agent_probs) != len(record["agents"]):
        raise ValueError(f"Prediction shape does not match trace {record['id']}")
    scores = [math.log(max(p, 1e-15)) + alpha * math.log(max(agent_probs[record["agents"].index(m["agent"])], 1e-15))
              for m, p in zip(record["messages"], step_probs)]
    best = max(range(len(scores)), key=scores.__getitem__)
    return record["messages"][best]["raw_index"]


def evaluate(rows: list[dict], by_id: dict[str, dict], alpha: float) -> dict:
    result = {}
    for subset in SUBSETS:
        group = [p for p in rows if p.get("subset", p.get("benchmark_subset")) == subset]
        correct_agent = sum(p["predicted_agent"] == p["actual_agent"] for p in group)
        correct_step = sum(chosen_step(by_id[p["id"]], p, alpha) == p["actual_step"] for p in group)
        correct_both = sum(p["predicted_agent"] == p["actual_agent"]
                           and chosen_step(by_id[p["id"]], p, alpha) == p["actual_step"] for p in group)
        result[subset] = {"n": len(group), "agent": correct_agent, "step": correct_step,
                          "joint": correct_both, "step_accuracy": correct_step / len(group)}
    return result


def run(stepfinder_data: Path, output: Path) -> dict:
    external, _ = load_records(stepfinder_data)
    original, exclusions = load_records(HERE / "data" / "Who&When")
    if exclusions or len(original) != 184:
        raise ValueError("Original benchmark data differs from the audited 184 traces")
    external_by_id = {r["id"]: r for r in external}
    original_by_id = {r["id"]: r for r in original}
    cv = read_jsonl(HERE / "benchmark_step" / "external_cv_predictions_l2_0_45.jsonl")
    test = read_jsonl(HERE / "benchmark_step" / "full_benchmark_predictions.jsonl")
    if len(cv) != 2004 or len(test) != 184 or len({p["question_id"] for p in cv}) != 58:
        raise ValueError("Saved predictions differ from the audited experiment")
    if any(p["id"] not in external_by_id for p in cv) or any(p["id"] not in original_by_id for p in test):
        raise ValueError("Predictions do not match parsed traces")
    if any(chosen_step(external_by_id[p["id"]], p, 0) != p["predicted_step"] for p in cv):
        raise ValueError("External baseline predictions cannot be reproduced")
    if any(chosen_step(original_by_id[p["id"]], p, 0) != p["predicted_step"] for p in test):
        raise ValueError("Original baseline predictions cannot be reproduced")
    cv_scores = {str(alpha): evaluate(cv, external_by_id, alpha) for alpha in ALPHAS}
    # Only external grouped out-of-fold rows choose the setting. The inspected benchmark is scored afterward.
    selected = max(ALPHAS, key=lambda a: (sum(cv_scores[str(a)][s]["step_accuracy"] for s in SUBSETS) / 2, -a))
    baseline = evaluate(test, original_by_id, 0)
    candidate = evaluate(test, original_by_id, selected)
    categories = {}
    for subset in SUBSETS:
        group = [p for p in test if p["subset"] == subset]
        mismatch = sum(not original_by_id[p["id"]]["label_message_matches"] for p in group)
        def band(prediction: dict) -> str:
            length = len(original_by_id[prediction["id"]]["messages"])
            return "1-5" if length <= 5 else "6-10" if length <= 10 else "11+"
        categories[subset] = {
            "n": len(group),
            "label_step_speaker_mismatch": mismatch,
            "agent_wrong_step_wrong": sum(p["predicted_agent"] != p["actual_agent"] and p["predicted_step"] != p["actual_step"] for p in group),
            "agent_right_step_wrong": sum(p["predicted_agent"] == p["actual_agent"] and p["predicted_step"] != p["actual_step"] for p in group),
            "agent_wrong_step_right": sum(p["predicted_agent"] != p["actual_agent"] and p["predicted_step"] == p["actual_step"] for p in group),
            "agent_right_step_right": sum(p["predicted_agent"] == p["actual_agent"] and p["predicted_step"] == p["actual_step"] for p in group),
            "trace_length_bands": dict(Counter(band(p) for p in group)),
            "step_errors_by_length": dict(Counter(band(p) for p in group if p["predicted_step"] != p["actual_step"])),
            "agent_errors_by_length": dict(Counter(band(p) for p in group if p["predicted_agent"] != p["actual_agent"])),
            "example_agent_right_step_wrong": [p["id"] for p in group
                                                if p["predicted_agent"] == p["actual_agent"]
                                                and p["predicted_step"] != p["actual_step"]][:3],
        }
    pairs = [(p["predicted_step"] == p["actual_step"],
              chosen_step(original_by_id[p["id"]], p, selected) == p["actual_step"]) for p in test]
    gain = sum(not old and new for old, new in pairs)
    loss = sum(old and not new for old, new in pairs)
    paired_p = binomtest(min(gain, loss), gain + loss, 0.5).pvalue if gain + loss else 1.0
    result = {"selected_alpha_external_cv": selected, "external_cv_scores": cv_scores,
              "original_baseline": baseline, "original_candidate": candidate,
              "error_audit": categories, "paired_step_gain": gain, "paired_step_loss": loss,
              "paired_step_exact_p_value": paired_p,
              "interpretation": "Alpha was selected using saved external question-group out-of-fold predictions. The original Who&When test was already inspected during project development; this result is exploratory."}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "step_consistency_metrics.json", result)
    write_jsonl(output / "step_consistency_predictions.jsonl", [
        {"id": p["id"], "subset": p["subset"], "actual_agent": p["actual_agent"],
         "actual_step": p["actual_step"], "predicted_agent": p["predicted_agent"],
         "baseline_step": p["predicted_step"], "candidate_step": chosen_step(original_by_id[p["id"]], p, selected)}
        for p in test])
    report = ["# Agent-aware step selection experiment", "",
              "The weakest reported area is exact step attribution. This experiment asks whether a step is easier to locate when its author's agent probability is included in the score.",
              "The setting was chosen on saved StepFinder question-group validation predictions. No Who&When labels selected it.", "",
              f"Selected agent weight: **{selected:g}**. Exact paired test on the 184 inspected traces: {gain} gains, {loss} losses, p={paired_p:.3f}.", "",
              "| Subset | Agent (unchanged) | Original exact step | Candidate exact step | Original joint | Candidate joint |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for subset in SUBSETS:
        b, c = baseline[subset], candidate[subset]
        report.append(f"| {subset} | {b['agent']}/{b['n']} | {b['step']}/{b['n']} | {c['step']}/{c['n']} | {b['joint']}/{b['n']} | {c['joint']}/{c['n']} |")
    report += ["", "The saved benchmark was inspected before this experiment, so this table does not establish an independent paper-matched win.", "",
               "## Error categories", ""]
    for subset in SUBSETS:
        c = categories[subset]
        report.append(f"- {subset}: {c['agent_wrong_step_wrong']} both wrong; {c['agent_right_step_wrong']} agent right/step wrong; {c['agent_wrong_step_right']} agent wrong/step right; {c['agent_right_step_right']} both right. {c['label_step_speaker_mismatch']} labels name a different speaker at the labeled step.")
        report.append(f"  Step errors by trace length: {c['step_errors_by_length']}. Examples where the agent was correct but the step was not: {', '.join(c['example_agent_right_step_wrong'])}.")
    (output / "Step_Consistency_Report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stepfinder-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "accuracy_push")
    args = parser.parse_args()
    result = run(args.stepfinder_data, args.output)
    print(json.dumps({"alpha": result["selected_alpha_external_cv"],
                      "baseline": result["original_baseline"],
                      "candidate": result["original_candidate"]}, indent=2))


if __name__ == "__main__":
    main()
