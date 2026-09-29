#!/usr/bin/env python3
"""Recompute saved step and hybrid scores and check attribution provenance."""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentdiag_v2.src.data import load_records, write_json


HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "Who&When"


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    records, exclusions = load_records(DATA)
    if exclusions:
        raise RuntimeError("Unexpected source exclusions")
    by_id = {r["id"]: r for r in records}
    checks = {}
    step_rows = jsonl(HERE / "benchmark_step" / "full_benchmark_predictions.jsonl")
    step_metrics = json.loads((HERE / "benchmark_step" / "metrics.json").read_text())["metrics"]
    checks["step_predictions_cover_all_source_traces"] = (
        len(step_rows) == len(records) and {r["id"] for r in step_rows} == set(by_id))
    for subset, metric in step_metrics.items():
        group = step_rows if subset == "Overall" else [r for r in step_rows if r["subset"] == subset]
        checks[f"{subset}_scores_recomputed"] = (
            len(group) == metric["traces"]
            and sum(r["predicted_agent"] == r["actual_agent"] for r in group) == metric["correct_agent"]
            and sum(r["predicted_step"] == r["actual_step"] for r in group) == metric["correct_step"])
    checks["step_probability_vectors_valid"] = all(
        len(row["step_probabilities"]) == len(by_id[row["id"]]["messages"])
        and abs(sum(row["step_probabilities"]) - 1) < 1e-9
        and abs(sum(row["agent_probabilities"]) - 1) < 1e-9
        for row in step_rows)
    hybrid = json.loads((HERE / "hybrid_upgrade" / "metrics.json").read_text())
    outer = jsonl(HERE / "hybrid_upgrade" / "nested_fold_predictions.jsonl")
    test = jsonl(HERE / "hybrid_upgrade" / "test_predictions.jsonl")
    checks["nested_predictions_cover_all_four_agent_traces"] = (
        len(outer) == 91 and len({r["id"] for r in outer}) == 91
        and {r["id"] for r in outer} == {r["id"] for r in records if len(r["agents"]) == 4})
    question_fold = {}
    safe = True
    for row in outer:
        qid = by_id[row["id"]]["question_id"]
        if qid in question_fold and question_fold[qid] != row["fold"]:
            safe = False
        question_fold[qid] = row["fold"]
    checks["question_groups_stay_in_one_outer_fold"] = safe
    splits = json.loads((HERE / "artifacts" / "phase0_protocol" / "splits.json").read_text())["splits"]
    checks["hybrid_test_rows_match_frozen_split"] = (
        {r["id"] for r in test} == {i for i in splits["test"] if len(by_id[i]["agents"]) == 4})
    for name, rows, expected in (("nested", outer, hybrid["nested_group_cv"]),
                                 ("test", test, hybrid["original_test_after_refit"])):
        for method in ("bn", "hybrid"):
            correct = sum(max(range(4), key=lambda i: row[method][i]) == row["actual"] for row in rows)
            checks[f"{name}_{method}_accuracy_recomputed"] = correct == expected[method]["correct"]
    checks["hybrid_probabilities_valid"] = all(
        abs(sum(row["hybrid"]) - 1) < 1e-9 and all(0 <= p <= 1 for p in row["hybrid"])
        for row in outer + test)
    result = {"passed": all(checks.values()), "checks": checks}
    write_json(HERE / "hybrid_upgrade" / "verification.json", result)
    (HERE / "hybrid_upgrade" / "Verification_Report.md").write_text(
        "# Upgrade verification\n\n" + "\n".join(
            f"- {'PASS' if good else 'FAIL'} — {name.replace('_', ' ')}" for name, good in checks.items())
        + f"\n\nOverall: {'PASS' if result['passed'] else 'FAIL'}.\n", encoding="utf-8")
    print("PASS" if result["passed"] else "FAIL", len(checks), "checks")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
