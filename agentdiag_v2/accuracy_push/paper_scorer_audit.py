"""Quantify the difference between strict matching and the released Who&When scorer."""

from __future__ import annotations

import json
from pathlib import Path

from agentdiag_v2.src.data import write_json


HERE = Path(__file__).resolve().parents[1]


def run() -> dict:
    rows = [json.loads(line) for line in (HERE / "benchmark_step/full_benchmark_predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    result = {}
    for subset in ("Algorithm-Generated", "Hand-Crafted"):
        group = [r for r in rows if r["subset"] == subset]
        strict_agent = sum(r["actual_agent"] == r["predicted_agent"] for r in group)
        strict_step = sum(r["actual_step"] == r["predicted_step"] for r in group)
        paper_agent = sum(str(r["actual_agent"]) in str(r["predicted_agent"]) for r in group)
        paper_step = sum(str(r["actual_step"]) in str(r["predicted_step"]) for r in group)
        result[subset] = {"n": len(group), "strict_agent": strict_agent, "strict_step": strict_step,
                          "released_scorer_agent": paper_agent, "released_scorer_step": paper_step,
                          "extra_step_ids": [r["id"] for r in group if r["actual_step"] != r["predicted_step"]
                                             and str(r["actual_step"]) in str(r["predicted_step"])]}
    output = HERE / "accuracy_push"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "paper_scorer_audit.json", result)
    lines = ["# Released Who&When scorer audit", "",
             "The released `Automated_FA/evaluate.py` counts an agent as correct when the actual agent name is a substring of the predicted string and counts a step when the actual step number is a substring of the predicted string. We applied those two comparisons to our saved step-model outputs.", "",
             "| Subset | Strict agent | Released-style agent | Strict raw step | Released-style step |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for subset, r in result.items():
        lines.append(f"| {subset} | {r['strict_agent']}/{r['n']} | {r['released_scorer_agent']}/{r['n']} | {r['strict_step']}/{r['n']} | {r['released_scorer_step']}/{r['n']} |")
    lines += ["", "This reproduces only the released scorer's final string comparisons on our saved predictions. It does not reproduce the paper's prompt, model run, all filtering decisions, or statistical protocol. Strict exact matching remains the project's main reported metric."]
    (output / "Paper_Scorer_Audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
