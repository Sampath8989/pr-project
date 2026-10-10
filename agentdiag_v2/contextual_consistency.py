"""Select soft agent/step agreement using external predictions only."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .contextual_upgrade import HERE, OUTPUT, external_data, read_cv, serialize
from .benchmark_step import exact_metrics
from .src.data import load_records, write_json, write_jsonl


def reconcile(record, prediction, agent_weight, step_weight):
    step = np.asarray(prediction["step_probabilities"], dtype=float)
    author = np.asarray([record["agents"].index(m["agent"]) for m in record["messages"]])
    marginal = np.bincount(author, weights=step, minlength=len(record["agents"]))
    agent = (1 - agent_weight) * np.asarray(prediction["agent_probabilities"]) + agent_weight * marginal
    score = np.log(step.clip(1e-12)) + step_weight * np.log(agent[author].clip(1e-12))
    adjusted = np.exp(score - score.max())
    adjusted /= adjusted.sum()
    return {"predicted_agent": record["agents"][int(agent.argmax())],
            "predicted_step": record["messages"][int(adjusted.argmax())]["raw_index"],
            "agent_probabilities": agent.tolist(), "step_probabilities": adjusted.tolist()}


def run(data: Path, output: Path):
    rows, _, _ = external_data(data)
    cv = read_cv(OUTPUT / "cv_selected.jsonl", rows)
    candidates, best = {}, None
    for a in (0, 0.25, 0.5, 0.75, 1):
        for s in (0, 0.25, 0.5, 1):
            predictions = [reconcile(r, p, a, s) for r, p in zip(rows, cv)]
            metrics = exact_metrics(rows, predictions)
            alg = metrics["Algorithm-Generated"]
            score = ((alg["agent_accuracy"] + alg["step_accuracy"]) / 2,
                     alg["joint_accuracy"], -a - s)
            candidates[f"{a}:{s}"] = metrics
            if best is None or score > best[0]:
                best = (score, a, s)
    selection = {"agent_weight": best[1], "step_weight": best[2],
                 "external_cv_metrics": candidates[f"{best[1]}:{best[2]}"],
                 "cv_sha256": hashlib.sha256((OUTPUT / "cv_selected.jsonl").read_bytes()).hexdigest(),
                 "protocol": "Weights selected only on external Algorithm-Generated OOF accuracy; exact joint is a tie breaker. Original predictions are scored only after selection."}
    output.mkdir(exist_ok=True)
    write_json(output / "consistency_selection.json", selection)
    write_json(output / "consistency_cv_metrics.json", candidates)
    original, _ = load_records(HERE / "data/Who&When")
    saved = read_cv(OUTPUT / "benchmark_predictions.jsonl", original)
    adjusted = [reconcile(r, p, best[1], best[2]) for r, p in zip(original, saved)]
    metrics = {"before": exact_metrics(original, saved), "after": exact_metrics(original, adjusted),
               "selection": selection,
               "limitation": "Previously inspected corpus; exploratory. No original labels choose weights."}
    write_json(output / "consistency_metrics.json", metrics)
    write_jsonl(output / "consistency_predictions.jsonl", serialize(original, adjusted))
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stepfinder-data", type=Path, default=HERE.parent.parent / "StepFinder/data")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.stepfinder_data, args.output)


if __name__ == "__main__":
    main()
