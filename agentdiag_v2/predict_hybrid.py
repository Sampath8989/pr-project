#!/usr/bin/env python3
"""Predict a four-agent culprit with the evaluated BN and step-model hybrid."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentdiag_v2.src.bayes import predict as bn_predict
from agentdiag_v2.src.data import parse_unlabeled
from agentdiag_v2.src.step_ranker import predict as step_predict


HERE = Path(__file__).resolve().parent


def diagnose(trace_path: Path, hybrid_path: Path = HERE / "hybrid_upgrade" / "hybrid_model.json") -> dict:
    trace = parse_unlabeled(trace_path)
    if len(trace["agents"]) != 4:
        raise ValueError("The evaluated hybrid model supports exactly four agents")
    saved = json.loads(hybrid_path.read_text())
    step_path = (hybrid_path.parent / saved["step_model_path"]).resolve()
    step_model = json.loads(step_path.read_text())
    bn_probabilities = bn_predict(saved["bn"], trace["evidence"])
    step_result = step_predict(step_model, [trace])[0]
    weight = saved["step_weight"]
    probabilities = [(1 - weight) * a + weight * b
                     for a, b in zip(bn_probabilities, step_result["agent_probabilities"])]
    winner = max(range(4), key=lambda index: probabilities[index])
    if abs(sum(probabilities) - 1) > 1e-9:
        raise AssertionError("Hybrid probabilities must sum to one")
    return {"trace": str(trace_path), "agents": trace["agents"],
            "predicted_culprit": trace["agents"][winner],
            "agent_probabilities": dict(zip(trace["agents"], probabilities)),
            "predicted_step_from_step_model": step_result["predicted_step"],
            "step_model_agent": step_result["predicted_agent"],
            "step_model_weight": weight,
            "scope": "Four-agent post-mortem attribution; results are exploratory and do not beat AgenTracer."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--model", type=Path, default=HERE / "hybrid_upgrade" / "hybrid_model.json")
    args = parser.parse_args()
    print(json.dumps(diagnose(args.trace, args.model), indent=2))


if __name__ == "__main__":
    main()
