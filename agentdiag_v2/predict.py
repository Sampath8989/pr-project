#!/usr/bin/env python3
"""Diagnose one new multi-agent trace with saved AgentDiag v2 artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentdiag_v2.src.bayes import predict
from agentdiag_v2.src.data import parse_unlabeled
from agentdiag_v2.src.evaluation import calibrate_temperature, normalized
from agentdiag_v2.src.temporal import forward
from agentdiag_v2.src.triage import packet


HERE = Path(__file__).resolve().parent


def diagnose(trace_path: Path, artifacts: Path, masked_agents: list[int] | None = None) -> dict:
    trace = parse_unlabeled(trace_path)
    model_data = json.loads((artifacts / "phase3_bayesian_inference" / "model.json").read_text())
    calibration = json.loads((artifacts / "phase5_calibration_evaluation" / "calibration.json").read_text())
    hmm_data = json.loads((artifacts / "phase6_temporal_hmm" / "hmm_model.json").read_text())
    n = trace["n_agents"]
    if str(n) not in model_data["models"]:
        raise ValueError(f"No trained model for {n} agents; supported counts: {list(model_data['models'])}")
    evidence = [dict(e) for e in trace["evidence"]]
    for index in masked_agents or []:
        if not 0 <= index < n:
            raise ValueError(f"masked agent index {index} outside 0..{n-1}")
        evidence[index] = None
    model = model_data["models"][str(n)]
    probabilities = calibrate_temperature(predict(model, evidence), calibration["temperature"])
    if not normalized(probabilities):
        raise AssertionError("diagnosis probabilities invalid")
    winner = max(range(n), key=lambda i: probabilities[i])
    effects = []
    for index in range(n):
        if index in (masked_agents or []):
            effects.append(None)
            continue
        ablated = list(evidence)
        ablated[index] = None
        without = calibrate_temperature(predict(model, ablated), calibration["temperature"])
        effects.append(probabilities[index] - without[index])
    hmm = hmm_data["model"]
    timeline = []
    for agent_index, agent in enumerate(trace["agents"]):
        if agent_index in (masked_agents or []):
            continue
        messages = [m for m in trace["messages"] if m["agent"] == agent]
        beliefs = forward(hmm, [m["flags"] for m in messages])
        timeline.extend({"agent": agent, "raw_index": m["raw_index"],
                         "danger_probability": belief[1] + belief[2]}
                        for m, belief in zip(messages, beliefs))
    timeline.sort(key=lambda item: item["raw_index"])
    return {"trace": str(trace_path), "agents": trace["agents"],
            "root_cause_probabilities": {agent: probabilities[i] for i, agent in enumerate(trace["agents"])},
            "predicted_culprit": trace["agents"][winner], "confidence": probabilities[winner],
            "masked_agent_indices": masked_agents or [], "model_training_count": model["training_count"],
            "model_features": model["features"], "reliability_timeline": timeline,
            "reliability_warning": "The temporal monitor gave many wrong-agent warnings on the held-out test; use this timeline only for inspection.",
            "review_packet": packet(trace, probabilities, model["features"], masked_agents, effects)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--artifacts", type=Path, default=HERE / "artifacts")
    parser.add_argument("--mask-agent-index", type=int, action="append", default=[])
    args = parser.parse_args()
    print(json.dumps(diagnose(args.trace, args.artifacts, args.mask_agent_index), indent=2))


if __name__ == "__main__":
    main()
