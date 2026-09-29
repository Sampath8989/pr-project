#!/usr/bin/env python3
"""Predict the responsible step and agent using the saved external-data model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentdiag_v2.src.data import parse_unlabeled
from agentdiag_v2.src.step_ranker import predict


HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--model", type=Path, default=HERE / "benchmark_step" / "step_model.json")
    args = parser.parse_args()
    model = json.loads(args.model.read_text())
    trace = parse_unlabeled(args.trace)
    result = predict(model, [trace])[0]
    result["trace"] = str(args.trace)
    result["agents"] = trace["agents"]
    result["step_indices"] = [m["raw_index"] for m in trace["messages"]]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
