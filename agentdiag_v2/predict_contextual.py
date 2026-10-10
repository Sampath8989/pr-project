"""Predict an agent and raw step from an unlabeled saved trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contextual_upgrade import OUTPUT, DEFAULT_CACHE, predict_selected
from .src.data import parse_unlabeled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--model-dir", type=Path, default=OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()
    record = parse_unlabeled(args.trace)
    output = predict_selected([record], args.model_dir, args.cache)[0]
    output["agents"] = record["agents"]
    output["raw_message_indices"] = [m["raw_index"] for m in record["messages"]]
    output["probabilities_are_calibrated"] = False
    output["task_question_available"] = bool(record["question"].strip())
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
