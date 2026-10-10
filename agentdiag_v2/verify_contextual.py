"""Recompute saved contextual-model metrics and confirm reproducible predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from .contextual_upgrade import HERE, OUTPUT, DEFAULT_CACHE, predict_selected, read_cv
from .benchmark_step import exact_metrics
from .src.data import load_records, parse_unlabeled, write_json


def verify(output: Path, cache: Path):
    rows, excluded = load_records(HERE / "data/Who&When")
    assert len(rows) == 184 and not excluded
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["selection_sha256"] == hashlib.sha256((output / "selection.json").read_bytes()).hexdigest()
    saved = read_cv(output / "benchmark_predictions.jsonl", rows)
    recalculated = exact_metrics(rows, saved)
    for subset, group in recalculated.items():
        for key, value in group.items():
            assert metrics["new"][subset][key] == value, (subset, key)
    for row, pred in zip(rows, saved):
        assert pred["predicted_agent"] in row["agents"]
        assert pred["predicted_step"] in [m["raw_index"] for m in row["messages"]]
        for head, length in (("step", len(row["messages"])), ("agent", len(row["agents"]))):
            p = np.asarray(pred[head + "_probabilities"])
            assert p.shape == (length,) and np.isfinite(p).all()
            assert (p >= 0).all() and np.isclose(p.sum(), 1, atol=1e-6)
    parsed = [parse_unlabeled(HERE / "data/Who&When" / r["id"]) for r in rows]
    fresh = predict_selected(parsed, output, cache)
    for original, restored in zip(saved, fresh):
        assert original["predicted_agent"] == restored["predicted_agent"]
        assert original["predicted_step"] == restored["predicted_step"]
        assert np.allclose(original["agent_probabilities"], restored["agent_probabilities"], atol=1e-6)
        assert np.allclose(original["step_probabilities"], restored["step_probabilities"], atol=1e-6)
    paired = {}
    for subset, heads in metrics["paired_changes"].items():
        paired[subset] = {}
        for head, counts in heads.items():
            total = counts["gains"] + counts["losses"]
            paired[subset][head] = {**counts, "two_sided_exact_p": float(
                binomtest(counts["gains"], total, p=0.5).pvalue) if total else 1.0}
    result = {"passed": True, "verified_traces": len(rows), "checks": [
        "selection file unchanged since benchmark prediction", "strict metrics recomputed",
        "valid raw indices and agents", "finite normalized probabilities",
        "saved checkpoints reproduce all 184 label-free predictions"], "paired_changes": paired}
    write_json(output / "verification.json", result)
    lines = ["# Contextual attribution experiment", "",
             "The configuration and blend were selected on external question-group validation. "
             "The final training used 2,004 filtered external traces from 58 questions; the original Who&When labels were not training targets.", "",
             "| Method | Algorithm agent | Algorithm exact step | Hand-crafted agent | Hand-crafted exact step |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for label, values in (("Previous local ranker", metrics["old"]), ("Selected contextual candidate", metrics["new"])):
        a, h = values["Algorithm-Generated"], values["Hand-Crafted"]
        lines.append(f"| {label} | {a['correct_agent']}/126 ({a['agent_accuracy']:.1%}) | "
                     f"{a['correct_step']}/126 ({a['step_accuracy']:.1%}) | "
                     f"{h['correct_agent']}/58 ({h['agent_accuracy']:.1%}) | "
                     f"{h['correct_step']}/58 ({h['step_accuracy']:.1%}) |")
    lines += ["| Who&When GPT-4o all-at-once, no answer supplied (paper) | 51.12% | 13.53% | 53.44% | 3.51% |",
              "| AgenTracer, no answer supplied (paper) | 63.73% | 37.30% | 63.82% | 20.68% |", "",
              "Paper sources: [Who&When](https://arxiv.org/html/2505.00212), "
              "[AgenTracer](https://arxiv.org/html/2509.03312). These are reported paper scores, not a matched head-to-head comparison.", "",
              "## Frozen selection", "", "```json", (output / "selection.json").read_text(), "```", "",
              "## Paired changes", "", "```json", json.dumps(paired, indent=2), "```", "",
              "## What changed", "",
              "The lexical comparator includes word pairs, preceding and subsequent messages, and an agent classifier trained directly on agent labels. "
              "The semantic candidates use a pinned Qwen3-Embedding-0.6B encoder, question interactions, contextual attention, "
              "optional bidirectional GRU, and separately supervised agent/step heads. Selection may retain only some components; "
              "the selection manifest specifies the model actually used.", "",
              "## Verification and limits", "",
              "All 184 checkpoint predictions were reproduced from label-free parsing. Exact metric counts and probability validity were checked. "
              "The model analyzes saved text logs; it does not run live agents, counterfactual replay or multimodal inputs. "
              "The original benchmark has been inspected during development. These exploratory results cannot establish independent benchmark leadership. "
              "Matching reported percentages does not establish statistical equivalence or identical scoring. "
              "The Bayesian network remains a separate probabilistic reasoning component, not the model responsible for these scores."]
    (output / "Contextual_Report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()
    print(json.dumps(verify(args.output, args.cache), indent=2))


if __name__ == "__main__":
    main()
