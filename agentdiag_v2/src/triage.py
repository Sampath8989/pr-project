"""Grounded review packets for uncertain agent attribution.

Observed excerpts are trace references, not claims that a message caused the
failure. The source indices let a reviewer verify each excerpt in the input.
"""

from __future__ import annotations

import re

from agentdiag_v2.src.data import CORRECT, ERROR, UNCERTAIN


PATTERN = {"error": ERROR, "uncertain": UNCERTAIN, "correction": CORRECT}


def _instruction_like(content: str) -> bool:
    return content.lstrip().lower().startswith("you are given:")


def _excerpt(content: str, feature: str, limit: int = 180) -> str:
    single_line = re.sub(r"\s+", " ", content).strip()
    match = PATTERN[feature].search(single_line)
    if match is None:
        raise AssertionError("selected flag has no matching phrase")
    start = max(0, match.start() - 65)
    end = min(len(single_line), start + limit)
    return ("…" if start else "") + single_line[start:end] + ("…" if end < len(single_line) else "")


def observations(trace: dict, agent: str, model_features: list[str], masked: bool = False) -> list[dict]:
    if masked:
        return [{"kind": "missing_evidence", "detail": "This agent's messages were masked."}]
    authored = [m for m in trace["messages"] if m["agent"] == agent]
    found = []
    for feature in model_features:
        if feature not in {"error", "uncertain", "correction"}:
            continue
        matching = [m for m in authored if m["flags"][feature]]
        matching.sort(key=lambda m: (_instruction_like(m["content"]), m["raw_index"]))
        for message in matching[:1]:
            found.append({"kind": "model_feature", "feature": feature,
                          "source_kind": "task_instruction" if _instruction_like(message["content"])
                          else "agent_message",
                          "raw_message_index": message["raw_index"],
                          "excerpt": _excerpt(message["content"], feature)})
    if not found:
        found.append({"kind": "model_feature", "detail": "No selected keyword flag was observed in this agent's messages."})
    return found


def packet(trace: dict, probabilities: list[float], model_features: list[str],
           masked_agents: list[int] | None = None,
           evidence_effects: list[float | None] | None = None) -> dict:
    if len(probabilities) != len(trace["agents"]):
        raise ValueError("one probability is required per agent")
    masked = set(masked_agents or [])
    order = sorted(range(len(probabilities)), key=lambda i: -probabilities[i])
    candidates = []
    for rank, i in enumerate(order, start=1):
        candidates.append({"rank": rank, "agent": trace["agents"][i],
                           "probability": probabilities[i],
                           "observed_evidence_probability_effect":
                           evidence_effects[i] if evidence_effects is not None else None,
                           "observations": observations(trace, trace["agents"][i],
                                                         model_features, i in masked)})
    return {"review_policy": "Inspect the first two candidates and the source trace; do not treat the top candidate as a verified cause.",
            "top_two_agents": [trace["agents"][i] for i in order[:2]],
            "top_two_probability_mass": sum(probabilities[i] for i in order[:2]),
            "candidates": candidates,
            "evidence_note": "Excerpts show observed keyword evidence, including task instructions when those trigger a flag. The probability effect is full-evidence probability minus probability after masking that agent's evidence. This is model sensitivity, not a causal intervention."}
