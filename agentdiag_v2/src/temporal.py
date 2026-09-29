"""Small supervised-proxy HMM for turn-level reliability monitoring."""

from __future__ import annotations

import itertools

from .data import FEATURES


STATES = ("Healthy", "Slipping", "Failed")


def _agent_sequences(record: dict):
    for agent in record["agents"]:
        yield agent, [m for m in record["messages"] if m["agent"] == agent]


def _proxy_states(record: dict, agent: str, messages: list[dict]) -> list[int]:
    if agent != record["culprit_name"]:
        return [0] * len(messages)
    earlier = [i for i, m in enumerate(messages) if m["raw_index"] < record["mistake_step"]]
    last_earlier = earlier[-1] if earlier else None
    return [2 if m["raw_index"] >= record["mistake_step"] else
            1 if i == last_earlier else 0 for i, m in enumerate(messages)]


def fit_hmm(records: list[dict], alpha: float = 2.0) -> dict:
    starts = [0.0] * 3
    transitions = [[0.0] * 3 for _ in range(3)]
    feature_counts = {f: [[0.0, 0.0] for _ in range(3)] for f in FEATURES}
    state_counts = [0] * 3
    for record in records:
        for agent, messages in _agent_sequences(record):
            if not messages:
                continue
            states = _proxy_states(record, agent, messages)
            starts[states[0]] += 1
            for i, (state, message) in enumerate(zip(states, messages)):
                state_counts[state] += 1
                for f in FEATURES:
                    feature_counts[f][state][message["flags"][f]] += 1
                if i:
                    transitions[states[i - 1]][state] += 1
    initial = [(starts[s] + alpha) / (sum(starts) + 3 * alpha) for s in range(3)]
    matrix = [[(transitions[s][t] + alpha) / (sum(transitions[s]) + 3 * alpha)
               for t in range(3)] for s in range(3)]
    emission = {f: [(feature_counts[f][s][1] + alpha) / (sum(feature_counts[f][s]) + 2 * alpha)
                    for s in range(3)] for f in FEATURES}
    return {"states": list(STATES), "features": list(FEATURES), "alpha": alpha,
            "initial": initial, "transition": matrix, "emission_p_true": emission,
            "state_training_counts": state_counts,
            "note": "States are training proxies derived from annotated mistake steps, not observed ground-truth health states."}


def emission_likelihood(model: dict, state: int, observation: dict) -> float:
    probability = 1.0
    for feature in model["features"]:
        p = model["emission_p_true"][feature][state]
        probability *= p if observation[feature] else 1 - p
    return probability


def forward(model: dict, observations: list[dict]) -> list[list[float]]:
    beliefs = []
    previous = None
    for obs in observations:
        predicted = (model["initial"][:] if previous is None else
                     [sum(previous[s] * model["transition"][s][t] for s in range(3))
                      for t in range(3)])
        current = [predicted[s] * emission_likelihood(model, s, obs) for s in range(3)]
        normalizer = sum(current)
        if normalizer <= 0:
            raise ValueError("zero HMM observation likelihood")
        previous = [v / normalizer for v in current]
        beliefs.append(previous)
    return beliefs


def brute_force_forward(model: dict, observations: list[dict]) -> list[float]:
    if not observations:
        return model["initial"]
    masses = [0.0] * 3
    for states in itertools.product(range(3), repeat=len(observations)):
        weight = model["initial"][states[0]]
        for t, obs in enumerate(observations):
            if t:
                weight *= model["transition"][states[t - 1]][states[t]]
            weight *= emission_likelihood(model, states[t], obs)
        masses[states[-1]] += weight
    total = sum(masses)
    return [m / total for m in masses]


def monitor(model: dict, record: dict, threshold: float) -> dict:
    warnings = {}
    timeline = []
    for agent, messages in _agent_sequences(record):
        beliefs = forward(model, [m["flags"] for m in messages])
        for message, belief in zip(messages, beliefs):
            danger = belief[1] + belief[2]
            if danger >= threshold and agent not in warnings:
                warnings[agent] = message["raw_index"]
            timeline.append({"agent": agent, "raw_index": message["raw_index"],
                             "danger": danger})
    timeline.sort(key=lambda x: x["raw_index"])
    culprit_warning = warnings.get(record["culprit_name"])
    eligible = any(m["raw_index"] < record["mistake_step"] for m in record["messages"])
    return {"id": record["id"], "actual_culprit": record["culprit_name"],
            "mistake_step": record["mistake_step"], "eligible": eligible,
            "warnings": warnings, "culprit_warning": culprit_warning,
            "correct_early": culprit_warning is not None and culprit_warning < record["mistake_step"],
            "false_agent_warning": any(a != record["culprit_name"] for a in warnings),
            "lead_messages": (record["mistake_step"] - culprit_warning) if culprit_warning is not None
                             and culprit_warning < record["mistake_step"] else None,
            "timeline": timeline}


def keyword_monitor(record: dict) -> dict:
    """Temporal baseline: warn at the first explicit error keyword per agent."""
    warnings = {}
    for message in record["messages"]:
        if message["flags"]["error"] and message["agent"] not in warnings:
            warnings[message["agent"]] = message["raw_index"]
    culprit_warning = warnings.get(record["culprit_name"])
    eligible = any(m["raw_index"] < record["mistake_step"] for m in record["messages"])
    return {"id": record["id"], "actual_culprit": record["culprit_name"],
            "mistake_step": record["mistake_step"], "eligible": eligible,
            "warnings": warnings, "culprit_warning": culprit_warning,
            "correct_early": culprit_warning is not None and culprit_warning < record["mistake_step"],
            "false_agent_warning": any(a != record["culprit_name"] for a in warnings),
            "lead_messages": (record["mistake_step"] - culprit_warning) if culprit_warning is not None
                             and culprit_warning < record["mistake_step"] else None}


def warning_summary(results: list[dict]) -> dict:
    eligible = [r for r in results if r["eligible"]]
    successes = [r for r in eligible if r["correct_early"]]
    false = sum(r["false_agent_warning"] for r in results)
    leads = [r["lead_messages"] for r in successes]
    return {"total": len(results), "eligible": len(eligible), "correct_early": len(successes),
            "early_recall_eligible": len(successes) / len(eligible) if eligible else None,
            "false_agent_warning_traces": false,
            "false_agent_warning_rate": false / len(results) if results else None,
            "mean_lead_messages": sum(leads) / len(leads) if leads else None}


def choose_threshold(model: dict, validation: list[dict]) -> tuple[float, list[dict]]:
    # Maximize a trace-level F1-like score; prefer fewer false-agent warnings on ties.
    candidates = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
    scored = []
    for threshold in candidates:
        results = [monitor(model, r, threshold) for r in validation]
        summary = warning_summary(results)
        success = summary["correct_early"]
        false = summary["false_agent_warning_traces"]
        f1 = 2 * success / (2 * success + false + max(summary["eligible"] - success, 0)) if summary["eligible"] else 0
        scored.append({"threshold": threshold, "score": f1, **summary})
    best = max(scored, key=lambda x: (x["score"], -x["false_agent_warning_traces"], x["threshold"]))
    return best["threshold"], scored
