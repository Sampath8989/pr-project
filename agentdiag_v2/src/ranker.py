"""Trace-aware candidate ranker with lexical and observed-handoff evidence.

This is a discriminative complement to the categorical Bayesian network. It
never reads mistake labels when making a prediction. Hashed text features keep
the trained model compact and permit inference on unseen agent names.
"""

from __future__ import annotations

import math
import re
import zlib
from collections import Counter

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix


HASH_SIZE = 4096
ROLE_SIZE = 256
NUMERIC_SIZE = 18
DIMENSION = 2 * HASH_SIZE + ROLE_SIZE + NUMERIC_SIZE
TOKEN = re.compile(r"[a-z][a-z0-9_]{2,}")


def _terms(text: str) -> list[str]:
    words = TOKEN.findall(text.lower())[:900]
    return words + [f"{a}~{b}" for a, b in zip(words, words[1:])]


def _hash_counts(text: str, offset: int, size: int) -> dict[int, float]:
    counts = Counter(zlib.crc32(word.encode()) % size for word in _terms(text))
    values = {offset + index: math.log1p(min(count, 8)) for index, count in counts.items()}
    norm = math.sqrt(sum(v * v for v in values.values()))
    return {k: v / norm for k, v in values.items()} if norm else {}


def _numeric(record: dict, agent: str) -> list[float]:
    messages = record["messages"]
    authored = [m for m in messages if m["agent"] == agent]
    index = record["agents"].index(agent)
    n_agents = len(record["agents"])
    first = authored[0]["raw_index"]
    last = authored[-1]["raw_index"]
    handoffs = [(m, messages[j + 1]) for j, m in enumerate(messages[:-1])
                if m["agent"] == agent and messages[j + 1]["agent"] != agent]
    first_error = next((j for j, m in enumerate(messages) if m["flags"]["error"]), None)
    later = [m for m in messages if m["raw_index"] > first and m["agent"] != agent]
    text_length = sum(len(m["content"]) for m in authored)
    return [
        index / max(n_agents - 1, 1), float(index == 0), float(index == 1),
        min(len(authored) / 10, 2), min(math.log1p(text_length) / 10, 2),
        min(first / 10, 2), min(last / max(record["raw_message_count"], 1), 1),
        float(authored[0]["content"].startswith("You are given:")),
        float(any(m["flags"]["error"] for m in authored)),
        float(any(m["flags"]["uncertain"] for m in authored)),
        float(any(m["flags"]["correction"] for m in authored)),
        sum(m["flags"]["error"] for m in later) / max(len(later), 1),
        sum(next_msg["flags"]["error"] for _, next_msg in handoffs) / max(len(handoffs), 1),
        sum(next_msg["flags"]["correction"] for _, next_msg in handoffs) / max(len(handoffs), 1),
        float(first_error is not None and first_error > 0
              and messages[first_error - 1]["agent"] == agent),
        min(len(handoffs) / 8, 2),
        float("terminal" in agent.lower() or "computer" in agent.lower()),
        float("valid" in agent.lower() or "verif" in agent.lower()),
    ]


def candidate_features(record: dict, agent: str, components: str = "text_graph") -> dict[int, float]:
    authored = [m["content"] for m in record["messages"] if m["agent"] == agent]
    result = {}
    if components in {"text", "text_graph"}:
        first_text = " ".join(authored[:2])[:5000]
        last_text = " ".join(authored[-2:])[:5000]
        result.update(_hash_counts(first_text, 0, HASH_SIZE))
        result.update(_hash_counts(last_text, HASH_SIZE, HASH_SIZE))
        result.update(_hash_counts(agent.replace("_", " "), 2 * HASH_SIZE, ROLE_SIZE))
    if components in {"graph", "text_graph"}:
        base = 2 * HASH_SIZE + ROLE_SIZE
        result.update({base + i: value for i, value in enumerate(_numeric(record, agent)) if value})
    return result


def design(records: list[dict], components: str = "text_graph"):
    data, indices, indptr = [], [], [0]
    starts, lengths, labels = [], [], []
    row_count = 0
    for record in records:
        starts.append(row_count)
        lengths.append(len(record["agents"]))
        labels.append(record.get("culprit"))
        for agent in record["agents"]:
            features = candidate_features(record, agent, components)
            for index, value in sorted(features.items()):
                indices.append(index)
                data.append(value)
            indptr.append(len(data))
            row_count += 1
    matrix = csr_matrix((np.array(data, dtype=np.float64), np.array(indices), np.array(indptr)),
                        shape=(row_count, DIMENSION))
    return matrix, np.array(starts, dtype=int), np.array(lengths, dtype=int), labels


def fit(records: list[dict], l2: float = 1.0, components: str = "text_graph", maxiter: int = 100) -> dict:
    matrix, starts, lengths, labels = design(records, components)
    if any(label is None for label in labels):
        raise ValueError("training records require culprit labels")
    correct_rows = starts + np.asarray(labels, dtype=int)

    def objective(weights):
        scores = np.asarray(matrix @ weights).ravel()
        scores -= np.repeat(np.maximum.reduceat(scores, starts), lengths)
        exponent = np.exp(scores)
        probabilities = exponent / np.repeat(np.add.reduceat(exponent, starts), lengths)
        value = -np.log(np.maximum(probabilities[correct_rows], 1e-12)).sum()
        value += l2 * float(weights @ weights) / 2
        residual = probabilities.copy()
        residual[correct_rows] -= 1
        gradient = np.asarray(matrix.T @ residual).ravel() + l2 * weights
        return value, gradient

    result = minimize(objective, np.zeros(DIMENSION), jac=True, method="L-BFGS-B",
                      options={"maxiter": maxiter})
    if not result.success and result.status != 1:
        raise RuntimeError(f"ranker optimization failed: {result.message}")
    return {"weights": result.x.tolist(), "l2": l2, "components": components,
            "training_traces": len(records), "iterations": int(result.nit)}


def predict(model: dict, records: list[dict]) -> list[list[float]]:
    matrix, starts, lengths, _ = design(records, model["components"])
    scores = np.asarray(matrix @ np.asarray(model["weights"])).ravel()
    scores -= np.repeat(np.maximum.reduceat(scores, starts), lengths)
    exponent = np.exp(scores)
    probabilities = exponent / np.repeat(np.add.reduceat(exponent, starts), lengths)
    return [probabilities[start:start + length].tolist() for start, length in zip(starts, lengths)]
