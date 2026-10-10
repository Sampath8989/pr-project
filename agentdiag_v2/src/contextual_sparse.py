"""Lexical contextual comparator with an explicitly trained agent head."""

from __future__ import annotations

import math
import re
import zlib
from collections import Counter

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix

from .step_ranker import features as old_features, hashes

STEP_D = 32768 + 64
AGENT_D = 16384 + 64


def terms(text: str, offset: int, size: int, bigrams: bool = False) -> dict:
    words = re.findall(r"[a-z][a-z0-9_]*", text.lower())[:1000]
    if bigrams:
        words = [a + " " + b for a, b in zip(words, words[1:])]
    counts = Counter(zlib.crc32(w.encode()) % size for w in words)
    v = {offset + i: math.log1p(min(n, 8)) for i, n in counts.items()}
    norm = math.sqrt(sum(x * x for x in v.values())) or 1
    return {i: x / norm for i, x in v.items()}


def step_features(r: dict, j: int) -> dict:
    msgs, m = r["messages"], r["messages"][j]
    f = terms(m["content"], 0, 8192)
    f.update(terms(m["content"], 8192, 8192, bigrams=True))
    if j + 1 < len(msgs):
        f.update(terms(msgs[j + 1]["content"], 16384, 4096))
    if j:
        f.update(terms(msgs[j - 1]["content"], 20480, 2048))
    f.update(terms(" ".join(x["content"] for x in msgs[j + 1:j + 4]), 22528, 4096))
    f.update(terms(m["agent"], 26624, 1024))
    f.update(terms(" ".join(x["content"] for x in msgs if x["agent"] == m["agent"]), 27648, 4096))
    old = old_features(r, j)
    f.update({32768 + i: old.get(6400 + i, 0) for i in range(32)})
    qwords = set(re.findall(r"\w+", r.get("question", "").lower()))
    mwords = set(re.findall(r"\w+", m["content"].lower()))
    f[32800] = len(qwords & mwords) / max(len(qwords), 1)
    return f


def agent_features(r: dict, a: int) -> dict:
    name, msgs = r["agents"][a], r["messages"]
    own = [m for m in msgs if m["agent"] == name]
    f = terms(" ".join(m["content"] for m in own), 0, 8192)
    f.update(terms(own[0]["content"], 8192, 2048))
    f.update(terms(own[-1]["content"], 10240, 2048))
    f.update(terms(name, 12288, 1024))
    following = [msgs[j + 1]["content"] for j, m in enumerate(msgs[:-1]) if m["agent"] == name]
    f.update(terms(" ".join(following), 13312, 2048))
    num = [a / max(len(r["agents"]) - 1, 1), len(own) / len(msgs),
           float(own[0] is msgs[0]), float(own[-1] is msgs[-1]),
           own[0]["raw_index"] / max(r["raw_message_count"] - 1, 1),
           own[-1]["raw_index"] / max(r["raw_message_count"] - 1, 1)]
    for flag in ("error", "uncertain", "correction"):
        vals = [m["flags"][flag] for m in own]
        num += [max(vals), sum(vals) / len(vals), vals[0], vals[-1]]
    num += [float(any(k in name.lower() for k in ("valid", "verif", "review"))),
            float(any(k in name.lower() for k in ("code", "computer", "terminal")))]
    f.update({16384 + i: v for i, v in enumerate(num) if v})
    return f


def design(records: list[dict], head: str):
    dimension, extractor = (STEP_D, step_features) if head == "step" else (AGENT_D, agent_features)
    data, indices, ptr, starts, lengths = [], [], [0], [], []
    for r in records:
        n = len(r["messages"]) if head == "step" else len(r["agents"])
        starts.append(len(ptr) - 1)
        lengths.append(n)
        for j in range(n):
            for index, value in sorted(extractor(r, j).items()):
                if value:
                    indices.append(index)
                    data.append(value)
            ptr.append(len(data))
    return csr_matrix((data, indices, ptr), shape=(len(ptr) - 1, dimension)), np.array(starts), np.array(lengths)


def fit_head(records: list[dict], head: str, l2: float):
    x, starts, lengths = design(records, head)
    target = [next(j for j, m in enumerate(r["messages"]) if m["raw_index"] == r["mistake_step"])
              if head == "step" else r["culprit"] for r in records]
    correct = starts + target
    def objective(w):
        s = np.asarray(x @ w).ravel()
        s -= np.repeat(np.maximum.reduceat(s, starts), lengths)
        e = np.exp(s)
        p = e / np.repeat(np.add.reduceat(e, starts), lengths)
        value = -np.log(p[correct].clip(1e-12)).sum() + l2 * (w @ w) / 2
        p[correct] -= 1
        return value, np.asarray(x.T @ p).ravel() + l2 * w
    res = minimize(objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B",
                   options={"maxiter": 100})
    if not res.success and res.status != 1:
        raise RuntimeError(res.message)
    return res.x


def fit(records: list[dict], l2: float = 1):
    return {"step_weights": fit_head(records, "step", l2).tolist(),
            "agent_weights": fit_head(records, "agent", l2).tolist(), "l2": l2}


def predict(model: dict, records: list[dict]):
    probabilities = {}
    for head in ("step", "agent"):
        x, starts, lengths = design(records, head)
        s = np.asarray(x @ np.asarray(model[head + "_weights"])).ravel()
        s -= np.repeat(np.maximum.reduceat(s, starts), lengths)
        p = np.exp(s)
        p /= np.repeat(np.add.reduceat(p, starts), lengths)
        probabilities[head] = [p[start:start + n] for start, n in zip(starts, lengths)]
    return [{"predicted_step": r["messages"][int(sp.argmax())]["raw_index"],
             "predicted_agent": r["agents"][int(ap.argmax())],
             "step_probabilities": sp.tolist(), "agent_probabilities": ap.tolist()}
            for r, sp, ap in zip(records, probabilities["step"], probabilities["agent"])]
