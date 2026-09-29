"""Baseline models, proper scores, validation calibration, and plots."""

from __future__ import annotations

import math
import random

import numpy as np
from scipy.optimize import minimize


def normalized(probs: list[float]) -> bool:
    return (bool(probs) and all(math.isfinite(p) and 0 <= p <= 1 for p in probs)
            and abs(sum(probs) - 1) < 1e-9)


def calibrate_temperature(probs: list[float], temperature: float) -> list[float]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = np.log(np.clip(np.asarray(probs, dtype=float), 1e-12, 1.0)) / temperature
    logits -= np.max(logits)
    values = np.exp(logits)
    return (values / values.sum()).tolist()


def fit_temperature(predictions: list[dict]) -> float:
    # Single parameter; choose on validation labels only.
    candidates = np.exp(np.linspace(math.log(0.4), math.log(5.0), 120))
    return float(min(candidates, key=lambda t: np.mean([
        -math.log(max(calibrate_temperature(p["probabilities"], float(t))[p["actual"]], 1e-12))
        for p in predictions])))


def prior_baseline(training: list[dict], n: int, alpha: float = 2.0) -> list[float]:
    rows = [r for r in training if r["n_agents"] == n]
    counts = [sum(r["culprit"] == i for r in rows) for i in range(n)]
    total = sum(counts) + n * alpha
    return [(count + alpha) / total for count in counts]


def keyword_baseline(row: dict, prior: list[float]) -> list[float]:
    # Explicitly simple heuristic: first agent whose log contains an error keyword.
    hits = [i for i, e in enumerate(row["evidence"]) if e["error"]]
    if not hits:
        return prior[:]
    result = [0.05 / len(prior)] * len(prior)
    result[hits[0]] += 0.95
    return result


def _feature_vector(row: dict) -> np.ndarray:
    from .data import FEATURES
    return np.array([1.0] + [float(e[f]) for e in row["evidence"] for f in FEATURES])


def fit_logistic_baseline(training: list[dict], n: int, l2: float = 3.0) -> dict:
    """A small, regularized multinomial baseline using the same evidence features."""
    rows = [r for r in training if r["n_agents"] == n]
    if len(rows) < 8:
        return {"n": n, "weights": None, "prior": prior_baseline(training, n)}
    x = np.stack([_feature_vector(r) for r in rows])
    y = np.array([r["culprit"] for r in rows], dtype=int)
    d = x.shape[1]

    def loss(flat):
        weights = flat.reshape(d, n)
        logits = x @ weights
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        value = -np.log(np.maximum(probs[np.arange(len(y)), y], 1e-12)).sum()
        value += 0.5 * l2 * np.sum(weights[1:] ** 2)
        residual = probs.copy()
        residual[np.arange(len(y)), y] -= 1
        gradient = x.T @ residual
        gradient[1:] += l2 * weights[1:]
        return value, gradient.ravel()

    result = minimize(loss, np.zeros(d * n), jac=True, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(f"logistic baseline optimization failed: {result.message}")
    return {"n": n, "weights": result.x.reshape(d, n).tolist(),
            "prior": prior_baseline(training, n), "l2": l2}


def predict_logistic_baseline(model: dict, row: dict) -> list[float]:
    if model["weights"] is None:
        return model["prior"][:]
    logits = _feature_vector(row) @ np.array(model["weights"])
    logits -= logits.max()
    values = np.exp(logits)
    return (values / values.sum()).tolist()


def metric_summary(predictions: list[dict], n_bins: int = 10) -> dict:
    if not predictions:
        return {"n": 0}
    for p in predictions:
        if not normalized(p["probabilities"]):
            raise ValueError(f"invalid posterior for {p.get('id')}")
    n = len(predictions)
    correct = sum(p["predicted"] == p["actual"] for p in predictions)
    brier = sum(sum((prob - int(i == p["actual"])) ** 2
                    for i, prob in enumerate(p["probabilities"])) for p in predictions) / n
    logloss = -sum(math.log(max(p["probabilities"][p["actual"]], 1e-12))
                   for p in predictions) / n
    bins = [{"low": i / n_bins, "high": (i + 1) / n_bins, "count": 0,
             "confidence_sum": 0.0, "correct": 0} for i in range(n_bins)]
    for p in predictions:
        confidence = max(p["probabilities"])
        b = bins[min(int(confidence * n_bins), n_bins - 1)]
        b["count"] += 1
        b["confidence_sum"] += confidence
        b["correct"] += int(p["predicted"] == p["actual"])
    ece = 0.0
    for b in bins:
        b["avg_confidence"] = b["confidence_sum"] / b["count"] if b["count"] else None
        b["accuracy"] = b["correct"] / b["count"] if b["count"] else None
        if b["count"]:
            ece += b["count"] / n * abs(b["accuracy"] - b["avg_confidence"])
    labels = sorted(set(p["actual"] for p in predictions))
    f1s = []
    for label in labels:
        tp = sum(p["actual"] == label and p["predicted"] == label for p in predictions)
        fp = sum(p["actual"] != label and p["predicted"] == label for p in predictions)
        fn = sum(p["actual"] == label and p["predicted"] != label for p in predictions)
        f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return {"n": n, "correct": correct, "accuracy": correct / n,
            "brier": brier, "log_loss": logloss, "ece": ece,
            "macro_f1": sum(f1s) / len(f1s), "bins": bins}


def bootstrap_accuracy_interval(predictions: list[dict], seed: int = 17, samples: int = 2000) -> list[float]:
    if not predictions:
        return [float("nan"), float("nan")]
    rng = random.Random(seed)
    n = len(predictions)
    scores = []
    for _ in range(samples):
        draw = [predictions[rng.randrange(n)] for _ in range(n)]
        scores.append(sum(p["predicted"] == p["actual"] for p in draw) / n)
    scores.sort()
    return [scores[int(0.025 * samples)], scores[int(0.975 * samples)]]


def draw_reliability(summary: dict, output_path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    populated = [b for b in summary["bins"] if b["count"]]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Perfect calibration")
    ax.scatter([b["avg_confidence"] for b in populated],
               [b["accuracy"] for b in populated], s=[30 + b["count"] * 8 for b in populated],
               color="steelblue", label="Test bins (size = trace count)")
    for b in populated:
        ax.annotate(str(b["count"]), (b["avg_confidence"], b["accuracy"]), xytext=(5, 5),
                    textcoords="offset points", fontsize=8)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted confidence",
           ylabel="Observed accuracy", title="Held-out root-cause calibration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
