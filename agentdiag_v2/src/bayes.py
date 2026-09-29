"""Categorical root-cause Bayesian network and hand-written Variable Elimination."""

from __future__ import annotations

import itertools
from collections import Counter

from .data import FEATURES


class Factor:
    def __init__(self, variables: tuple[str, ...], domains: dict[str, tuple[int, ...]], table: dict[tuple[int, ...], float]):
        self.variables, self.domains, self.table = variables, domains, table

    def value(self, assignment: dict[str, int]) -> float:
        return self.table[tuple(assignment[v] for v in self.variables)]

    def reduce(self, evidence: dict[str, int]) -> "Factor":
        remaining = tuple(v for v in self.variables if v not in evidence)
        table = {}
        for values in itertools.product(*(self.domains[v] for v in remaining)):
            assignment = dict(zip(remaining, values)) | evidence
            table[values] = self.value(assignment)
        return Factor(remaining, self.domains, table)

    def sum_out(self, variable: str) -> "Factor":
        remaining = tuple(v for v in self.variables if v != variable)
        table = {}
        for values in itertools.product(*(self.domains[v] for v in remaining)):
            assignment = dict(zip(remaining, values))
            table[values] = sum(self.value(assignment | {variable: state})
                                for state in self.domains[variable])
        return Factor(remaining, self.domains, table)


def multiply(left: Factor, right: Factor) -> Factor:
    variables = tuple(dict.fromkeys(left.variables + right.variables))
    domains = left.domains
    table = {}
    for values in itertools.product(*(domains[v] for v in variables)):
        assignment = dict(zip(variables, values))
        table[values] = left.value(assignment) * right.value(assignment)
    return Factor(variables, domains, table)


def variable_elimination(factors: list[Factor], evidence: dict[str, int], query: str = "C") -> list[float]:
    domains = factors[0].domains
    working = [f.reduce({k: v for k, v in evidence.items() if k in f.variables}) for f in factors]
    hidden = [v for v in domains if v != query and v not in evidence]
    for var in hidden:
        related = [f for f in working if var in f.variables]
        working = [f for f in working if var not in f.variables]
        if related:
            product = related[0]
            for f in related[1:]:
                product = multiply(product, f)
            working.append(product.sum_out(var))
    product = working[0]
    for f in working[1:]:
        product = multiply(product, f)
    values = [product.table[(c,)] for c in domains[query]]
    total = sum(values)
    if total <= 0:
        raise ValueError("evidence has zero probability")
    return [v / total for v in values]


def fit_model(rows: list[dict], n: int, features: tuple[str, ...], alpha: float = 2.0) -> dict:
    selected = [r for r in rows if r["n_agents"] == n]
    if not selected:
        raise ValueError(f"no training rows with {n} agents")
    prior_count = Counter(r["culprit"] for r in selected)
    prior = [(prior_count[c] + alpha) / (len(selected) + n * alpha) for c in range(n)]
    # Very small groups use a smoothed prior only: their CPTs cannot be estimated reliably.
    features = features if len(selected) >= 8 else ()
    cpts = {}
    for feature in features:
        first = []
        later = []
        for c in range(n):
            group = [r for r in selected if r["culprit"] == c]
            hits = sum(r["evidence"][0][feature] for r in group)
            first.append((hits + alpha) / (len(group) + 2 * alpha))
            rows_by_i = []
            for i in range(1, n):
                per_previous = []
                for prev in (0, 1):
                    subset = [r for r in group if r["evidence"][i - 1][feature] == prev]
                    count = sum(r["evidence"][i][feature] for r in subset)
                    per_previous.append((count + alpha) / (len(subset) + 2 * alpha))
                rows_by_i.append(per_previous)
            later.append(rows_by_i)
        cpts[feature] = {"first": first, "later": later}
    return {"n": n, "features": list(features), "alpha": alpha,
            "training_count": len(selected), "prior": prior, "cpts": cpts}


def factors_for(model: dict) -> list[Factor]:
    n = model["n"]
    domains = {"C": tuple(range(n))}
    for f in model["features"]:
        for i in range(n):
            domains[f"{f}_{i}"] = (0, 1)
    result = [Factor(("C",), domains, {(c,): p for c, p in enumerate(model["prior"])})]
    for f in model["features"]:
        cpt = model["cpts"][f]
        for i in range(n):
            cur = f"{f}_{i}"
            if i == 0:
                table = {(c, x): cpt["first"][c] if x else 1 - cpt["first"][c]
                         for c in range(n) for x in (0, 1)}
                result.append(Factor(("C", cur), domains, table))
            else:
                prev_name = f"{f}_{i-1}"
                table = {(c, p, x): cpt["later"][c][i-1][p] if x else 1 - cpt["later"][c][i-1][p]
                         for c in range(n) for p in (0, 1) for x in (0, 1)}
                result.append(Factor(("C", prev_name, cur), domains, table))
    return result


def predict(model: dict, evidence: list[dict | None]) -> list[float]:
    if len(evidence) != model["n"]:
        raise ValueError("evidence agent count differs from model")
    observed = {f"{f}_{i}": int(evidence[i][f])
                for f in model["features"] for i in range(model["n"])
                if evidence[i] is not None and f in evidence[i] and evidence[i][f] is not None}
    return variable_elimination(factors_for(model), observed)


def brute_force(model: dict, evidence: list[dict | None]) -> list[float]:
    factors = factors_for(model)
    domains = factors[0].domains
    observed = {f"{f}_{i}": int(evidence[i][f])
                for f in model["features"] for i in range(model["n"])
                if evidence[i] is not None and f in evidence[i] and evidence[i][f] is not None}
    free = [v for v in domains if v not in observed]
    masses = [0.0] * model["n"]
    for values in itertools.product(*(domains[v] for v in free)):
        assignment = dict(zip(free, values)) | observed
        mass = 1.0
        for f in factors:
            mass *= f.value(assignment)
        masses[assignment["C"]] += mass
    total = sum(masses)
    return [v / total for v in masses]


def fit_all(rows: list[dict], features: tuple[str, ...]) -> dict[str, dict]:
    return {str(n): fit_model(rows, n, features) for n in sorted({r["n_agents"] for r in rows})}
