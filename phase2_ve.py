#!/usr/bin/env python3
"""
Phase 2: Parameter Learning & Hand-Rolled Variable Elimination
==============================================================
Who&When multi-agent LLM failure benchmark — 4-agent subset.

Pipeline
--------
1. Parse every JSON trace; keep only files whose conversation history contains
   exactly 4 unique agents (after Phase-1 role cleaning).
2. Map agents to generic nodes by order of first appearance: A1, A2, A3, A4,
   and append a Final_Output node (always Failed=True: the pipeline failed).
   The node matching the ground-truth `mistake_agent` gets Failed=True.
3. Learn the CPTs of the chain network  A1 -> A2 -> A3 -> A4 -> Final_Output
   by counting occurrences across all 4-agent files (Laplace smoothing).
4. Hand-rolled Variable Elimination (basic Python + numpy broadcasting ONLY —
   no pgmpy / networkx / any probabilistic-library math) computes
   P(Ai = Failed | Final_Output = Failed) for i = 1..4.
5. Evaluate on 10 random 4-agent files: predicted culprit = argmax posterior;
   compare against the actual ground-truth mistake agent.
6. A brute-force joint-enumeration routine cross-validates the VE engine.

Outputs (./phase2_output/):
    Phase2_Results.txt      methodology + learned CPTs + 10 test blocks + summary
    phase2_cpts.json        learned CPTs (machine-readable)

Run locally:
    python3 phase2_ve.py                       # uses ./Agents_Failure_Attribution/Who&When
    python3 phase2_ve.py --data-root PATH      # custom dataset root
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import random
import re
import textwrap
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
AGENTS = ("A1", "A2", "A3", "A4")
FINAL_NODE = "Final_Output"
HUMAN_ROLES = {"human", "user"}
IGNORE_SUBSTRINGS = ("thought",)

STATE_LABEL = {0: "False", 1: "True"}


# --------------------------------------------------------------------------- #
# 1. Dataset parsing (Phase-1 role cleaning, reused)
# --------------------------------------------------------------------------- #
def clean_role(msg: dict):
    """Agent name for a history message, or None to ignore."""
    name = msg.get("name")
    if name is not None and str(name).strip():
        return str(name).strip()                      # Algorithm-Generated schema
    role = str(msg.get("role", "")).strip()
    low = role.lower()
    if low in HUMAN_ROLES or any(s in low for s in IGNORE_SUBSTRINGS):
        return None
    cleaned = re.sub(r"\s*\([^)]*\)", "", role).strip()
    return cleaned or None


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def speaker_sequence(history: list) -> list:
    seq = []
    for msg in history:
        agent = clean_role(msg) if isinstance(msg, dict) else None
        if agent and (not seq or seq[-1] != agent):
            seq.append(agent)
    return seq


def parse_dataset(root: Path):
    """Parse all JSONs; return records for files with exactly 4 unique agents."""
    records, skipped = [], collections.Counter()
    for path in sorted(root.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            skipped["unparseable"] += 1
            continue
        seq = speaker_sequence(data.get("history") or [])
        nodes = list(dict.fromkeys(seq))
        if len(nodes) != len(AGENTS):
            skipped[f"{len(nodes)}_agents"] += 1
            continue

        mistake_agent = str(data.get("mistake_agent", ""))
        mistake_pos = None                            # 1-based position
        if mistake_agent in nodes:
            mistake_pos = nodes.index(mistake_agent) + 1
        else:                                          # defensive fallback
            n_ma = norm(mistake_agent)
            for i, n in enumerate(nodes):
                if n_ma and n_ma == norm(n):
                    mistake_pos = i + 1
                    break
        if mistake_pos is None:
            skipped["mistake_agent_not_matched"] += 1
            continue

        records.append({
            "file_name": path.relative_to(root).as_posix(),
            "subset": path.parent.name,
            "agents": nodes,                          # real agent names, A1..A4 order
            "mistake_agent": mistake_agent,
            "mistake_position": mistake_pos,
        })
    return records, skipped


# --------------------------------------------------------------------------- #
# 2. Hand-rolled factor algebra (basic Python + numpy ONLY)
# --------------------------------------------------------------------------- #
class Factor:
    """Discrete factor over binary variables, stored as a dense numpy table."""

    __slots__ = ("variables", "table")

    def __init__(self, variables, table):
        self.variables = tuple(variables)
        self.table = np.asarray(table, dtype=float)
        expected = (2,) * len(self.variables)
        if self.table.shape != expected:
            raise ValueError(f"table shape {self.table.shape} != {expected} for {self.variables}")

    def _expanded(self, target_vars):
        """Broadcast this factor's table onto the axes `target_vars`."""
        missing = [v for v in target_vars if v not in self.variables]
        t = self.table.reshape(self.table.shape + (1,) * len(missing))
        order = list(self.variables) + missing
        t = np.transpose(t, [order.index(v) for v in target_vars])
        return np.broadcast_to(t, (2,) * len(target_vars))


def f_multiply(f1: Factor, f2: Factor) -> Factor:
    union = list(f1.variables) + [v for v in f2.variables if v not in f1.variables]
    return Factor(union, f1._expanded(union) * f2._expanded(union))


def f_sum_out(f: Factor, var: str) -> Factor:
    ax = f.variables.index(var)
    return Factor([v for v in f.variables if v != var], f.table.sum(axis=ax))


def f_reduce(f: Factor, var: str, state_index: int) -> Factor:
    ax = f.variables.index(var)
    return Factor([v for v in f.variables if v != var],
                  np.take(f.table, state_index, axis=ax))


def f_normalize(f: Factor) -> Factor:
    total = f.table.sum()
    if total <= 0:
        raise ValueError("Cannot normalize a factor with zero mass")
    return Factor(f.variables, f.table / total)


# --------------------------------------------------------------------------- #
# 3. CPT learning from the 4-agent records
# --------------------------------------------------------------------------- #
def learn_cpts(records: list, alpha: float = 1.0):
    """
    Learn chain CPTs by counting. Laplace (+alpha) smoothing keeps every
    conditional a proper distribution even for unobserved combinations.

    Returns (factors, cpt_report):
      factors    — list of Factor objects ready for inference
      cpt_report — dict of counts/tables for reporting
    """
    n_prior = {0: 0, 1: 0}                                        # A1
    n_child = {i: collections.Counter() for i in (1, 2, 3)}       # (parent,child) pairs
    n_final = collections.Counter()                               # (A4, Final_Output)

    for r in records:
        pos = r["mistake_position"]
        labels = [1 if pos == i + 1 else 0 for i in range(4)]     # exactly one True
        n_prior[labels[0]] += 1
        for i in (1, 2, 3):
            n_child[i][(labels[i - 1], labels[i])] += 1
        n_final[(labels[3], 1)] += 1                              # Final_Output always True

    def row(counts, parent_state, alpha):
        tot = sum(counts[(parent_state, c)] for c in (0, 1))
        return np.array([(counts[(parent_state, 0)] + alpha) / (tot + 2 * alpha),
                         (counts[(parent_state, 1)] + alpha) / (tot + 2 * alpha)])

    # P(A1)
    tot1 = n_prior[0] + n_prior[1]
    p_a1 = np.array([(n_prior[0] + alpha) / (tot1 + 2 * alpha),
                     (n_prior[1] + alpha) / (tot1 + 2 * alpha)])
    # P(A_{i+1} | A_i) and P(Final_Output | A4)
    cond_tables = {}
    for i in (1, 2, 3):
        rows = np.stack([row(n_child[i], p, alpha) for p in (0, 1)])
        cond_tables[i] = rows
    rows_f = np.stack([row(n_final, p, alpha) for p in (0, 1)])

    factors = [
        Factor(("A1",), p_a1),
        Factor(("A1", "A2"), cond_tables[1]),
        Factor(("A2", "A3"), cond_tables[2]),
        Factor(("A3", "A4"), cond_tables[3]),
        Factor(("A4", FINAL_NODE), rows_f),
    ]
    report = {
        "alpha": alpha,
        "num_training_files": len(records),
        "A1_counts": dict(n_prior),
        "child_counts": {f"A{i}->A{i+1}": {f"{STATE_LABEL[p]}->{STATE_LABEL[c]}": n_child[i][(p, c)]
                                           for p in (0, 1) for c in (0, 1)}
                         for i in (1, 2, 3)},
        "final_counts": {f"{STATE_LABEL[p]}->{STATE_LABEL[c]}": n_final[(p, c)]
                         for p in (0, 1) for c in (0, 1)},
        "P_A1": p_a1.tolist(),
        "P_A2_given_A1": cond_tables[1].tolist(),
        "P_A3_given_A2": cond_tables[2].tolist(),
        "P_A4_given_A3": cond_tables[3].tolist(),
        "P_Final_given_A4": rows_f.tolist(),
    }
    return factors, report


# --------------------------------------------------------------------------- #
# 4. Hand-rolled Variable Elimination
# --------------------------------------------------------------------------- #
def variable_elimination(factors: list, query_vars: list,
                         evidence: dict | None = None,
                         elim_order: list | None = None) -> Factor:
    """
    Classic VE: restrict by evidence, eliminate hidden variables one by one
    (multiply the factors mentioning the variable, sum it out), then multiply
    and normalize what remains. Pure numpy; no probabilistic libraries.
    """
    evidence = evidence or {}

    # (a) incorporate evidence by slicing factors
    working = []
    for f in factors:
        g = f
        for var, val in evidence.items():
            if var in g.variables:
                g = f_reduce(g, var, int(bool(val)))
        working.append(g)

    # (b) elimination order for hidden variables
    all_vars = list(dict.fromkeys(v for f in factors for v in f.variables))
    if elim_order is None:
        elim_order = all_vars
    hidden = [v for v in elim_order if v not in query_vars and v not in evidence]

    # (c) eliminate
    for var in hidden:
        related = [f for f in working if var in f.variables]
        rest = [f for f in working if var not in f.variables]
        if not related:
            continue
        prod = related[0]
        for f in related[1:]:
            prod = f_multiply(prod, f)
        working = rest + [f_sum_out(prod, var)]

    # (d) multiply remaining factors and normalize
    result = working[0]
    for f in working[1:]:
        result = f_multiply(result, f)
    result = f_normalize(result)

    # ensure query var ordering
    if set(result.variables) != set(query_vars):
        raise RuntimeError(f"VE produced {result.variables}, expected {query_vars}")
    return result


def posterior_failure_probabilities(factors: list,
                                    agents=AGENTS,
                                    final_node=FINAL_NODE,
                                    elim_order=None) -> dict:
    """P(Ai = Failed | Final_Output = Failed) for each agent node."""
    posteriors = {}
    for q in agents:
        others = [a for a in agents if a != q]
        f = variable_elimination(factors, [q], evidence={final_node: True},
                                 elim_order=others if elim_order is None else elim_order)
        posteriors[q] = float(f.table[1])             # index 1 == state True
    return posteriors


# --------------------------------------------------------------------------- #
# 5. Brute-force cross-validation (independent check of the VE engine)
# --------------------------------------------------------------------------- #
def brute_force_posteriors(factors: list, agents=AGENTS,
                           final_node=FINAL_NODE) -> dict:
    names = list(agents) + [final_node]
    masses = {q: 0.0 for q in agents}
    p_final_true = 0.0
    for assign in itertools.product((0, 1), repeat=len(names)):
        env = dict(zip(names, assign))
        if env[final_node] != 1:
            continue
        p = 1.0
        for f in factors:
            p *= f.table[tuple(env[v] for v in f.variables)]
        p_final_true += p
        for q in agents:
            if env[q] == 1:
                masses[q] += p
    return {q: masses[q] / p_final_true for q in agents}


# --------------------------------------------------------------------------- #
# 6. Results reporting
# --------------------------------------------------------------------------- #
def fmt_cpt_row(row) -> str:
    return f"[Failed=False: {row[0]:.6f} | Failed=True: {row[1]:.6f}]"


def write_results(out_path: Path, cpt_report: dict, posteriors: dict,
                  test_blocks: list, summary: dict, bf_max_diff: float,
                  args) -> None:
    W = 100
    L = []
    ap = L.append

    ap("=" * W)
    ap("PHASE 2: PARAMETER LEARNING & HAND-ROLLED VARIABLE ELIMINATION — Who&When benchmark")
    ap("=" * W)
    ap(f"Dataset root            : {args.data_root}")
    ap(f"Training files (exactly 4 unique agents): {cpt_report['num_training_files']}")
    ap(f"Network structure       : A1 -> A2 -> A3 -> A4 -> {FINAL_NODE}")
    ap(f"Node semantics          : agents mapped to A1..A4 by order of first appearance;")
    ap(f"                          {FINAL_NODE} appended and always Failed=True (pipeline failed).")
    ap(f"Failure labeling        : node matching ground-truth mistake_agent -> Failed=True; others False.")
    ap(f"Laplace smoothing alpha : {cpt_report['alpha']}")
    ap(f"VE implementation       : hand-rolled (numpy broadcasting + axis sums only; "
       f"no pgmpy/networkx).")
    ap(f"Random seed (test pick) : {args.seed}   |  num test files: {args.num_test}")
    ap("=" * W)

    ap("\nLEARNED CPTs (from counting over all 4-agent files)")
    ap("-" * W)
    ap(f"P(A1 Failed)                       = {fmt_cpt_row(cpt_report['P_A1'])}")
    ap(f"    raw A1 counts: False={cpt_report['A1_counts'].get(0, 0)}, "
       f"True={cpt_report['A1_counts'].get(1, 0)}")
    for i, key in ((1, "P_A2_given_A1"), (2, "P_A3_given_A2"), (3, "P_A4_given_A3")):
        tbl = cpt_report[key]
        ap(f"P(A{i+1} Failed | A{i}=False)      = {fmt_cpt_row(tbl[0])}")
        ap(f"P(A{i+1} Failed | A{i}=True)       = {fmt_cpt_row(tbl[1])}")
        cc = cpt_report["child_counts"][f"A{i}->A{i+1}"]
        ap(f"    raw counts A{i}->A{i+1}: " + ", ".join(f"{k}:{v}" for k, v in cc.items()))
    ap(f"P({FINAL_NODE} | A4=False)         = {fmt_cpt_row(cpt_report['P_Final_given_A4'][0])}")
    ap(f"P({FINAL_NODE} | A4=True)          = {fmt_cpt_row(cpt_report['P_Final_given_A4'][1])}")
    ap(f"    raw counts A4->Final: " + ", ".join(
        f"{k}:{v}" for k, v in cpt_report["final_counts"].items()))

    ap("\nPOSTERIOR P(Ai = Failed | Final_Output = True)  [hand-rolled VE]")
    ap("-" * W)
    for a in AGENTS:
        ap(f"    P({a} Failed | Final_Output=True) = {posteriors[a]:.6f}")
    predicted_node = max(AGENTS, key=lambda a: posteriors[a])
    ap(f"    argmax node (global prediction) = {predicted_node}")
    ap(f"\nBrute-force cross-check (32-state joint enumeration): "
       f"max |VE - brute force| over A1..A4 = {bf_max_diff:.3e}")

    ap("\n" + "=" * W)
    ap(f"TEST EVALUATION — {len(test_blocks)} random 4-agent files "
       f"(seed {args.seed})")
    ap("=" * W)
    for i, b in enumerate(test_blocks, 1):
        ap(f"\n{'#' * W}")
        ap(f"TEST BLOCK {i} / {len(test_blocks)}")
        ap(f"{'#' * W}")
        ap(f"File Name          : {b['file_name']}")
        ap(f"Agents (A1..A4)    : {', '.join(b['agents'])}")
        ap("VE Probabilities   :")
        for a in AGENTS:
            ap(f"    P({a} Failed | Final_Output=True) = {b['probs'][a]:.6f}")
        ap(f"Predicted Culprit  : {b['predicted_label']} "
           f"({b['predicted_agent_real']})")
        ap(f"Actual Culprit     : {b['actual_label']} "
           f"({b['actual_agent_real']})")
        ap(f"Correct Prediction : {'YES' if b['correct'] else 'NO'}")

    ap("\n" + "=" * W)
    ap("EVALUATION SUMMARY")
    ap("=" * W)
    ap(f"Correct predictions on the {len(test_blocks)} random test files: "
       f"{summary['num_correct']}/{len(test_blocks)} "
       f"({summary['num_correct'] / len(test_blocks):.1%})")
    ap(f"Accuracy of always predicting argmax node on ALL "
       f"{summary['num_all']} 4-agent files: "
       f"{summary['num_correct_all']}/{summary['num_all']} "
       f"({summary['num_correct_all'] / summary['num_all']:.1%})")
    ap("\nMistake-position distribution over all 4-agent files:")
    for pos in (1, 2, 3, 4):
        cnt = summary["position_dist"].get(pos, 0)
        ap(f"    A{pos}: {cnt:>3} files ({cnt / summary['num_all']:.1%})")
    ap("\nNOTE — why the 10 blocks show identical probabilities:")
    ap("  The network is a single global model and the only evidence is "
       "Final_Output=True, which is the same for every failed trace.")
    ap("  Per-file posteriors therefore coincide; the model's prediction is a "
       "learned positional prior (the most failure-prone slot, A1 here).")
    ap("  File-specific inference would require instance-level evidence "
       "(e.g., observed agent behaviors) — a natural Phase 3 extension.")

    ap("\n" + "=" * W)
    ap("END OF PHASE 2 REPORT")
    ap("=" * W)
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description="Phase 2: CPT learning + hand-rolled VE")
    p.add_argument("--data-root", default="Agents_Failure_Attribution/Who&When")
    p.add_argument("--output-dir", default="phase2_output")
    p.add_argument("--num-test", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Laplace smoothing pseudo-count")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. parse -------------------------------------------------------------- #
    records, skipped = parse_dataset(Path(args.data_root))
    print(f"Parsed dataset: {len(records)} files with exactly 4 unique agents "
          f"(skipped: {dict(skipped)})")

    # 2. learn CPTs --------------------------------------------------------- #
    factors, cpt_report = learn_cpts(records, alpha=args.alpha)
    print("Learned CPTs for chain A1->A2->A3->A4->Final_Output")

    # 3. VE posteriors + brute-force validation ------------------------------ #
    posteriors = posterior_failure_probabilities(factors)
    bf = brute_force_posteriors(factors)
    bf_max_diff = max(abs(posteriors[a] - bf[a]) for a in AGENTS)
    print("VE posteriors:", {a: round(v, 6) for a, v in posteriors.items()})
    print(f"Brute-force validation max diff: {bf_max_diff:.3e}")
    assert bf_max_diff < 1e-9, "VE disagrees with brute-force enumeration!"

    # 4. test on random files ------------------------------------------------ #
    rng = random.Random(args.seed)
    test_records = rng.sample(records, min(args.num_test, len(records)))
    test_blocks, num_correct = [], 0
    global_pred = max(AGENTS, key=lambda a: posteriors[a])
    for r in test_records:
        probs = posterior_failure_probabilities(factors)   # run VE per test file
        pred_label = max(AGENTS, key=lambda a: probs[a])
        actual_label = f"A{r['mistake_position']}"
        block = {
            "file_name": r["file_name"],
            "agents": r["agents"],
            "probs": probs,
            "predicted_label": pred_label,
            "predicted_agent_real": r["agents"][int(pred_label[1:]) - 1],
            "actual_label": actual_label,
            "actual_agent_real": r["mistake_agent"],
            "correct": pred_label == actual_label,
        }
        num_correct += block["correct"]
        test_blocks.append(block)

    # 5. summary over all files ---------------------------------------------- #
    position_dist = collections.Counter(r["mistake_position"] for r in records)
    num_correct_all = sum(1 for r in records
                          if f"A{r['mistake_position']}" == global_pred)

    # 6. outputs -------------------------------------------------------------- #
    (out_dir / "phase2_cpts.json").write_text(
        json.dumps(cpt_report, indent=2), encoding="utf-8")
    write_results(out_dir / "Phase2_Results.txt", cpt_report, posteriors,
                  test_blocks,
                  {"num_correct": num_correct,
                   "num_all": len(records),
                   "num_correct_all": num_correct_all,
                   "position_dist": position_dist},
                  bf_max_diff, args)
    print(f"Wrote {out_dir / 'Phase2_Results.txt'}")
    print(f"Wrote {out_dir / 'phase2_cpts.json'}")
    print(f"Test accuracy: {num_correct}/{len(test_blocks)} "
          f"(always-{global_pred} baseline on all files: "
          f"{num_correct_all}/{len(records)})")


if __name__ == "__main__":
    main()
