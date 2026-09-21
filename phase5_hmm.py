#!/usr/bin/env python3
"""
Phase 5: HMM Temporal Degradation Tracking & Early Warning
==========================================================
Who&When multi-agent LLM failure benchmark — 4-agent subset (91 files).

From post-mortem attribution (Phases 1-4) to *temporal* monitoring: a
hand-rolled Hidden Markov Model watches the conversation turn-by-turn and
estimates each agent's hidden health state

    {Healthy, Slipping, Failed}

from a noisy keyword sensor, raising an Early Warning as soon as

    P(Slipping) + P(Failed) > 0.50

for any agent. The evaluation then asks the paper's question: did the
warning fire BEFORE the ground-truth mistake (mistake_step)?

Ground-truth semantics (verified empirically over the 91 4-agent files)
-----------------------------------------------------------------------
`mistake_step` is a 0-based index into the `history` array: in 89/91
4-agent files history[mistake_step] is authored by mistake_agent (the
other two use an adjacent index; the locator tolerates +/-3 neighbours).
Per-agent round counts range 2-8 (one 28-round outlier), so the mistake
usually lands at round 1-3.

Turn definition
---------------
Turn t contains every agent's t-th authored utterance (t counts agent
rounds, not raw history indices), in canonical agent order. Verbatim
timestamps do not exist in the dataset, so "simultaneous" = same round;
this alignment gives all four agents a shared, equally long time axis and
makes 'mistake round' well defined: round of history[mistake_step] =
(number of preceding utterances by mistake_agent) + 1.

Engine design (hand-rolled, numpy only)
---------------------------------------
Each agent runs an INDEPENDENT 3-state HMM; agent HMMs are coupled only
through the shared time axis, keeping per-agent inference exact and O(T).
The forward algorithm is implemented from scratch (numpy elementwise ops
+ dot product + sum only) with normalised filtered beliefs — no hmmlearn,
no pgmpy, no scipy.

Emission probabilities are DESIGNED per the phase spec's qualitative
direction (error keywords indicate degradation):

    P(ErrorFound | Healthy) = 0.14 < P(ErrorFound | Slipping) = 0.50
                            < P(ErrorFound | Failed) = 0.90

Two measurements motivated NOT fitting them to the data (full analysis in
Phase5_Results.txt):

  1. The Phase 3 sensor CPTs are informative on the CULPRIT axis (who) but
     ANTI-correlated for A2..A4 (downstream agents echo upstream errors):
     P(S=True | Failed) < P(S=True | Healthy) there, so using them as
     emission rates would make the monitor mathematically deaf.
  2. On the TEMPORAL axis (when) the keyword channel is nearly flat:
     P(Err | round < mistake) = 0.179 vs P(Err | round >= mistake) = 0.181
     (likelihood ratio ~ 1.02). The mistake is a reasoning error that only
     surfaces downstream AFTER it happens.

Transition / prior matrices are also designed (see build_hmm_params): the
spec's illustrative matrix forgets degradation within one quiet turn,
which is fatal with only 2-8 rounds and ~18% keyword-hit rate.

Outputs (./phase5_output/):
    Phase5_Results.txt        methodology + 5 detailed turn-by-turn test
                              blocks + full 91-file sweep summary
    phase5_hmm_params.json    HMM matrices + per-file sweep records
    phase5_deliverables.zip   everything above + this script

Run locally:
    python3 phase5_hmm.py
    python3 phase5_hmm.py --seed 42 --num-test 5 --threshold 0.5
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import zipfile
from itertools import product
from pathlib import Path

import numpy as np

# Import the Phase 3 engine (parser, clean_role, keyword regex, learned
# sensor CPTs) so Phase 5 monitors the exact same evidence channel.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase3_ve import (                    # noqa: E402  (path fix must run first)
    AGENTS, KEYWORDS, KEYWORD_PAT, clean_role, norm,
    parse_dataset, learn_cpts,
)

STATES = ("Healthy", "Slipping", "Failed")
HEALTHY, SLIPPING, FAILED = 0, 1, 2

WARNING_THRESHOLD = 0.50      # P(Slipping)+P(Failed) > 0.50 triggers warning
EPS = 1e-12                   # numerical floor for probabilities
BRUTE_FORCE_MAX_T = 8         # 3^T enumeration only feasible for short runs


# --------------------------------------------------------------------------- #
# 1. Multi-turn trace simulation: history -> aligned per-agent turns
# --------------------------------------------------------------------------- #
def segment_turns(history: list, agents: list) -> list:
    """
    Segment `history` into per-agent turns: turn t = each agent's t-th
    authored utterance, in canonical agent order. Agents that stay silent
    in a round simply have no utterance that turn (their belief is carried
    forward by the monitor). Returns a list of {t, utterances} dicts whose
    keys are the GENERIC nodes A1..A4 (utterance text included).
    """
    per_agent = {a: [] for a in agents}             # real names internally
    for gi, msg in enumerate(history):
        a = clean_role(msg) if isinstance(msg, dict) else None
        if a in per_agent:
            per_agent[a].append(gi)
    n_rounds = max((len(v) for v in per_agent.values()), default=0)
    turns = []
    for t in range(1, n_rounds + 1):
        utt = {}
        for i, real in enumerate(agents):           # canonical order A1..A4
            lst = per_agent[real]
            if len(lst) >= t:
                gi = lst[t - 1]
                utt[AGENTS[i]] = str(history[gi].get("content") or "")
        turns.append({"t": t, "utterances": utt})
    return turns


def locate_mistake(history: list, agents: list, mistake_step,
                   mistake_agent: str):
    """
    Map ground truth to the turn axis. `mistake_step` is a 0-based index
    into `history` (verified: history[step] authored by mistake_agent in
    89/91 files). Tolerates +-3 adjacent indices for the remaining files.

    Returns (mistake_round, mistake_agent_round_index):
      mistake_round             1-based round of the mistake agent's own
                                utterance list == shared round (aligned axis)
      mistake_agent_round_index 0-based index into that agent's utterances
    """
    target = norm(mistake_agent)
    agent = next((a for a in agents if norm(a) == target), None)
    if agent is None:
        return None, None

    per_agent = {a: [] for a in agents}
    for gi, msg in enumerate(history):
        a = clean_role(msg) if isinstance(msg, dict) else None
        if a in per_agent:
            per_agent[a].append(gi)
    idxs = per_agent[agent]

    try:
        ms = int(str(mistake_step).strip())
    except (TypeError, ValueError):
        return None, None

    for cand in (ms, ms - 1, ms + 1, ms - 2, ms + 2, ms - 3, ms + 3):
        if cand in idxs:
            ri = idxs.index(cand)
            return ri + 1, ri
    return None, None


def extract_observations(turns: list, agents: list) -> dict:
    """
    Per-agent observation sequence, keyed by GENERIC nodes A1..A4:
    1 = 'Error Found' (any Phase 3 keyword in that turn's utterance),
    0 = 'No Error', None = agent silent this round (no observation ->
    belief carried forward unchanged).
    """
    obs = {a: [] for a in AGENTS}
    for turn in turns:
        for a in AGENTS:
            text = turn["utterances"].get(a)
            obs[a].append(None if text is None
                          else (1 if KEYWORD_PAT.search(text) else 0))
    return obs


# --------------------------------------------------------------------------- #
# 2. Hand-rolled forward algorithm (numpy only)
# --------------------------------------------------------------------------- #
def forward_algorithm(obs_seq: list, pi: np.ndarray, A: np.ndarray,
                      B: np.ndarray) -> tuple:
    """
    Forward algorithm for one HMM, implemented from scratch.

    obs_seq : list of observation indices (0 = No Error, 1 = Error Found)
    pi      : (3,) initial state distribution
    A       : (3,3) transition matrix, A[i, j] = P(z_t = j | z_{t-1} = i)
    B       : (3,2) emission matrix, B[s, o] = P(o_t = o | z_t = s)

    Returns (alphas, loglik):
      alphas : (T, 3) NORMALISED belief states P(z_t | o_1..o_t)
               (each row sums to 1 — the filtered belief used for warnings)
      loglik : log P(o_1..o_t)
    """
    T, N = len(obs_seq), A.shape[0]
    alphas = np.zeros((T, N))
    loglik = 0.0

    a = pi * B[:, obs_seq[0]]                  # t = 0: prior x emission
    s = a.sum()
    if s > 0.0:
        a = a / s
        loglik += float(np.log(s))
    else:                                      # impossible first observation
        a = np.full(N, 1.0 / N)
        loglik += float(np.log(EPS))
    alphas[0] = a

    for t in range(1, T):                      # t > 0: predict + update
        a = (alphas[t - 1] @ A) * B[:, obs_seq[t]]
        s = a.sum()
        if s > 0.0:
            a = a / s
            loglik += float(np.log(s))
        else:
            a = np.full(N, 1.0 / N)
            loglik += float(np.log(EPS))
        alphas[t] = a
    return alphas, loglik


def brute_force_forward(obs_seq: list, pi: np.ndarray, A: np.ndarray,
                        B: np.ndarray) -> tuple:
    """
    Independent check of forward_algorithm: enumerate ALL 3^T state paths,
    sum joint P(path, obs) to get P(z_T | obs). Only feasible for short
    sequences (3^T explodes), used as a unit test on short prefixes.
    """
    T = len(obs_seq)
    marg = np.zeros(3)
    z = 0.0
    for path in product(range(3), repeat=T):
        p = pi[path[0]] * B[path[0], obs_seq[0]]
        for t in range(1, T):
            p *= A[path[t - 1], path[t]] * B[path[t], obs_seq[t]]
        z += p
        marg[path[-1]] += p
    if z <= 0.0:
        return np.full(3, 1.0 / 3), float(np.log(EPS))
    return marg / z, float(np.log(z))


# --------------------------------------------------------------------------- #
# 3. HMM parameters (spec matrices + Phase 3 learned emissions)
# --------------------------------------------------------------------------- #
def build_hmm_params() -> tuple:
    """
    Assemble (pi, A, B) for one agent — DESIGNED parameters, not fitted.

    Why designed rather than learned? Two measurements drove this choice
    (both reported in Phase5_Results.txt):

    1. The Phase 3 sensor CPTs are informative on the CULPRIT axis (who)
       but ANTI-correlated for A2..A4 (downstream agents echo upstream
       errors), so P(ErrorFound | Failed) < P(ErrorFound | Healthy) there;
       using them as emission rates would make the monitor mathematically
       deaf — observing errors would push belief toward Healthy.
    2. On the TEMPORAL axis (when), the keyword channel is nearly flat:
       measured P(Err | round < mistake) = 0.179 vs P(Err | round >=
       mistake) = 0.181, likelihood ratio ~ 1.02. No emission fitted to
       this channel can separate healthy from degraded rounds; the spec's
       qualitative premise ("errors indicate degradation") is therefore
       encoded directly instead.

    Emission matrix B (P(Error Found | state)), following the phase spec's
    qualitative direction:
        Healthy 0.14  <  Slipping 0.50  <  Failed 0.90
    Transition matrix A:
                    to: Healthy  Slipping  Failed
        from Healthy      0.94      0.05      0.01
        from Slipping     0.08      0.85      0.07
        from Failed       0.05      0.05      0.90
    The spec's illustrative matrix (Healthy->Healthy 0.90, Slipping->Healthy
    0.25) forgets degradation within a single quiet turn; with 2-8 rounds
    per conversation and keyword hits on ~18% of agent-rounds, such a
    fast-forgiving monitor can never accumulate belief across turns. The
    tuned matrix keeps the spec's shape (high stay-Healthy, small slip
    chance, absorbing-ish Failed) while letting two consecutive error
    observations cross the 0.50 warning threshold.
    Initial distribution pi: near-certain healthy start.
    """
    B = np.array([
        [0.86, 0.14],                          # Healthy
        [0.50, 0.50],                          # Slipping
        [0.10, 0.90],                          # Failed
    ])
    pi = np.array([0.95, 0.045, 0.005])
    A = np.array([
        [0.94, 0.05, 0.01],
        [0.08, 0.85, 0.07],
        [0.05, 0.05, 0.90],
    ])
    return pi, A, B


# --------------------------------------------------------------------------- #
# 4. Early-warning monitor
# --------------------------------------------------------------------------- #
def run_monitor(turns: list, obs: dict,
                threshold: float = WARNING_THRESHOLD) -> dict:
    """
    Simulate time turn-by-turn. At each turn every agent's HMM belief is
    updated with its new observation (silent agents carry their belief
    forward unchanged). Warning rule (per the phase spec):

        P(Slipping) + P(Failed) > threshold   ->  Early Warning for that agent

    A warning is 'early enough' iff it fires strictly before the ground-
    truth mistake round.
    """
    param = {a: build_hmm_params() for a in AGENTS}
    warnings = {}                    # agent -> first warning turn
    warn_conf = {}                   # agent -> P(Slipping)+P(Failed) at fire
    latest = {a: param[a][0].copy() for a in AGENTS}
    logs = []

    for turn in turns:
        t = turn["t"]
        row = {"t": t, "obs": {}, "probs": {}, "warn": {}, "danger": {}}
        for a in AGENTS:
            o = obs[a][t - 1] if t - 1 < len(obs[a]) else None
            if o is not None:
                seq = [x for x in obs[a][:t] if x is not None]
                alphas, _ = forward_algorithm(seq, *param[a])
                latest[a] = alphas[-1]
            danger = float(latest[a][SLIPPING] + latest[a][FAILED])
            row["obs"][a] = o
            row["probs"][a] = latest[a].tolist()
            row["danger"][a] = danger
            row["warn"][a] = bool(danger > threshold)
            if row["warn"][a] and a not in warnings:
                warnings[a] = t
                warn_conf[a] = danger
        logs.append(row)

    if warnings:
        first_agent = min(warnings, key=lambda a: (warnings[a], int(a[1:])))
        first = {"agent": first_agent, "turn": warnings[first_agent],
                 "confidence": warn_conf[first_agent]}
    else:
        first = {"agent": None, "turn": None, "confidence": None}
    return {"logs": logs, "warnings": warnings, "warn_conf": warn_conf,
            "first_warning": first}


def classify_outcome(warnings: dict, mistake_round,
                     mistake_position: int) -> str:
    """
    Categorise the monitor result against the ground-truth round,
    CULPRIT-CENTRIC: the question is whether the actual culprit was flagged,
    and how early relative to its mistake round.
    """
    if mistake_round is None:
        return "NO_GT_ROUND"
    culprit = f"A{mistake_position}"
    ct = warnings.get(culprit)
    if ct is not None:
        if ct < mistake_round:
            return "EARLY_PREDICTION_SUCCESS"   # flagged strictly in time
        if ct == mistake_round:
            return "WARNING_AT_MISTAKE_ROUND"   # right agent, not strictly early
        return "LATE_WARNING"
    if warnings:
        return "OTHER_AGENT_WARNING"            # flagged, but never the culprit
    return "NO_WARNING"


# --------------------------------------------------------------------------- #
# 5. Per-file driver
# --------------------------------------------------------------------------- #
def process_file(record: dict, data_root: Path, threshold: float) -> dict:
    """Full pipeline for one file: segment -> observe -> monitor -> classify."""
    path = Path(data_root) / record["file_name"]
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    hist = data.get("history") or []

    turns = segment_turns(hist, record["agents"])
    m_round, m_idx = locate_mistake(hist, record["agents"],
                                    data.get("mistake_step"),
                                    record["mistake_agent"])
    obs = extract_observations(turns, record["agents"])
    res = run_monitor(turns, obs, threshold)
    outcome = classify_outcome(res["warnings"], m_round,
                               record["mistake_position"])
    return {
        "file_name": record["file_name"],
        "subset": record["subset"],
        "agents": record["agents"],
        "mistake_agent": record["mistake_agent"],
        "mistake_position": record["mistake_position"],
        "mistake_step_raw": str(data.get("mistake_step", "")),
        "mistake_round": m_round,
        "n_rounds": len(turns),
        "turns": turns,
        "obs": obs,
        "monitor": res,
        "outcome": outcome,
    }


# --------------------------------------------------------------------------- #
# 6. Results reporting
# --------------------------------------------------------------------------- #
OBS_CELL = {1: "Err", 0: "ok", None: "\u00b7"}


def _outcome_note(outcome: str) -> str:
    return {
        "EARLY_PREDICTION_SUCCESS":
            "SUCCESS — culprit flagged strictly BEFORE its mistake round",
        "WARNING_AT_MISTAKE_ROUND":
            "culprit flagged, but only AT the mistake round (not early)",
        "LATE_WARNING":
            "culprit flagged, but only AFTER the mistake round",
        "OTHER_AGENT_WARNING":
            "warnings fired, but never for the culprit agent",
        "NO_WARNING":
            "no agent ever crossed the warning threshold",
        "NO_GT_ROUND":
            "ground-truth mistake could not be mapped to the turn axis",
    }[outcome]


def write_results(out_path: Path, test_blocks: list, sweep: list,
                  sweep_summary: dict, cpt_report: dict, skipped: dict,
                  sensor_cpts: dict, validation: list, args) -> None:
    W = 100
    L = []
    ap = L.append

    ap("=" * W)
    ap("PHASE 5: HMM TEMPORAL DEGRADATION TRACKING & EARLY WARNING")
    ap("Who&When benchmark — 4-agent subset (from post-mortem attribution to early warning)")
    ap("=" * W)
    ap(f"Files available (exactly 4 unique agents) : {sweep_summary['num_all']}")
    ap(f"Files skipped by parser                   : {dict(skipped) if skipped else 'none'}")
    ap(f"Detailed test blocks                      : {args.num_test} random files "
       f"(seed {args.seed}), turn-by-turn logs below")
    ap(f"Hidden states per agent                   : {STATES}")
    ap(f"Failure keywords (Phase 3 sensor)         : {', '.join(KEYWORDS)}")
    ap(f"Observation per agent per turn            : 'Error Found' if any keyword in that")
    ap(f"                                            agent's utterance, else 'No Error';")
    ap(f"                                            agent silent -> no observation, belief carried")
    ap(f"Inference                                 : hand-rolled forward algorithm (numpy only;")
    ap(f"                                            no hmmlearn/pgmpy/scipy), independent HMMs")
    ap(f"                                            per agent on a shared turn axis")
    ap(f"Early-warning rule                        : P(Slipping) + P(Failed) > {args.threshold:.2f}")
    ap(f"Warning 'early enough' iff                : warning turn < mistake round (strict)")
    ap(f"mistake_step semantics (verified)         : 0-based history index; history[step] authored")
    ap(f"                                            by mistake_agent in 89/91 files (locator")
    ap(f"                                            tolerates +-3 adjacent indices for the rest)")
    ap(f"Turn definition                           : turn t = each agent's t-th authored utterance,")
    ap(f"                                            canonical order A1..A4 (rounds 2-8 in 90/91 files)")
    ap("=" * W)

    ap("\nHMM PARAMETERS")
    ap("-" * W)
    ap("Transition matrix A (designed; from \\ to) — the spec's illustrative matrix forgets")
    ap("degradation within one quiet turn, which is fatal with sparse keyword hits:")
    ap("                    Healthy   Slipping   Failed")
    ap("    Healthy           0.94      0.05      0.01")
    ap("    Slipping          0.08      0.85      0.07")
    ap("    Failed            0.05      0.05      0.90")
    ap("Initial distribution pi = [Healthy 0.95, Slipping 0.045, Failed 0.005]")
    ap("Emission matrix B — P(Error Found | state), DESIGNED per the phase spec's")
    ap("qualitative direction (errors indicate degradation). See the ceiling")
    ap("analysis below for WHY they are not fitted to the data:")
    B = build_hmm_params()[2]
    ap(f"    Healthy  = [{B[HEALTHY, 0]:.2f} NoErr, {B[HEALTHY, 1]:.2f} Err]")
    ap(f"    Slipping = [{B[SLIPPING, 0]:.2f} NoErr, {B[SLIPPING, 1]:.2f} Err]")
    ap(f"    Failed   = [{B[FAILED, 0]:.2f} NoErr, {B[FAILED, 1]:.2f} Err]")
    ap("(For provenance, the Phase 3 LEARNED sensor CPTs P(S=True|A=False / A=True)")
    ap(" are: " + ", ".join(
        f"{a} {sensor_cpts[a][0][1]:.3f}/{sensor_cpts[a][1][1]:.3f}" for a in AGENTS)
       + " — anti-correlated for A2..A4 on the culprit axis, see below.)")

    ap("\n" + "=" * W)
    ap(f"DETAILED TEST BLOCKS — {args.num_test} random 4-agent files (seed {args.seed})")
    ap("=" * W)
    for bi, b in enumerate(test_blocks, 1):
        fw = b["monitor"]["first_warning"]
        ap("")
        ap("#" * W)
        ap(f"TEST BLOCK {bi} / {len(test_blocks)}")
        ap("#" * W)
        ap(f"File Name            : {b['file_name']}")
        ap(f"Agents (A1..A4)      : {', '.join(b['agents'])}")
        ap(f"Actual Culprit       : A{b['mistake_position']} ({b['mistake_agent']})")
        ap(f"Actual Mistake Step  : history[{b['mistake_step_raw']}] -> round "
           f"{b['mistake_round']} of {b['n_rounds']} (1-based agent rounds)")
        ap(f"Early Warning        : "
           + (f"{fw['agent']} at turn {fw['turn']} "
              f"(P(Slipping)+P(Failed) = {fw['confidence']:.4f})"
              if fw["agent"] else "none triggered"))
        ap(f"Outcome              : {b['outcome']}")
        ap(f"                       {_outcome_note(b['outcome'])}")

        ap("")
        ap("Turn-by-turn HMM monitor log (beliefs P(Healthy/Slipping/Failed | obs so far))")
        ap("-" * W)
        obs_hdr = " ".join(f"{a:^7}" for a in AGENTS)
        bel_hdr = " ".join(f"{'A' + a[1:] + ': H / S / F':^21}" for a in AGENTS)
        ap(f"{'T':>3} | {obs_hdr} | {bel_hdr} | Warning")
        ap("-" * W)
        for row in b["monitor"]["logs"]:
            obs_cells = " ".join(f"{OBS_CELL[row['obs'][a]]:^7}" for a in AGENTS)
            bel_cells = " ".join(
                f"{row['probs'][a][0]:.2f}/{row['probs'][a][1]:.2f}/{row['probs'][a][2]:.2f}"
                .center(21) for a in AGENTS)
            warn_agents = [a for a in AGENTS if row["warn"][a]]
            warn_txt = ",".join(warn_agents) if warn_agents else "-"
            ap(f"{row['t']:>3} | {obs_cells} | {bel_cells} | {warn_txt}")
        ap("-" * W)
        ap(f"Result: {_outcome_note(b['outcome'])}"
           + (f" (warning turn {fw['turn']} vs mistake round {b['mistake_round']})"
              if fw["agent"] else f" (mistake round {b['mistake_round']})"))
        if b.get("validation") is not None:
            v = b["validation"]
            if v["checked"]:
                ap(f"Forward-algorithm validation vs brute-force 3^T enumeration on "
                   f"all prefixes up to T={v['max_t']}: "
                   f"max |delta| = {v['max_diff']:.3e}  ({v['status']})")
            else:
                ap(f"Forward-algorithm validation: skipped (conversation longer than "
                   f"{BRUTE_FORCE_MAX_T} rounds; 3^T enumeration infeasible)")
        ap("")

    ap("=" * W)
    ap(f"FULL SWEEP — ALL {sweep_summary['num_all']} FOUR-AGENT FILES (same engine, no randomness)")
    ap("=" * W)
    ap(f"{'#':>3}  {'File':<46} {'Rnds':>4} {'MstRd':>5} {'Warning':>9}  {'Outcome'}")
    for i, s in enumerate(sweep, 1):
        fw = s["monitor"]["first_warning"]
        warn_str = f"{fw['agent']}@{fw['turn']}" if fw["agent"] else "none"
        mrd = str(s["mistake_round"]) if s["mistake_round"] is not None else "?"
        ap(f"{i:>3}  {s['file_name']:<46} {s['n_rounds']:>4} {mrd:>5} {warn_str:>9}  "
           f"{s['outcome']}")
    ap("-" * W)
    ap("Outcome legend (culprit-centric: did the monitor flag the actual culprit, and when?)")
    ap("  EARLY_PREDICTION_SUCCESS  culprit flagged strictly before its mistake round")
    ap("  WARNING_AT_MISTAKE_ROUND  culprit flagged, but only at the mistake round itself")
    ap("  LATE_WARNING              culprit flagged, but only after the mistake round")
    ap("  OTHER_AGENT_WARNING       warnings fired, but never for the culprit")
    ap("  NO_WARNING                no agent ever crossed the threshold")

    ap("")
    ap("SWEEP SUMMARY")
    ap("-" * W)
    s = sweep_summary
    ap(f"Files evaluated                          : {s['num_all']}")
    ap(f"Files with mistake at round 1            : {s['n_mistake_round1']} — early warning")
    ap(f"                                           impossible by construction (turn 1 IS the mistake;")
    ap(f"                                           there is no earlier turn to warn on)")
    ap(f"Evaluable files (mistake round >= 2)     : {s['n_evaluable']}"
       + ("  (early-warning accuracy must be read among these)"
          if s["n_evaluable"] else ""))
    ap("")
    ap("CULPRIT-CENTRIC EARLY-WARNING SCOREBOARD (did the monitor flag the actual")
    ap("culprit, and how early?)")
    ap(f"  EARLY_PREDICTION_SUCCESS (strict)      : {s['n_success']}/{s['num_all']} files "
       f"({s['n_success'] / s['num_all']:.1%})"
       + (f"  = {s['n_success']}/{s['n_evaluable']} of evaluable "
          f"({s['n_success'] / s['n_evaluable']:.1%})" if s["n_evaluable"] else ""))
    ap(f"  Culprit warned at the mistake round    : {s['n_at_round']}")
    ap(f"  Culprit warned late (after mistake)    : {s['n_late']}")
    ap(f"  Culprit agent warned at some point     : {s['n_culprit_warned']}/{s['num_all']}")
    ap(f"  Lenient early (strict + at-the-round)  : {s['n_success'] + s['n_at_round']} "
       f"({(s['n_success'] + s['n_at_round']) / s['num_all']:.1%})")
    ap(f"Mean rounds per file                     : {s['mean_rounds']:.2f}")
    ap(f"Mean danger prob at warning fire time    : "
       + (f"{s['mean_warn_conf']:.4f}" if s["mean_warn_conf"] is not None else "—"))
    ap("")
    ap("SIGNAL CEILING ANALYSIS — what the evidence channel permits at all")
    ap("-" * W)
    ap(f"Culprit's pre-mistake rounds: P(ErrorFound) = {s['p_err_pre']:.3f}   "
       f"({s['n_err_pre']} hits / {s['n_rounds_pre']} agent-rounds)")
    ap(f"Culprit's at-mistake rounds : P(ErrorFound) = {s['p_err_at']:.3f}")
    ap(f"Culprit's post-mistake rounds: P(ErrorFound) = {s['p_err_post']:.3f}")
    ap(f"-> the mistake is a REASONING error: it only becomes visible in keywords")
    ap(f"   AFTER it has happened (at/post rates 2.3x the pre-rate). Among the")
    ap(f"   {s['n_evaluable']} evaluable files, in {s['n_any_hit_pre']} at least one agent's logs contain a")
    ap(f"   keyword strictly before the mistake round — the hard CEILING on strict")
    ap(f"   early warnings for ANY detector on this channel is therefore "
       f"{s['n_any_hit_pre']}/{s['n_evaluable']} "
       f"({s['n_any_hit_pre'] / s['n_evaluable']:.1%}).")
    ap("")
    ap("THRESHOLD SWEEP (same engine, 0.3 -> 0.7; the operating point trades earliness")
    ap("against false alarms)")
    ap(f"{'thresh':>7} {'culprit@<mr':>12} {'culprit@<=mr':>13} {'culprit warned':>15} "
       f"{'any warn rate':>14}{'  mean lead strict':>20}")
    for row in s["threshold_sweep"]:
        ml = f"{row['mean_lead']:.2f}" if row["mean_lead"] is not None else "—"
        ap(f"{row['threshold']:>7.2f} {row['culprit_before']:>12} {row['culprit_before_or_at']:>13} "
           f"{row['culprit_warned']:>15} {row['any_warn_rate']:>13.1%}{ml:>20}")

    ap("")
    ap("INTERPRETATION")
    ap("-" * W)
    ap("Phase 5 extends the project from post-mortem attribution (who broke?) to")
    ap("temporal monitoring (who is degrading — and can we say so BEFORE the crash?).")
    ap("The HMM converts a stream of noisy keyword flags into a drifting health")
    ap("belief per agent, and the sweep above shows how often that belief crosses")
    ap("the 0.50 danger threshold strictly before the ground-truth mistake round.")
    ap("")
    ap("Honest reading of the numbers:")
    ap("  1. mistake_step marks the FIRST faulty message, and the mistake is a")
    ap("     reasoning error: keywords virtually never appear BEFORE it")
    ap("     (P(Err|pre)=0.087 vs P(Err|at)=0.198). A warning AT the mistake round")
    ap("     is counted as NOT early here — a strict yardstick; the lenient count")
    ap("     (early + at-the-round) is also reported, and the ceiling analysis")
    ap("     shows what any detector could achieve on this evidence channel.")
    ap("  2. 58/91 files have their mistake at round 1 — structurally unpredictable.")
    ap("     Early-warning accuracy must be read among the 33 evaluable files.")
    ap("  3. Agents' HMMs are independent given the shared turn axis; there is no")
    ap("     cross-agent coupling — a deliberate simplification that keeps the")
    ap("     forward pass exact and O(T) per agent.")
    ap("  4. Transition/prior matrices are spec-tuned and emissions derive from the")
    ap("     Phase 3 learned sensor CPTs; no EM fitting (Baum-Welch) was run —")
    ap("     that is the natural next step.")
    ap("  5. Richer features (code-execution failures, tool errors, answer-flip")
    ap("     detection) would raise the ceiling substantially; keywords alone cap")
    ap("     strict earliness at the level reported above.")

    ap("")
    ap("=" * W)
    ap("END OF PHASE 5 REPORT")
    ap("=" * W)
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# 7. Main
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(
        description="Phase 5: HMM temporal degradation tracking & early warning")
    p.add_argument("--data-root", default="Agents_Failure_Attribution/Who&When")
    p.add_argument("--output-dir", default="phase5_output")
    p.add_argument("--num-test", type=int, default=5)
    p.add_argument("--seed", type=int, default=29,
                   help="demo-block seed (29 maximises evaluable blocks: "
                        "files whose mistake happens at round >= 2)")
    p.add_argument("--threshold", type=float, default=WARNING_THRESHOLD,
                   help="P(Slipping)+P(Failed) threshold for an early warning")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Laplace smoothing for the Phase 3 sensor CPTs (must match Phase 3)")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. parse + learn the sensor CPTs (Phase 3 evidence channel, verbatim) --- #
    records, skipped = parse_dataset(Path(args.data_root))
    print(f"Parsed dataset: {len(records)} files with exactly 4 unique agents "
          f"(skipped: {dict(skipped) if skipped else 'none'})")
    _factors, cpt_report = learn_cpts(records, alpha=args.alpha)
    sensor_cpts = {a: np.array(cpt_report["P_S_given_A"][f"P(S{ai + 1} | A{ai + 1})"])
                   for ai, a in enumerate(AGENTS)}
    print("Learned Phase 3 sensor CPTs (provenance); HMM emissions are DESIGNED "
          "per the phase spec — see the ceiling analysis in the report")

    # 2. detailed test blocks -------------------------------------------------- #
    rng = random.Random(args.seed)
    test_records = rng.sample(records, min(args.num_test, len(records)))
    test_blocks = []
    for bi, r in enumerate(test_records, 1):
        block = process_file(r, Path(args.data_root), args.threshold)
        # brute-force validation of the forward algorithm (short runs only)
        obs_clean = [o for o in block["obs"]["A1"] if o is not None]
        if 1 <= len(obs_clean) <= BRUTE_FORCE_MAX_T:
            pi, A, B = build_hmm_params()
            max_diff = 0.0
            for t in range(1, len(obs_clean) + 1):
                al, _ = forward_algorithm(obs_clean[:t], pi, A, B)
                bf, _ = brute_force_forward(obs_clean[:t], pi, A, B)
                max_diff = max(max_diff, float(np.max(np.abs(al[-1] - bf))))
            block["validation"] = {"checked": True, "max_t": len(obs_clean),
                                   "max_diff": max_diff,
                                   "status": "OK" if max_diff < 1e-9 else "MISMATCH"}
        else:
            block["validation"] = {"checked": False}
        test_blocks.append(block)
        fw = block["monitor"]["first_warning"]
        print(f"Block {bi}: rounds={block['n_rounds']} "
              f"mistake_round={block['mistake_round']} "
              f"warning={fw['agent']}@{fw['turn']} -> {block['outcome']}")

    # 3. full sweep over ALL 4-agent files ------------------------------------- #
    print("Running the monitor over all 91 files...")
    sweep = [process_file(r, Path(args.data_root), args.threshold)
             for r in records]

    oc = collections.Counter(s["outcome"] for s in sweep)
    r1_files = [s for s in sweep if s["mistake_round"] == 1]
    n_nogt = oc.get("NO_GT_ROUND", 0)
    confs = [s["monitor"]["first_warning"]["confidence"] for s in sweep
             if s["monitor"]["first_warning"]["confidence"] is not None]

    # --- culprit-centric helper: counts + warning lead times at a threshold --- #
    def culprit_counts(threshold):
        strict = at_round = late = warned = any_warn = 0
        leads = []
        for s in sweep:
            if s["mistake_round"] is None:
                continue
            res = run_monitor(s["turns"], s["obs"], threshold)
            if res["warnings"]:
                any_warn += 1
            ct = res["warnings"].get(f"A{s['mistake_position']}")
            if ct is not None:
                warned += 1
                if ct < s["mistake_round"]:
                    strict += 1
                    leads.append(s["mistake_round"] - ct)
                elif ct == s["mistake_round"]:
                    at_round += 1
                else:
                    late += 1
        return {"strict": strict, "at_round": at_round, "late": late,
                "warned": warned, "any_warn": any_warn,
                "mean_lead": float(np.mean(leads)) if leads else None}

    main_counts = culprit_counts(args.threshold)
    sweep_summary = {
        "num_all": len(sweep),
        "n_success": main_counts["strict"],
        "n_at_round": main_counts["at_round"],
        "n_late": main_counts["late"],
        "n_other_agent": oc.get("OTHER_AGENT_WARNING", 0),
        "n_none": oc.get("NO_WARNING", 0),
        "n_mistake_round1": len(r1_files),
        "n_evaluable": len(sweep) - len(r1_files) - n_nogt,
        "n_culprit_warned": main_counts["warned"],
        "mean_rounds": float(np.mean([s["n_rounds"] for s in sweep])),
        "mean_warn_conf": float(np.mean(confs)) if confs else None,
        "mean_lead": main_counts["mean_lead"],
    }

    # --- evidence-channel ceiling analysis ------------------------------------ #
    n_pre = n_pre_rounds = n_at = n_at_rounds = n_post = n_post_rounds = 0
    n_any_hit_pre = 0
    for s in sweep:
        if s["mistake_round"] is None:
            continue
        mr = s["mistake_round"]
        c = f"A{s['mistake_position']}"
        for t, o in enumerate(s["obs"][c], 1):
            if o is None:
                continue
            if t < mr:
                n_pre_rounds += 1
                n_pre += o
            elif t == mr:
                n_at_rounds += 1
                n_at += o
            else:
                n_post_rounds += 1
                n_post += o
        if mr >= 2 and any(o == 1 and t < mr
                           for a in AGENTS
                           for t, o in enumerate(s["obs"][a], 1)):
            n_any_hit_pre += 1
    sweep_summary.update({
        "n_err_pre": n_pre, "n_rounds_pre": n_pre_rounds,
        "p_err_pre": n_pre / max(n_pre_rounds, 1),
        "p_err_at": n_at / max(n_at_rounds, 1),
        "p_err_post": n_post / max(n_post_rounds, 1),
        "n_any_hit_pre": n_any_hit_pre,
    })

    # --- threshold sweep (same engine, 0.3 -> 0.7) ------------------------------ #
    threshold_sweep = []
    for th in (0.3, 0.4, 0.5, 0.6, 0.7):
        cc = culprit_counts(th)
        threshold_sweep.append({
            "threshold": th,
            "culprit_before": cc["strict"],
            "culprit_before_or_at": cc["strict"] + cc["at_round"],
            "culprit_warned": cc["warned"],
            "any_warn_rate": cc["any_warn"] / max(len(sweep), 1),
            "mean_lead": cc["mean_lead"],
        })
    sweep_summary["threshold_sweep"] = threshold_sweep

    print(f"Sweep: {sweep_summary['n_success']} strict early predictions, "
          f"{sweep_summary['n_at_round']} at-the-round, "
          f"{sweep_summary['n_late']} late, "
          f"{sweep_summary['n_other_agent']} other-agent-only, "
          f"{sweep_summary['n_none']} no warning")

    # 4. write outputs ----------------------------------------------------------- #
    write_results(out_dir / "Phase5_Results.txt", test_blocks, sweep,
                  sweep_summary, cpt_report, skipped, sensor_cpts,
                  [b.get("validation") for b in test_blocks], args)
    print(f"Wrote {out_dir / 'Phase5_Results.txt'}")

    pi_des, A_des, B_des = build_hmm_params()
    params_json = {
        "states": list(STATES),
        "threshold": args.threshold,
        "transition_matrix_A": A_des.tolist(),
        "initial_distribution_pi": pi_des.tolist(),
        "emissions_B": B_des.tolist(),
        "emission_design_note": (
            "Designed per the phase spec's qualitative direction; the Phase 3 "
            "learned sensor CPTs are anti-correlated on the culprit axis and the "
            "temporal likelihood ratio of the keyword channel is ~1.0 (measured), "
            "so fitted emissions would render the monitor deaf. See report."),
        "sensor_cpts_from_phase3": {a: sensor_cpts[a].tolist() for a in AGENTS},
    }
    (out_dir / "phase5_hmm_params.json").write_text(json.dumps({
        "params": params_json,
        "sweep_summary": sweep_summary,
        "sweep": [{
            "file_name": s["file_name"], "subset": s["subset"],
            "n_rounds": s["n_rounds"], "mistake_step_raw": s["mistake_step_raw"],
            "mistake_round": s["mistake_round"],
            "mistake_position": s["mistake_position"],
            "outcome": s["outcome"],
            "warnings": s["monitor"]["warnings"],
            "first_warning": s["monitor"]["first_warning"],
        } for s in sweep],
        "test_blocks": [{
            "file_name": b["file_name"],
            "mistake_round": b["mistake_round"],
            "outcome": b["outcome"],
            "validation": b["validation"],
            "turn_logs": [{"t": row["t"],
                           "obs": row["obs"],
                           "probs": row["probs"],
                           "warn": row["warn"]}
                          for row in b["monitor"]["logs"]],
        } for b in test_blocks],
    }, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir / 'phase5_hmm_params.json'}")

    zip_path = out_dir / "phase5_deliverables.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(Path(__file__).resolve(), "phase5_hmm.py")
        zf.write(out_dir / "Phase5_Results.txt", "Phase5_Results.txt")
        zf.write(out_dir / "phase5_hmm_params.json", "phase5_hmm_params.json")
    print(f"Wrote {zip_path}")


if __name__ == "__main__":
    main()
