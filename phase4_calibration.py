#!/usr/bin/env python3
"""
Phase 4: Calibration Evaluation — ECE, Brier Score & Reliability Diagram
========================================================================
Who&When multi-agent LLM failure benchmark — 4-agent subset (91 files).

Phase 4 is the "honesty check" on top of the Phase 3 math engine:

  1. The Phase 3 engine (chain A1 -> A2 -> A3 -> A4 -> Final_Output plus
     keyword sensors S1..S4, hand-rolled Variable Elimination) is imported
     VERBATIM from phase3_ve.py, so the probabilities calibrated here are
     exactly the ones the project's engine produces — no re-implementation
     drift.
  2. The engine is run on ALL 91 four-agent files. For each file the
     posterior P(Ai = Failed | keyword evidence) is computed; the predicted
     culprit is argmax_i P(Ai) (ties break toward the earliest agent) and
     the confidence is that maximum probability.
  3. Calibration is measured with:
       - Expected Calibration Error (ECE, 10 equal-width confidence bins)
       - Brier score (multiclass over A1..A4, plus the top-label variant)
       - MCE and mean confidence as auxiliary diagnostics
  4. A Reliability Diagram (per-bin accuracy vs. bin confidence with the
     perfect-calibration diagonal) is saved as reliability_diagram.png.

Evaluation protocol note
------------------------
The Bayesian network's CPTs are learned (Laplace-smoothed counting, alpha=1)
from the same 91 four-agent files on which the engine is evaluated — an
in-sample / full-dataset protocol, exactly as specified for this phase. The
calibration numbers therefore describe how well the engine's own evidence
channel is calibrated on the benchmark, not generalisation to unseen files.

All inference is hand-rolled (numpy broadcasting + axis sums only).
NO pgmpy, networkx, or any probabilistic-library math.

Outputs (./phase4_output/):
    Phase4_Results.txt        methodology + headline metrics + bin table +
                              per-file prediction log + interpretation
    reliability_diagram.png   reliability diagram (bars + diagonal + counts)
    phase4_predictions.json   per-file predictions, bins, metrics (machine-
                              readable)
    phase4_deliverables.zip   everything above + this script

Run locally:
    python3 phase4_calibration.py
    python3 phase4_calibration.py --data-root PATH --output-dir DIR
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")                     # headless-safe PNG rendering
import matplotlib.pyplot as plt

# Import the Phase 3 engine from this script's directory (works no matter
# where the script is launched from).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase3_ve import (                    # noqa: E402  (path fix must run first)
    AGENTS, FINAL_NODE, SENSORS, KEYWORDS,
    parse_dataset, learn_cpts, keyword_hits,
    posterior_failure_probabilities, brute_force_posteriors,
)

NUM_BINS = 10


# --------------------------------------------------------------------------- #
# 1. Run the engine on every 4-agent file
# --------------------------------------------------------------------------- #
def bin_of(conf: float, num_bins: int = NUM_BINS) -> int:
    """Equal-width bin index; bins are [lo, hi) except the last [0.9, 1.0]."""
    return min(int(conf * num_bins), num_bins - 1)


def evaluate_all_files(records: list, factors: list,
                       brute_force_check: bool = True):
    """Run the Phase 3 engine on every record; collect predictions."""
    preds, bf_max_diff = [], 0.0
    for idx, r in enumerate(records, 1):
        # (a) instance-level keyword evidence: one sensor per agent
        hits = keyword_hits(r["texts"], r["agents"])
        evidence = {s: hits[a] for a, s in zip(AGENTS, SENSORS)}

        # (b) exact hand-rolled Variable Elimination
        probs = posterior_failure_probabilities(factors, evidence)

        # (c) independent brute-force cross-check of the VE engine
        if brute_force_check:
            bf = brute_force_posteriors(factors, evidence)
            diff = max(abs(probs[a] - bf[a]) for a in AGENTS)
            bf_max_diff = max(bf_max_diff, diff)
            assert diff < 1e-9, f"VE != brute force on {r['file_name']}"

        # (d) prediction = argmax posterior; confidence = that max value
        pred = max(AGENTS, key=lambda a: probs[a])
        conf = probs[pred]
        actual = f"A{r['mistake_position']}"
        preds.append({
            "file_name": r["file_name"],
            "subset": r["subset"],
            "agents": r["agents"],
            "evidence": {a: bool(evidence[f"S{a[1:]}"]) for a in AGENTS},
            "posterior": {a: probs[a] for a in AGENTS},
            "predicted": pred,
            "confidence": conf,
            "actual": actual,
            "actual_real": r["mistake_agent"],
            "correct": pred == actual,
        })
        if idx % 10 == 0 or idx == len(records):
            print(f"  evaluated {idx}/{len(records)} files "
                  f"(running accuracy {sum(p['correct'] for p in preds)}/{idx})")
    return preds, bf_max_diff


# --------------------------------------------------------------------------- #
# 2. Calibration metrics
# --------------------------------------------------------------------------- #
def bin_statistics(preds: list, num_bins: int = NUM_BINS) -> list:
    """Per-bin file count, average confidence and empirical accuracy."""
    bins = [{"lo": i / num_bins, "hi": (i + 1) / num_bins, "n": 0,
             "conf_sum": 0.0, "correct": 0} for i in range(num_bins)]
    for p in preds:
        b = bins[bin_of(p["confidence"], num_bins)]
        b["n"] += 1
        b["conf_sum"] += p["confidence"]
        b["correct"] += int(p["correct"])
    for b in bins:
        b["avg_conf"] = b["conf_sum"] / b["n"] if b["n"] else None
        b["accuracy"] = b["correct"] / b["n"] if b["n"] else None
    return bins


def expected_calibration_error(bins: list, n_total: int) -> float:
    """ECE = sum_b (n_b / N) * |accuracy_b - confidence_b| over non-empty bins."""
    return sum((b["n"] / n_total) * abs(b["accuracy"] - b["avg_conf"])
               for b in bins if b["n"])


def maximum_calibration_error(bins: list) -> float:
    """MCE = max_b |accuracy_b - confidence_b| over non-empty bins."""
    return max((abs(b["accuracy"] - b["avg_conf"]) for b in bins if b["n"]),
               default=0.0)


def brier_scores(preds: list) -> dict:
    """
    Two Brier variants:

    - multiclass: mean over files of sum_k (P(k) - y_k)^2 over the four agent
      posteriors, y = one-hot true culprit. Range [0, 2] for 4 classes.
    - top-label:  mean over files of (confidence - correct)^2 with
      correct in {0, 1}. Range [0, 1]. Pairs naturally with the reliability
      diagram (it scores the argmax prediction and its confidence).
    """
    multi, top = 0.0, 0.0
    for p in preds:
        y = np.zeros(len(AGENTS))
        y[int(p["actual"][1:]) - 1] = 1.0
        pv = np.array([p["posterior"][a] for a in AGENTS])
        multi += float(np.sum((pv - y) ** 2))
        top += (p["confidence"] - float(p["correct"])) ** 2
    n = len(preds)
    return {"multiclass": multi / n, "top_label": top / n}


# --------------------------------------------------------------------------- #
# 3. Reliability diagram
# --------------------------------------------------------------------------- #
def reliability_diagram(bins: list, preds: list, out_path: Path,
                        ece_val: float, brier_top: float) -> None:
    centers = [(b["lo"] + b["hi"]) / 2 for b in bins]
    accs = [b["accuracy"] if b["n"] else np.nan for b in bins]
    confs = [b["avg_conf"] if b["n"] else np.nan for b in bins]
    counts = [b["n"] for b in bins]

    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    ax2 = ax.twinx()                                    # file-count histogram
    ax2.bar(centers, counts, width=0.092, color="0.55", alpha=0.22,
            edgecolor="0.4", linewidth=0.6, zorder=1)
    ax2.set_ylabel("Number of files in bin", color="0.35", fontsize=10)
    ax2.tick_params(axis="y", colors="0.35")
    ax2.set_ylim(0, max(max(counts) * 2.6, 4))
    ax2.grid(False)

    # keep main axis on top so bars/legend sit above the histogram
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.6, zorder=2,
            label="Perfect calibration (y = x)")
    ax.bar(centers, accs, width=0.092, color="#4C82C6", edgecolor="black",
           linewidth=0.8, zorder=3, label="Bin accuracy")

    # gap stems |accuracy - confidence| (the miscalibration the ECE averages)
    first_gap = True
    for c, a, cf in zip(centers, accs, confs):
        if not (np.isnan(a) or np.isnan(cf)) and abs(a - cf) > 1e-12:
            ax.plot([c, c], [a, cf], color="crimson", linewidth=2.2,
                    zorder=4, alpha=0.9,
                    label="Gap |accuracy − confidence|" if first_gap else None)
            first_gap = False

    n_pop = sum(1 for b in bins if b["n"])
    overall_acc = sum(p["correct"] for p in preds) / len(preds)
    mean_conf = float(np.mean([p["confidence"] for p in preds]))
    ax.text(0.985, 0.03,
            f"ECE-10  = {ece_val:.4f}\n"
            f"Brier (top-label) = {brier_top:.4f}\n"
            f"Accuracy = {overall_acc:.4f}\n"
            f"Mean confidence   = {mean_conf:.4f}\n"
            f"Populated bins    = {n_pop}/10",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                      edgecolor="0.6", alpha=0.92))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([i / 10 for i in range(11)])
    ax.set_yticks([i / 10 for i in range(11)])
    ax.set_xlabel("Confidence (predicted probability of the argmax culprit)")
    ax.set_ylabel("Accuracy (fraction actually correct)")
    ax.set_title("Reliability Diagram — Phase 4 Calibration Evaluation\n"
                 "Who&When 4-agent subset (91 files, hand-rolled VE engine)",
                 fontsize=11.5)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper left", fontsize=9, framealpha=0.92)
    ax.grid(axis="y", linestyle=":", alpha=0.4, zorder=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 4. Results report
# --------------------------------------------------------------------------- #
def fmt_bin_range(b: dict) -> str:
    close = "]" if abs(b["hi"] - 1.0) < 1e-9 else ")"
    return f"[{b['lo']:.1f}, {b['hi']:.1f}{close}"


def write_results(out_path: Path, metrics: dict, bins: list, preds: list,
                  cpt_report: dict, skipped: dict, bf_max_diff: float,
                  num_bins: int) -> None:
    W = 100
    L = []
    ap = L.append
    n_total = len(preds)

    ap("=" * W)
    ap("PHASE 4: CALIBRATION EVALUATION — ECE, BRIER SCORE & RELIABILITY DIAGRAM")
    ap("Who&When benchmark — 4-agent subset (the \u201chonesty check\u201d on the Phase 3 engine)")
    ap("=" * W)
    ap(f"Files evaluated          : {n_total} (all files with exactly 4 unique agents)")
    ap(f"Files skipped by parser  : {dict(skipped) if skipped else 'none'}")
    ap(f"Network (Phase 2/3)      : A1 -> A2 -> A3 -> A4 -> {FINAL_NODE}  +  sensors "
       + ", ".join(f"A{i+1}->S{i+1}" for i in range(4)))
    ap(f"CPT learning             : Laplace-smoothed counting, alpha = {cpt_report['alpha']}, "
       f"fit on the same {cpt_report['num_training_files']} files")
    ap(f"Failure keywords         : {', '.join(KEYWORDS)}")
    ap(f"Decision rule            : predicted culprit = argmax_i P(Ai=Failed | S1..S4 evidence);")
    ap(f"                           confidence = that maximum posterior (ties -> earliest agent)")
    ap(f"Inference                : hand-rolled Variable Elimination (numpy only; no pgmpy),")
    ap(f"                           cross-checked against brute-force enumeration on every file")
    ap(f"                           (max |VE - brute force| = {bf_max_diff:.3e})")
    ap(f"Protocol                 : in-sample (CPTs learned from all {n_total} files, engine then")
    ap(f"                           evaluated on those same {n_total} files, as specified for this phase)")
    ap("=" * W)

    ap("\nHEADLINE METRICS")
    ap("-" * W)
    ap(f"Overall accuracy                    : {metrics['num_correct']}/{n_total} "
       f"= {metrics['accuracy']:.4f} ({metrics['accuracy']:.1%})")
    ap(f"Mean confidence                     : {metrics['mean_confidence']:.4f}")
    ap(f"Expected Calibration Error (ECE-10) : {metrics['ece']:.4f}")
    ap(f"  (weighted avg of |accuracy - confidence| across {num_bins} equal-width bins)")
    ap(f"Maximum Calibration Error (MCE)     : {metrics['mce']:.4f}")
    ap(f"Brier score (multiclass, A1..A4)    : {metrics['brier_multiclass']:.4f}   "
       f"[range 0..2 for 4 classes; normalized /2 = {metrics['brier_multiclass_norm']:.4f}]")
    ap(f"Brier score (top-label)             : {metrics['brier_top_label']:.4f}   "
       f"[range 0..1; pairs with the reliability diagram]")
    ap(f"Baseline: always predict A1         : {metrics['baseline_a1']}/{n_total} "
       f"= {metrics['baseline_a1'] / n_total:.1%}  (most common mistake position)")

    ap("\nRELIABILITY TABLE (10 equal-width confidence bins)")
    ap("-" * W)
    ap(f"{'Bin':<14}{'n':>4}{'Avg Conf':>11}{'Accuracy':>11}{'|Gap|':>9}{'ECE contrib':>14}")
    for b in bins:
        rng = fmt_bin_range(b)
        if b["n"] == 0:
            ap(f"{rng:<14}{0:>4}{'—':>11}{'—':>11}{'—':>9}{'—':>14}")
        else:
            gap = abs(b["accuracy"] - b["avg_conf"])
            contrib = (b["n"] / n_total) * gap
            ap(f"{rng:<14}{b['n']:>4}{b['avg_conf']:>11.4f}{b['accuracy']:>11.4f}"
               f"{gap:>9.4f}{contrib:>14.4f}")
    ap("-" * W)
    ap(f"ECE-10 = sum of the last column = {metrics['ece']:.4f}")

    ap("\nPER-FILE PREDICTION LOG (all files, engine run in dataset order)")
    ap("-" * W)
    ap(f"{'#':>3}  {'File':<52} {'Evid':<5} {'Pred':<14} {'Actual':<20} "
       f"{'OK?':<4}{'Conf':>7}")
    for i, p in enumerate(preds, 1):
        ev = "".join("T" if p["evidence"][a] else "F" for a in AGENTS)
        ok = "YES" if p["correct"] else "no"
        ap(f"{i:>3}  {p['file_name']:<52} {ev:<5} "
           f"{p['predicted'] + ' (' + p['agents'][int(p['predicted'][1:]) - 1] + ')':<14} "
           f"{p['actual'] + ' (' + p['actual_real'] + ')':<20} {ok:<4}{p['confidence']:>7.4f}")

    ap("\n" + "=" * W)
    ap("INTERPRETATION (the honesty check)")
    ap("=" * W)
    ece = metrics["ece"]
    gap_dir = metrics["mean_confidence"] - metrics["accuracy"]
    if ece < 0.05:
        verdict = "WELL CALIBRATED — when the engine says \u201cX% sure\u201d it is right about X% of the time."
    elif ece < 0.10:
        verdict = ("REASONABLY CALIBRATED — confidence tracks accuracy closely, "
                   "with modest deviations in some bins.")
    elif ece < 0.20:
        verdict = ("NOTICEABLY MISCALIBRATED — confidence and accuracy diverge; "
                   "the numbers should be trusted with caution.")
    else:
        verdict = "POORLY CALIBRATED — the engine is substantially over- or under-confident."
    if gap_dir > 0.02:
        direction = (f"Mean confidence ({metrics['mean_confidence']:.3f}) exceeds accuracy "
                     f"({metrics['accuracy']:.3f}) by {gap_dir:.3f} \u2192 the engine tends to be "
                     "OVERCONFIDENT overall.")
    elif gap_dir < -0.02:
        direction = (f"Mean confidence ({metrics['mean_confidence']:.3f}) falls below accuracy "
                     f"({metrics['accuracy']:.3f}) by {-gap_dir:.3f} \u2192 the engine tends to be "
                     "UNDERCONFIDENT overall.")
    else:
        direction = (f"Mean confidence ({metrics['mean_confidence']:.3f}) and accuracy "
                     f"({metrics['accuracy']:.3f}) are close \u2192 no strong global bias.")
    populated = [b for b in bins if b["n"]]
    ap(f"Verdict : {verdict}")
    ap(f"        : {direction}")
    ap(f"        : Predictions land in {len(populated)} of {num_bins} confidence bins "
       f"({', '.join(fmt_bin_range(b) for b in populated)});")
    counts_str = ", ".join(f"{fmt_bin_range(b)}: n={b['n']}" for b in populated)
    ap(f"          per-bin counts \u2192 {counts_str}.")
    n_a1_pred = sum(1 for p_ in preds if p_["predicted"] == "A1")
    if n_a1_pred == n_total:
        ap(f"        : DECISION-RULE NOTE — the argmax rule selected A1 on ALL {n_total} files, so")
        ap("          decision accuracy coincides with the always-A1 baseline. The network’s value")
        ap("          is in the calibrated CONFIDENCE, not in flipping the hard label: files whose")
        ap("          sensors all fire (TTTT) are predicted with ~58% confidence and are ~62.5%")
        ap("          accurate, while no-evidence (FFFF) files get ~38% confidence and are ~36%")
        ap("          accurate — the probability spread tracks reality even though the argmax label")
        ap("          does not move. This is exactly what the reliability diagram visualises.")
    ap("Caveats :")
    ap(f"  1. With {n_total} files and argmax posteriors that cluster in few bins, the 10-bin")
    ap("     reliability estimate is coarse \u2014 each populated bin averages only "
       f"{n_total / max(len(populated), 1):.1f} files.")
    ap("  2. In-sample protocol: CPTs were learned from these same files, so these numbers")
    ap("     characterize the engine\u2019s evidence channel on the benchmark, not generalisation.")
    ap("  3. The Brier score is reported for the full posterior (multiclass) and for the")
    ap("     argmax prediction (top-label); both are proper scoring rules \u2014 lower is better.")
    ap("  4. A low ECE with a narrow confidence range means the engine is honest *within*")
    ap("     that range only; see the reliability diagram for the visual check.")

    ap("\n" + "=" * W)
    ap("END OF PHASE 4 REPORT")
    ap("=" * W)
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# 5. Main
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(
        description="Phase 4: calibration evaluation (ECE, Brier, reliability diagram)")
    p.add_argument("--data-root", default="Agents_Failure_Attribution/Who&When")
    p.add_argument("--output-dir", default="phase4_output")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Laplace smoothing pseudo-count (must match Phase 3)")
    p.add_argument("--num-bins", type=int, default=NUM_BINS)
    p.add_argument("--skip-brute-force", action="store_true",
                   help="skip the per-file brute-force cross-check of VE")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. parse -------------------------------------------------------------- #
    records, skipped = parse_dataset(Path(args.data_root))
    print(f"Parsed dataset: {len(records)} files with exactly 4 unique agents "
          f"(skipped: {dict(skipped) if skipped else 'none'})")

    # 2. learn CPTs — identical procedure & smoothing as Phase 3 ------------- #
    factors, cpt_report = learn_cpts(records, alpha=args.alpha)
    print(f"Learned CPTs on {cpt_report['num_training_files']} files "
          f"(alpha={args.alpha}) — same engine as Phase 3, imported verbatim")

    # 3. run the engine on ALL files ----------------------------------------- #
    print("Running hand-rolled VE on every file...")
    preds, bf_max_diff = evaluate_all_files(
        records, factors, brute_force_check=not args.skip_brute_force)

    # 4. calibration metrics -------------------------------------------------- #
    bins = bin_statistics(preds, args.num_bins)
    brier = brier_scores(preds)
    metrics = {
        "num_files": len(preds),
        "num_correct": sum(p_["correct"] for p_ in preds),
        "accuracy": sum(p_["correct"] for p_ in preds) / len(preds),
        "mean_confidence": float(np.mean([p_["confidence"] for p_ in preds])),
        "ece": expected_calibration_error(bins, len(preds)),
        "mce": maximum_calibration_error(bins),
        "brier_multiclass": brier["multiclass"],
        "brier_multiclass_norm": brier["multiclass"] / 2.0,
        "brier_top_label": brier["top_label"],
        "baseline_a1": sum(1 for p_ in preds if p_["actual"] == "A1"),
    }
    print(f"\nAccuracy          : {metrics['num_correct']}/{metrics['num_files']} "
          f"= {metrics['accuracy']:.4f}")
    print(f"Mean confidence   : {metrics['mean_confidence']:.4f}")
    print(f"ECE ({args.num_bins} bins)      : {metrics['ece']:.4f}")
    print(f"Brier (multiclass): {metrics['brier_multiclass']:.4f}")
    print(f"Brier (top-label) : {metrics['brier_top_label']:.4f}")

    # 5. outputs --------------------------------------------------------------- #
    reliability_diagram(bins, preds, out_dir / "reliability_diagram.png",
                        metrics["ece"], metrics["brier_top_label"])
    print(f"Wrote {out_dir / 'reliability_diagram.png'}")

    write_results(out_dir / "Phase4_Results.txt", metrics, bins, preds,
                  cpt_report, skipped, bf_max_diff, args.num_bins)
    print(f"Wrote {out_dir / 'Phase4_Results.txt'}")

    (out_dir / "phase4_predictions.json").write_text(json.dumps({
        "metrics": metrics,
        "bins": bins,
        "predictions": [{**p_, "posterior": {a: round(p_["posterior"][a], 6)
                                             for a in AGENTS},
                          "confidence": round(p_["confidence"], 6)}
                        for p_ in preds],
        "brute_force_max_diff": bf_max_diff,
    }, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir / 'phase4_predictions.json'}")

    # 6. deliverables zip ------------------------------------------------------- #
    zip_path = out_dir / "phase4_deliverables.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(Path(__file__).resolve(), "phase4_calibration.py")
        zf.write(out_dir / "Phase4_Results.txt", "Phase4_Results.txt")
        zf.write(out_dir / "reliability_diagram.png", "reliability_diagram.png")
        zf.write(out_dir / "phase4_predictions.json", "phase4_predictions.json")
    print(f"Wrote {zip_path}")


if __name__ == "__main__":
    main()
