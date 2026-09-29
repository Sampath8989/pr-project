#!/usr/bin/env python3
"""Run every AgentDiag v2 phase and write its own evidence-backed outputs."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentdiag_v2.src.bayes import brute_force, fit_model, predict
from agentdiag_v2.src.data import (FEATURES, ROLE_FEATURES, audit, feature_rows, load_records,
                                   split_records, write_json, write_jsonl)
from agentdiag_v2.src.evaluation import (bootstrap_accuracy_interval, calibrate_temperature, draw_reliability,
                                         fit_logistic_baseline, fit_temperature, keyword_baseline, metric_summary,
                                         normalized, predict_logistic_baseline, prior_baseline)
from agentdiag_v2.src.temporal import (brute_force_forward, choose_threshold, fit_hmm, forward,
                                       keyword_monitor, monitor, warning_summary)
from agentdiag_v2.src.triage import packet


HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "data" / "Who&When"


def phase_dir(output: Path, number: int, name: str) -> Path:
    path = output / f"phase{number}_{name}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_predictions(rows: list[dict], models: dict, temperature: float = 1.0) -> list[dict]:
    result = []
    for r in rows:
        model = models[str(r["n_agents"])]
        raw = predict(model, r["evidence"])
        probabilities = calibrate_temperature(raw, temperature)
        if not normalized(probabilities):
            raise AssertionError(f"invalid probabilities for {r['id']}")
        result.append({"id": r["id"], "subset": r["subset"], "n_agents": r["n_agents"],
                       "agents": r["agents"], "actual": r["culprit"],
                       "predicted": max(range(len(probabilities)), key=lambda i: probabilities[i]),
                       "probabilities": probabilities, "raw_probabilities": raw})
    return result


def baseline_predictions(rows: list[dict], training: list[dict], kind: str) -> list[dict]:
    predictions = []
    for r in rows:
        prior = prior_baseline(training, r["n_agents"])
        probabilities = prior if kind == "position_prior" else keyword_baseline(r, prior)
        predictions.append({"id": r["id"], "actual": r["culprit"],
                            "predicted": max(range(len(probabilities)), key=lambda i: probabilities[i]),
                            "probabilities": probabilities})
    return predictions


def save_text(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def metric_line(label: str, metric: dict) -> str:
    return (f"- **{label}:** {metric['correct']}/{metric['n']} correct "
            f"({metric['accuracy']:.1%}); Brier {metric['brier']:.3f}; "
            f"ECE {metric['ece']:.3f}; log loss {metric['log_loss']:.3f}.")


def run(dataset: Path, output: Path, seed: int) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    records, exclusions = load_records(dataset)
    if not records:
        raise RuntimeError("No usable benchmark traces found")
    by_id = {r["id"]: r for r in records}
    paths = {}

    # Phase 0: protocol and group-safe split.
    p0 = phase_dir(output, 0, "protocol")
    splits = split_records(records, seed)
    for part, ids in splits.items():
        if len(ids) != len(set(ids)):
            raise AssertionError("duplicate IDs in split")
    if set(splits["train"]) & set(splits["validation"]) or set(splits["train"]) & set(splits["test"]) or set(splits["validation"]) & set(splits["test"]):
        raise AssertionError("overlapping split IDs")
    group_partition = {}
    for part, ids in splits.items():
        for trace_id in ids:
            qid = by_id[trace_id]["question_id"]
            if qid in group_partition and group_partition[qid] != part:
                raise AssertionError("question ID leaked across splits")
            group_partition[qid] = part
    write_json(p0 / "splits.json", {"seed": seed, "splits": splits})
    split_counts = {part: dict(Counter(len(by_id[i]["agents"]) for i in ids)) for part, ids in splits.items()}
    write_json(p0 / "audit.json", audit(records, exclusions) | {"split_counts_by_agent_count": split_counts})
    save_text(p0 / "Phase0_Results.md", f"# Phase 0 — Frozen protocol\n\n"
              f"- Source: `{dataset}`. Parsed {len(records)} of {len(records)+len(exclusions)} traces.\n"
              f"- Question-group-safe split with seed {seed}: train {len(splits['train'])}, "
              f"validation {len(splits['validation'])}, test {len(splits['test'])}.\n"
              f"- Primary experiment: four-agent traces; counts by split: {split_counts}.\n"
              "- Post-mortem attribution uses full logs. Early warning uses only messages observed so far.\n"
              "- Model/calibration/threshold selection never uses test labels.\n")
    paths["phase0"] = str((p0 / "Phase0_Results.md").relative_to(output))

    # Phase 1: clean records, graph summary.
    p1 = phase_dir(output, 1, "parsing")
    write_jsonl(p1 / "clean_records.jsonl", records)
    write_json(p1 / "label_step_exceptions.json",
               [{"id": r["id"], "mistake_step": r["mistake_step"],
                 "culprit": r["culprit_name"]} for r in records if not r["label_message_matches"]])
    topology = Counter()
    for r in records:
        sequence = [m["agent"] for m in r["messages"]]
        edges = set(zip(sequence, sequence[1:]))
        topology[(len(r["agents"]), len(edges))] += 1
    write_json(p1 / "interaction_graph_summary.json",
               [{"agent_count": a, "directed_edges": e, "trace_count": count}
                for (a, e), count in sorted(topology.items())])
    save_text(p1 / "Phase1_Results.md", "# Phase 1 — Parsing\n\n"
              f"- Parsed: {len(records)}; exclusions: {len(exclusions)}.\n"
              f"- Culprit label matches the agent at annotated raw mistake step in "
              f"{sum(r['label_message_matches'] for r in records)}/{len(records)} traces. "
              "Exceptions are retained and audited.\n"
              f"- Agent counts: {dict(Counter(len(r['agents']) for r in records))}.\n"
              "- Speaker-transition graph counts are descriptive, not claimed causal edges.\n")
    paths["phase1"] = str((p1 / "Phase1_Results.md").relative_to(output))

    # Phase 2: features and honest simple baselines.
    p2 = phase_dir(output, 2, "evidence_baselines")
    rows = feature_rows(records)
    rows_by_id = {r["id"]: r for r in rows}
    train = [rows_by_id[i] for i in splits["train"]]
    val = [rows_by_id[i] for i in splits["validation"]]
    test = [rows_by_id[i] for i in splits["test"]]
    train4, val4, test4 = ([r for r in group if r["n_agents"] == 4] for group in (train, val, test))
    if not train4 or not val4 or not test4:
        raise RuntimeError("four-agent split has an empty partition")
    write_jsonl(p2 / "features.jsonl", rows)
    baseline_metrics = {}
    for kind in ("position_prior", "first_error_keyword"):
        baseline_metrics[kind] = {"validation_4agent": metric_summary(baseline_predictions(val4, train, kind)),
                                  "test_4agent": metric_summary(baseline_predictions(test4, train, kind))}
    logistic_model = fit_logistic_baseline(train, 4)
    logistic_predictions = lambda group: [{"id": r["id"], "actual": r["culprit"],
        "probabilities": (probs := predict_logistic_baseline(logistic_model, r)),
        "predicted": max(range(len(probs)), key=lambda i: probs[i])} for r in group]
    baseline_metrics["regularized_logistic"] = {
        "validation_4agent": metric_summary(logistic_predictions(val4)),
        "test_4agent": metric_summary(logistic_predictions(test4))}
    write_json(p2 / "baselines.json", {"metrics": baseline_metrics, "logistic_model": logistic_model})
    feature_counts = {f: sum(e[f] for r in rows for e in r["evidence"]) for f in FEATURES}
    save_text(p2 / "Phase2_Results.md", "# Phase 2 — Evidence and baselines\n\n"
              f"- Trace-level features: {', '.join(FEATURES)}. Counts of positive agent features: {feature_counts}.\n"
              f"- Four-agent validation n={len(val4)}; test n={len(test4)}.\n"
              + "\n".join(metric_line(kind + " (test)", baseline_metrics[kind]["test_4agent"])
                          for kind in baseline_metrics) + "\n")
    paths["phase2"] = str((p2 / "Phase2_Results.md").relative_to(output))

    # Phase 3: choose evidence-set on validation, fit all available agent counts.
    p3 = phase_dir(output, 3, "bayesian_inference")
    candidate_features = [("error",), ("error", "uncertain"), FEATURES,
                          FEATURES + ("role_validator",), FEATURES + ROLE_FEATURES]
    selection = []
    for features in candidate_features:
        candidate = fit_model(train, 4, features)
        preds = make_predictions(val4, {"4": candidate})
        selection.append({"features": list(features), "validation": metric_summary(preds)})
    choice = min(selection, key=lambda c: (c["validation"]["brier"], -c["validation"]["accuracy"]))
    selected_features = tuple(choice["features"])
    models = {}
    for n in sorted({r["n_agents"] for r in rows}):
        if any(r["n_agents"] == n for r in train):
            models[str(n)] = fit_model(train, n, selected_features)
        else:
            models[str(n)] = {"n": n, "features": [], "alpha": 2.0,
                              "training_count": 0, "prior": [1 / n] * n, "cpts": {}}
    val_predictions = make_predictions(val4, models)
    brute_checks = []
    for row in val4[:min(6, len(val4))]:
        model = models["4"]
        exact = predict(model, row["evidence"])
        brute = brute_force(model, row["evidence"])
        brute_checks.append(max(abs(a - b) for a, b in zip(exact, brute)))
    max_diff = max(brute_checks, default=0.0)
    if max_diff > 1e-9:
        raise AssertionError(f"VE/brute-force disagreement {max_diff}")
    write_json(p3 / "model.json", {"models": models, "selected_features": selected_features,
                                   "selection": selection, "max_ve_brute_force_difference": max_diff})
    write_jsonl(p3 / "validation_predictions.jsonl", val_predictions)
    val_metric = metric_summary(val_predictions)
    save_text(p3 / "Phase3_Results.md", "# Phase 3 — Bayesian network and custom VE\n\n"
              f"- Selected evidence features on validation: {selected_features}.\n"
              f"- Categorical culprit model trained on {len(train4)} four-agent traces.\n"
              f"- Validation: {val_metric['correct']}/{val_metric['n']} correct ({val_metric['accuracy']:.1%}); "
              f"Brier {val_metric['brier']:.3f}.\n"
              f"- VE versus independent joint enumeration: maximum difference {max_diff:.3g}.\n"
              "- All posterior vectors sum to 1. Small agent-count groups use prior-only fallback.\n")
    paths["phase3"] = str((p3 / "Phase3_Results.md").relative_to(output))

    # Fit the validation-only calibrator before all held-out scenarios.
    temperature = fit_temperature(val_predictions)

    # Phase 4: missing logs on held-out traces, under the same calibration.
    p4 = phase_dir(output, 4, "missing_evidence")
    rng = random.Random(seed + 1)
    missing_rows = []
    for row in test4:
        n = row["n_agents"]
        scenarios = {"complete": [], "middle_A3_missing": [2],
                     "one_random_missing": [rng.randrange(n)],
                     "two_random_missing": sorted(rng.sample(range(n), 2))}
        for name, masked_positions in scenarios.items():
            evidence = [None if i in masked_positions else dict(e) for i, e in enumerate(row["evidence"])]
            probabilities = calibrate_temperature(predict(models["4"], evidence), temperature)
            if not normalized(probabilities):
                raise AssertionError("missing-log posterior invalid")
            missing_rows.append({"id": row["id"], "scenario": name, "masked_positions": masked_positions,
                                 "actual": row["culprit"], "predicted": max(range(n), key=lambda i: probabilities[i]),
                                 "probabilities": probabilities})
    missing_metrics = {name: metric_summary([r for r in missing_rows if r["scenario"] == name])
                       for name in ("complete", "middle_A3_missing", "one_random_missing", "two_random_missing")}
    write_jsonl(p4 / "missing_evidence_predictions.jsonl", missing_rows)
    write_json(p4 / "metrics.json", missing_metrics)
    save_text(p4 / "Phase4_Results.md", "# Phase 4 — Missing-log inference\n\n"
              + "\n".join(metric_line(name, metric) for name, metric in missing_metrics.items())
              + "\n\nMissing evidence is marginalized by VE; it is not treated as a clean log. "
                "Every scenario uses the validation-fitted temperature. Masks are fixed by the recorded seed.\n")
    paths["phase4"] = str((p4 / "Phase4_Results.md").relative_to(output))

    # Phase 5: validation-set calibration and untouched test-set evaluation.
    p5 = phase_dir(output, 5, "calibration_evaluation")
    raw_test = make_predictions(test4, models)
    calibrated_test = make_predictions(test4, models, temperature)
    exploratory_all_test = make_predictions(test, models)
    exploratory_by_count = {str(n): metric_summary([p for p in exploratory_all_test if p["n_agents"] == n])
                            for n in sorted({p["n_agents"] for p in exploratory_all_test})}
    raw_metric, calibrated_metric = metric_summary(raw_test), metric_summary(calibrated_test)
    calibrated_metric["accuracy_95pct_bootstrap_interval"] = bootstrap_accuracy_interval(calibrated_test)
    write_json(p5 / "calibration.json", {"temperature": temperature, "validation_fit_count": len(val4),
                                         "uncalibrated_test": raw_metric, "calibrated_test": calibrated_metric})
    write_jsonl(p5 / "test_predictions.jsonl", calibrated_test)
    write_jsonl(p5 / "all_agent_counts_test_predictions.jsonl", exploratory_all_test)
    write_json(p5 / "exploratory_agent_count_metrics.json", exploratory_by_count)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    draw_reliability(calibrated_metric, p5 / "reliability_diagram.png")
    prior_test = baseline_metrics["position_prior"]["test_4agent"]
    save_text(p5 / "Phase5_Results.md", "# Phase 5 — Held-out calibration and attribution\n\n"
              f"- Temperature chosen on validation only: {temperature:.3f}.\n"
              + metric_line("BN uncalibrated", raw_metric) + "\n"
              + metric_line("BN calibrated", calibrated_metric) + "\n"
              + metric_line("Position-prior baseline", prior_test) + "\n"
              f"- Accuracy 95% trace-bootstrap interval: {calibrated_metric['accuracy_95pct_bootstrap_interval']}.\n"
              f"- Exploratory uncalibrated test breakdown by agent count: "
              + ", ".join(f"{n} agents {v['correct']}/{v['n']}" for n, v in exploratory_by_count.items()) + ".\n"
              "- This is a small held-out subset; literature scores are not directly comparable without a matched protocol.\n")
    paths["phase5"] = str((p5 / "Phase5_Results.md").relative_to(output))

    # Phase 6: temporal model fitted on train, threshold tuned on validation.
    p6 = phase_dir(output, 6, "temporal_hmm")
    # Only temporal records whose annotated step is authored by the labeled
    # culprit have a defensible target for strictly-before-mistake evaluation.
    train_records = [by_id[i] for i in splits["train"] if by_id[i]["label_message_matches"]]
    val_records = [by_id[i] for i in splits["validation"] if len(by_id[i]["agents"]) == 4
                   and by_id[i]["label_message_matches"]]
    test_records = [by_id[i] for i in splits["test"] if len(by_id[i]["agents"]) == 4
                    and by_id[i]["label_message_matches"]]
    hmm = fit_hmm(train_records)
    sequence = [m["flags"] for m in train_records[0]["messages"][:2]]
    hmm_diff = max(abs(a - b) for a, b in zip(forward(hmm, sequence)[-1], brute_force_forward(hmm, sequence)))
    if hmm_diff > 1e-9:
        raise AssertionError("HMM forward/brute-force disagreement")
    threshold, sweep = choose_threshold(hmm, val_records)
    warnings = [monitor(hmm, record, threshold) for record in test_records]
    hmm_summary = warning_summary(warnings)
    keyword_summary = warning_summary([keyword_monitor(record) for record in test_records])
    write_json(p6 / "hmm_model.json", {"model": hmm, "selected_threshold": threshold,
                                       "validation_threshold_sweep": sweep,
                                       "train_excluded_ambiguous_steps": len(splits["train"]) - len(train_records),
                                       "validation_excluded_ambiguous_steps": len(val4) - len(val_records),
                                       "test_excluded_ambiguous_steps": len(test4) - len(test_records),
                                       "forward_brute_force_max_difference": hmm_diff})
    write_jsonl(p6 / "test_warnings.jsonl", warnings)
    write_json(p6 / "metrics.json", {"hmm": hmm_summary, "keyword_baseline": keyword_summary})
    save_text(p6 / "Phase6_Results.md", "# Phase 6 — Temporal HMM and early warning\n\n"
              f"- HMM fitted on train traces with annotation-derived proxy states; threshold {threshold:.2f} chosen on validation.\n"
              f"- Ambiguous mistake-step labels excluded: train {len(splits['train'])-len(train_records)}, "
              f"four-agent validation {len(val4)-len(val_records)}, four-agent test {len(test4)-len(test_records)}.\n"
              f"- Held-out four-agent traces: {hmm_summary['total']}; eligible for pre-mistake warning: {hmm_summary['eligible']}.\n"
              f"- Correct-culprit early warnings: {hmm_summary['correct_early']}/{hmm_summary['eligible']} eligible.\n"
              f"- Traces with a warning for another agent: {hmm_summary['false_agent_warning_traces']}/{hmm_summary['total']}.\n"
              f"- Keyword baseline correct early: {keyword_summary['correct_early']}/{keyword_summary['eligible']}; "
              f"wrong-agent warning traces: {keyword_summary['false_agent_warning_traces']}/{keyword_summary['total']}.\n"
              f"- Forward versus brute-force max difference: {hmm_diff:.3g}.\n"
              "- A fitted HMM does not establish early-warning usefulness unless these held-out results support it.\n")
    paths["phase6"] = str((p6 / "Phase6_Results.md").relative_to(output))

    # Phase 7: explicit modality evidence audit; no synthetic multimodal claims.
    p7 = phase_dir(output, 7, "multimodal_scope")
    modality_keys = {"image", "images", "audio", "video", "frames"}
    modalities = Counter()
    multimodal_trace_count = 0
    for record in records:
        source = json.loads((dataset / record["id"]).read_text(encoding="utf-8-sig"))
        trace_has_modality = False
        for message in source.get("history", []):
            if isinstance(message, dict):
                for key in modality_keys & message.keys():
                    modalities[key] += 1
                    trace_has_modality = True
        multimodal_trace_count += int(trace_has_modality)
    multimodal = {"modality_fields": dict(modalities),
                  "labeled_multimodal_trace_count": multimodal_trace_count,
                  "conclusion": ("The supplied benchmark supports multi-agent LLM attribution; "
                                 "no separately labeled multimodal benchmark was supplied."
                                 if multimodal_trace_count == 0 else
                                 "Some traces contain explicit modality fields; a separate modality-specific evaluation is still required.")}
    write_json(p7 / "modality_audit.json", multimodal)
    save_text(p7 / "Phase7_Results.md", "# Phase 7 — Multimodal scope audit\n\n"
              f"- Explicit image/audio/video fields in supplied traces: {dict(modalities)}; "
              f"labeled traces with such fields: {multimodal_trace_count}.\n"
              "- This project is reported as **multi-agent**, matching the PPT's technical scope. "
              "A separate modality-specific performance claim is not made.\n")
    paths["phase7"] = str((p7 / "Phase7_Results.md").relative_to(output))

    # Phase 9: verifiable, cautious review output using the frozen BN scores.
    p9 = phase_dir(output, 9, "evidence_triage")
    review_packets = []
    for prediction in calibrated_test:
        record = by_id[prediction["id"]]
        row = rows_by_id[prediction["id"]]
        effects = []
        for index in range(row["n_agents"]):
            masked = [dict(e) for e in row["evidence"]]
            masked[index] = None
            without = calibrate_temperature(predict(models["4"], masked), temperature)
            effects.append(prediction["probabilities"][index] - without[index])
        review = packet(record, prediction["probabilities"], list(selected_features),
                        evidence_effects=effects)
        for candidate in review["candidates"]:
            for observation in candidate["observations"]:
                if "raw_message_index" in observation:
                    original = next(m for m in record["messages"]
                                    if m["raw_index"] == observation["raw_message_index"])
                    if original["agent"] != candidate["agent"]:
                        raise AssertionError("review excerpt points to a different agent")
        review_packets.append({"id": prediction["id"], "actual": prediction["actual"],
                               "top_choice_correct": prediction["predicted"] == prediction["actual"],
                               "top_two_contains_actual": record["agents"][prediction["actual"]]
                               in review["top_two_agents"], "review": review})
    top_two_correct = sum(p["top_two_contains_actual"] for p in review_packets)
    write_jsonl(p9 / "test_review_packets.jsonl", review_packets)
    triage_metrics = {"test_count": len(review_packets),
                      "top_one_correct": calibrated_metric["correct"],
                      "top_two_contains_actual": top_two_correct,
                      "top_two_coverage": top_two_correct / len(review_packets),
                      "interpretation": "Top-two coverage measures shortlist usefulness, not automatic attribution accuracy."}
    write_json(p9 / "metrics.json", triage_metrics)
    save_text(p9 / "Phase9_Results.md", "# Phase 9 — Evidence-backed review packets\n\n"
              f"- Frozen BN top choice: {calibrated_metric['correct']}/{len(review_packets)} correct.\n"
              f"- True agent in top two: {top_two_correct}/{len(review_packets)} "
              f"({top_two_correct / len(review_packets):.1%}).\n"
              "- Each packet gives ranked probabilities, the two-agent shortlist, selected evidence flags, "
              "source-message indices with excerpts, and exact model sensitivity to masking each agent's evidence.\n"
              "- Every cited message index is checked against the parsed trace and agent.\n"
              "- Excerpts are observations, not causal proof. Human review remains necessary.\n")
    paths["phase9"] = str((p9 / "Phase9_Results.md").relative_to(output))

    # Phase 8: concise final assessment and provenance manifest.
    p8 = phase_dir(output, 8, "final_report")
    summary = {"dataset": str(dataset), "split_counts": {k: len(v) for k, v in splits.items()},
               "four_agent_counts": {"train": len(train4), "validation": len(val4), "test": len(test4)},
               "selected_features": selected_features, "test_calibrated": calibrated_metric,
               "test_uncalibrated": raw_metric, "test_position_prior": prior_test,
               "exploratory_agent_count_metrics": exploratory_by_count,
               "missing_evidence": missing_metrics, "temporal": hmm_summary,
               "evidence_triage": triage_metrics,
               "temporal_keyword_baseline": keyword_summary,
               "multimodal": multimodal, "phase_reports": paths}
    write_json(p8 / "summary.json", summary)
    save_text(p8 / "Final_Report.md", "# AgentDiag v2 — Final report\n\n"
              "This is an independent implementation of the multi-agent scope in the PPT. "
              f"It parses {len(records)} Who&When traces and uses a question-group-safe split.\n\n"
              "## Root-cause attribution (held-out four-agent traces)\n\n"
              + metric_line("Bayesian network, calibrated", calibrated_metric) + "\n"
              + metric_line("Position-prior baseline", prior_test) + "\n"
              f"- Selected features: {selected_features}; temperature: {temperature:.3f}.\n"
              f"- Accuracy 95% bootstrap interval: {calibrated_metric['accuracy_95pct_bootstrap_interval']}.\n"
              "- Categorical posteriors are checked to sum to one. VE is checked against joint enumeration.\n\n"
              f"- Evidence-backed two-agent shortlist contains the true agent in {top_two_correct}/{len(review_packets)} "
              "held-out traces. This is review coverage, not automatic accuracy.\n\n"
              "- Exploratory uncalibrated test breakdown: "
              + ", ".join(f"{n} agents {v['correct']}/{v['n']}" for n, v in exploratory_by_count.items())
              + ". These smaller groups are not the primary calibrated result.\n\n"
              "## Missing logs\n\n"
              + "\n".join(metric_line(name, metric) for name, metric in missing_metrics.items()) + "\n\n"
              "## Early warning\n\n"
              f"- Correct culprit warned before the mistake: {hmm_summary['correct_early']}/{hmm_summary['eligible']} "
              "eligible held-out traces.\n"
              f"- Wrong-agent warning traces: {hmm_summary['false_agent_warning_traces']}/{hmm_summary['total']}.\n\n"
              f"- Keyword baseline: {keyword_summary['correct_early']}/{keyword_summary['eligible']} correct early, "
              f"{keyword_summary['false_agent_warning_traces']}/{keyword_summary['total']} wrong-agent warning traces.\n\n"
              "## Scope and claims\n\n"
              "- This is multi-agent failure analysis. A separate multimodal performance claim is not supported "
              "by the current evaluation.\n"
              "- Comparison to AgenTracer is not asserted: a matched evaluation protocol/predictions were not available locally.\n"
              "- The source benchmark contains failed runs, so final-failure prediction from successful/failed examples is not evaluated.\n"
              "- The PPT should use these held-out results and explicitly identify any goal not achieved.\n")
    save_text(p8 / "PPT_Alignment.md", "# Current PPT claims versus verified AgentDiag v2 results\n\n"
              "- **Bayesian network and data-driven CPTs:** Implemented using training-only counts.\n"
              "- **Hand-implemented Variable Elimination:** Implemented; compared with exact joint enumeration.\n"
              "- **Probability distribution over root causes:** Implemented as one categorical culprit variable; every posterior sums to one.\n"
              "- **Missing observations:** Implemented by marginalizing absent evidence; evaluated on held-out masks.\n"
              "- **Calibration metrics and reliability diagram:** Implemented on held-out data after validation-only temperature fitting. "
              f"Test ECE is {calibrated_metric['ece']:.3f} versus {prior_test['ece']:.3f} for the positional baseline, "
              "so superior calibration is **not established**.\n"
              "- **HMM reliability tracking:** Implemented and tested. "
              f"Correct early warnings are {hmm_summary['correct_early']}/{hmm_summary['eligible']} eligible traces, "
              f"with wrong-agent warnings on {hmm_summary['false_agent_warning_traces']}/{hmm_summary['total']} traces; "
              "early identification is **not reliable**.\n"
              "- **Competitive performance versus AgenTracer:** **Not established**; no matched runnable evaluation is available.\n"
              "- **Multimodal agents:** Outside the current PPT's multi-agent scope and without a separate modality-specific evaluation.\n")
    paths["phase8"] = str((p8 / "Final_Report.md").relative_to(output))
    from agentdiag_v2.make_presentation import build as build_presentation
    build_presentation(p8 / "summary.json", p8 / "AgentDiag_v2_Final_Review.pptx")
    write_json(output / "phase_manifest.json", paths)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=HERE / "artifacts")
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    summary = run(args.dataset.resolve(), args.output.resolve(), args.seed)
    m = summary["test_calibrated"]
    print(f"Complete: {m['correct']}/{m['n']} held-out four-agent attributions correct "
          f"({m['accuracy']:.1%}); see {args.output / 'phase8_final_report/Final_Report.md'}")


if __name__ == "__main__":
    main()
