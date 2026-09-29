"""Focused correctness and integration checks for the independent pipeline."""

import json
from pathlib import Path

import pytest
from pptx import Presentation

from agentdiag_v2.predict import diagnose
from agentdiag_v2.run_pipeline import run
from agentdiag_v2.src.bayes import brute_force, fit_model, predict
from agentdiag_v2.src.data import feature_rows, load_records, split_records
from agentdiag_v2.src.evaluation import normalized
from agentdiag_v2.src.temporal import brute_force_forward, fit_hmm, forward


PROJECT = Path(__file__).resolve().parents[2]
DATASET = PROJECT / "Agents_Failure_Attribution" / "Who&When"


@pytest.fixture(scope="module")
def records():
    parsed, exclusions = load_records(DATASET)
    assert len(parsed) == 184
    assert not exclusions
    return parsed


def test_question_groups_never_cross_splits(records):
    splits = split_records(records)
    by_id = {r["id"]: r for r in records}
    mapping = {}
    for part, ids in splits.items():
        for trace_id in ids:
            qid = by_id[trace_id]["question_id"]
            assert mapping.get(qid, part) == part
            mapping[qid] = part
    assert sum(map(len, splits.values())) == len(records)


def test_ve_matches_enumeration_with_missing_agent(records):
    rows = feature_rows(records)
    train_ids = set(split_records(records)["train"])
    train = [r for r in rows if r["id"] in train_ids]
    example = next(r for r in rows if r["n_agents"] == 4)
    model = fit_model(train, 4, ("error", "uncertain", "correction"))
    for evidence in [example["evidence"],
                     [example["evidence"][0], None, example["evidence"][2], example["evidence"][3]]]:
        exact, enumerated = predict(model, evidence), brute_force(model, evidence)
        assert normalized(exact)
        assert max(abs(a - b) for a, b in zip(exact, enumerated)) < 1e-10


def test_hmm_forward_matches_enumeration(records):
    exact = [r for r in records if r["label_message_matches"]]
    model = fit_hmm(exact[:20])
    observations = [m["flags"] for m in exact[0]["messages"][:3]]
    filtered = forward(model, observations)[-1]
    enumerated = brute_force_forward(model, observations)
    assert max(abs(a - b) for a, b in zip(filtered, enumerated)) < 1e-10


def test_full_run_writes_every_phase_and_diagnoses_unlabeled_trace(tmp_path, records):
    output = tmp_path / "artifacts"
    summary = run(DATASET, output, 20260929)
    manifest = json.loads((output / "phase_manifest.json").read_text())
    assert set(manifest) == {f"phase{i}" for i in range(10)}
    assert all(Path(path).is_file() for path in manifest.values())
    assert summary["test_calibrated"]["n"] > 0
    assert summary["test_calibrated"]["n"] == summary["test_position_prior"]["n"]
    assert summary["multimodal"]["labeled_multimodal_trace_count"] == 0
    assert len(Presentation(output / "phase8_final_report" / "AgentDiag_v2_Final_Review.pptx").slides) == 10
    original = json.loads((DATASET / "Algorithm-Generated" / "1.json").read_text())
    original.pop("mistake_agent", None)
    original.pop("mistake_step", None)
    trace_path = tmp_path / "new_trace.json"
    trace_path.write_text(json.dumps(original))
    diagnosis = diagnose(trace_path, output)
    assert abs(sum(diagnosis["root_cause_probabilities"].values()) - 1) < 1e-9
    assert diagnosis["predicted_culprit"] in diagnosis["agents"]
    assert len(diagnosis["review_packet"]["top_two_agents"]) == 2
    assert all(c["observed_evidence_probability_effect"] is not None
               for c in diagnosis["review_packet"]["candidates"])
    assert len(diagnosis["reliability_timeline"]) > 0
    masked = diagnose(trace_path, output, [2])
    assert next(c for c in masked["review_packet"]["candidates"]
                if c["agent"] == diagnosis["agents"][2])["observed_evidence_probability_effect"] is None
    assert all(item["agent"] != diagnosis["agents"][2]
               for item in masked["reliability_timeline"])
