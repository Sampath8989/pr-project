import json
from pathlib import Path

from agentdiag_v2.predict_hybrid import diagnose


ROOT = Path(__file__).resolve().parents[2]


def test_hybrid_prediction_matches_saved_evaluation_without_labels(tmp_path):
    source = ROOT / "Agents_Failure_Attribution" / "Who&When" / "Algorithm-Generated" / "121.json"
    raw = json.loads(source.read_text())
    labeled = diagnose(source)
    raw.pop("mistake_agent")
    raw.pop("mistake_step")
    unlabeled_path = tmp_path / "new_trace.json"
    unlabeled_path.write_text(json.dumps(raw))
    unlabeled = diagnose(unlabeled_path)
    assert labeled["predicted_culprit"] == unlabeled["predicted_culprit"]
    assert labeled["agent_probabilities"] == unlabeled["agent_probabilities"]
    saved = [json.loads(line) for line in
             (ROOT / "agentdiag_v2" / "hybrid_upgrade" / "test_predictions.jsonl").read_text().splitlines()]
    expected = next(row for row in saved if row["id"] == "Algorithm-Generated/121.json")
    assert list(labeled["agent_probabilities"].values()) == expected["hybrid"]
    assert abs(sum(labeled["agent_probabilities"].values()) - 1) < 1e-9
