import json
from pathlib import Path

from agentdiag_v2.src.data import parse_unlabeled
from agentdiag_v2.src.step_ranker import predict


ROOT = Path(__file__).resolve().parents[2]


def test_saved_step_model_does_not_read_labels(tmp_path):
    source = ROOT / "Agents_Failure_Attribution" / "Who&When" / "Algorithm-Generated" / "121.json"
    raw = json.loads(source.read_text())
    model = json.loads((ROOT / "agentdiag_v2" / "benchmark_step" / "step_model.json").read_text())
    with_label = parse_unlabeled(source)
    raw.pop("mistake_agent")
    raw.pop("mistake_step")
    unlabeled_path = tmp_path / "trace.json"
    unlabeled_path.write_text(json.dumps(raw))
    without_label = parse_unlabeled(unlabeled_path)
    first, second = predict(model, [with_label, without_label])
    assert first == second
    assert abs(sum(first["agent_probabilities"]) - 1) < 1e-9
    assert abs(sum(first["step_probabilities"]) - 1) < 1e-9
    assert first["predicted_step"] in [message["raw_index"] for message in with_label["messages"]]
