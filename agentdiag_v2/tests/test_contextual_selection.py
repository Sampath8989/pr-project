"""Integration check for independently selected agent and step components."""
import json

import numpy as np

from agentdiag_v2.src.data import parse_unlabeled


def test_selected_heads_use_frozen_components_without_labels(tmp_path, monkeypatch):
    import agentdiag_v2.contextual_upgrade as upgrade
    from agentdiag_v2.src.contextual_sparse import AGENT_D, STEP_D
    base = tmp_path / "benchmark_step"
    base.mkdir()
    (base / "step_model.json").write_text(json.dumps({"weights": [0] * 6432}))
    model_dir = tmp_path / "selected"
    model_dir.mkdir()
    (model_dir / "selection.json").write_text(json.dumps({"selection": {
        "agent": {"candidate": "sparse_0.5", "weight": 1},
        "step": {"candidate": "baseline", "weight": 0}}}))
    agent_weights = [0] * AGENT_D
    agent_weights[16384] = 10
    (model_dir / "model_sparse_0.5.json").write_text(json.dumps({
        "step_weights": [0] * STEP_D, "agent_weights": agent_weights}))
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"question": "Solve a task", "history": [
        {"name": "user", "content": "task"}, {"name": "A", "content": "first attempt"},
        {"name": "B", "content": "second attempt"}]}))
    monkeypatch.setattr(upgrade, "HERE", tmp_path)
    result = upgrade.predict_selected([parse_unlabeled(path)], model_dir, tmp_path / "cache")[0]
    assert result["predicted_agent"] == "B"
    assert result["predicted_step"] == 1
    assert np.isclose(sum(result["agent_probabilities"]), 1)
    assert np.isclose(sum(result["step_probabilities"]), 1)
