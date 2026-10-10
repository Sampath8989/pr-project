"""Agreement must use observed speakers and probabilities, never targets."""
from agentdiag_v2.contextual_consistency import reconcile


def test_agreement_ignores_labels_and_preserves_raw_indices():
    record = {"agents": ["A", "B"], "culprit_name": "B", "mistake_step": 9,
              "messages": [{"agent": "A", "raw_index": 3}, {"agent": "B", "raw_index": 9}]}
    prediction = {"agent_probabilities": [0.25, 0.75], "step_probabilities": [0.9, 0.1],
                  "actual_agent": "B", "actual_step": 9}
    first = reconcile(record, prediction, 0.5, 1)
    record.update(culprit_name="A", mistake_step=3)
    prediction.update(actual_agent="A", actual_step=3)
    assert first == reconcile(record, prediction, 0.5, 1)
    assert first["predicted_agent"] == "A" and first["predicted_step"] == 3
    assert abs(sum(first["agent_probabilities"]) - 1) < 1e-12
    assert abs(sum(first["step_probabilities"]) - 1) < 1e-12
