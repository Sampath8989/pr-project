"""Checks for observable inputs, masking, joint losses and portable predictions."""

import json

import numpy as np
import pytest

from agentdiag_v2.src.data import parse_unlabeled


def test_unlabeled_parser_keeps_task_and_ignores_answers(tmp_path):
    path = tmp_path / "trace.json"
    raw = {"question": "Find the sum.", "history": [
        {"name": "A", "content": "Try addition."},
        {"name": "B", "content": "The result is 7."}],
        "mistake_agent": "A", "mistake_step": 0, "ground_truth": "answer 1"}
    path.write_text(json.dumps(raw))
    first = parse_unlabeled(path)
    raw.update(mistake_agent="B", mistake_step=1, ground_truth="answer 2")
    path.write_text(json.dumps(raw))
    assert first == parse_unlabeled(path)
    assert first["question"] == "Find the sum."


def test_contextual_heads_ignore_padding_and_serialize(tmp_path):
    torch = pytest.importorskip("torch")
    from agentdiag_v2.src.contextual_ranker import ContextualRanker, loss_for_batch
    torch.manual_seed(12)
    model = ContextualRanker(embedding_dim=8, numeric_dim=4, hidden=16, dropout=0)
    x = torch.randn(2, 4, 8)
    q = torch.randn(2, 8)
    num = torch.randn(2, 4, 4)
    agent = torch.tensor([[0, 1, 0, -1], [0, 1, 2, 0]])
    mask = agent >= 0
    batch = {"content": x, "question": q, "numeric": num,
             "agent_index": agent, "mask": mask, "agent_count": torch.tensor([2, 3]),
             "step_target": torch.tensor([1, 2]), "agent_target": torch.tensor([1, 1]),
             "raw_index": torch.tensor([[2, 4, 6, -1], [0, 1, 3, 4]])}
    model.eval()
    with torch.no_grad():
        step, culprit = model(batch)
        changed = {**batch, "content": x.clone(), "numeric": num.clone()}
        changed["content"][0, 3] = 999
        changed["numeric"][0, 3] = -999
        step2, culprit2 = model(changed)
    assert torch.allclose(step, step2, atol=1e-6)
    assert torch.allclose(culprit, culprit2, atol=1e-6)
    assert step.softmax(-1)[0, 3] == 0
    assert culprit.softmax(-1)[0, 2] == 0
    assert torch.allclose(culprit.softmax(-1).sum(-1), torch.ones(2))
    step, culprit = model(batch)
    for sigma in (0, 1):
        value = loss_for_batch(step, culprit, batch, sigma=sigma)
        assert torch.isfinite(value)
    value.backward()
    assert model.agent_head[-1].weight.grad.abs().sum() > 0
    checkpoint = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    clone = ContextualRanker(embedding_dim=8, numeric_dim=4, hidden=16, dropout=0)
    clone.load_state_dict(torch.load(checkpoint, weights_only=True))
    clone.eval()
    with torch.no_grad():
        s3, a3 = clone(batch)
    assert torch.allclose(step, s3)
    assert torch.allclose(culprit, a3)


def test_cache_key_uses_encoder_configuration():
    from agentdiag_v2.src.semantic_encoder import cache_key
    assert cache_key("hello", 256, 384) != cache_key("hello", 128, 384)
    assert cache_key("hello", 256, 384) != cache_key("hello", 256, 512)
    assert cache_key("hello", 256, 384) == cache_key("hello", 256, 384)


def test_empty_text_has_neutral_embedding_without_model_download(tmp_path):
    pytest.importorskip("torch")
    from agentdiag_v2.src.semantic_encoder import SemanticEncoder
    encoder = SemanticEncoder(tmp_path)
    assert np.array_equal(encoder.encode([""])[""], np.zeros(256))
    assert encoder.model is None


def test_packed_cache_preserves_vectors_without_loading_encoder(tmp_path):
    pytest.importorskip("torch")
    from agentdiag_v2.src.semantic_encoder import SemanticEncoder, cache_key
    vectors = tmp_path / "vectors"
    vectors.mkdir()
    expected = np.arange(256, dtype=np.float32)
    np.save(vectors / (cache_key("synthetic test") + ".npy"), expected)
    first = SemanticEncoder(tmp_path)
    assert np.array_equal(first.encode(["synthetic test"])["synthetic test"], expected)
    (vectors / (cache_key("synthetic test") + ".npy")).unlink()
    second = SemanticEncoder(tmp_path)
    assert np.array_equal(second.encode(["synthetic test"])["synthetic test"], expected)
    assert second.model is None


def test_sparse_contextual_heads_return_observed_raw_indices(tmp_path):
    from agentdiag_v2.src.contextual_sparse import AGENT_D, STEP_D, predict
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"question": "Find a number", "history": [
        {"name": "user", "content": "task"}, {"name": "A", "content": "maybe 3"},
        {"name": "B", "content": "recheck the number"}]}))
    row = parse_unlabeled(path)
    model = {"step_weights": [0] * STEP_D, "agent_weights": [0] * AGENT_D}
    output = predict(model, [row])[0]
    assert output["predicted_step"] == 1
    assert output["predicted_agent"] == "A"
    assert np.allclose(output["step_probabilities"], [0.5, 0.5])
    assert np.allclose(output["agent_probabilities"], [0.5, 0.5])
