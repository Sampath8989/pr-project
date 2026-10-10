"""Compact temporal model with directly supervised agent and step heads."""

from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .step_ranker import features


def numeric_features(record: dict) -> np.ndarray:
    rows = []
    for j in range(len(record["messages"])):
        values = features(record, j)
        rows.append([values.get(6400 + i, 0) for i in range(32)])
    return np.asarray(rows, dtype=np.float32)


def prepare(record: dict, embeddings: dict[str, np.ndarray]) -> dict:
    n = len(record["messages"])
    data = {"content": np.stack([embeddings[m["content"]] for m in record["messages"]]),
            "agent_semantic": np.stack([embeddings[m["agent"]] for m in record["messages"]]),
            "question": embeddings[record.get("question", "")],
            "numeric": numeric_features(record),
            "agent_index": np.asarray([record["agents"].index(m["agent"])
                                       for m in record["messages"]]),
            "raw_index": np.asarray([m["raw_index"] for m in record["messages"]]),
            "agent_count": len(record["agents"]), "length": n, "record": record}
    if "mistake_step" in record:
        positions = np.where(data["raw_index"] == record["mistake_step"])[0]
        data["step_target"] = int(positions[0]) if len(positions) else -1
        data["agent_target"] = record["culprit"]
    return data


def collate(rows: list[dict], device: str = "cpu") -> dict:
    b, t, d = len(rows), max(r["length"] for r in rows), rows[0]["content"].shape[1]
    num = rows[0]["numeric"].shape[1]
    batch = {"content": torch.zeros(b, t, d), "agent_semantic": torch.zeros(b, t, d),
             "numeric": torch.zeros(b, t, num),
             "question": torch.from_numpy(np.stack([r["question"] for r in rows])),
             "agent_index": torch.full((b, t), -1, dtype=torch.long),
             "raw_index": torch.full((b, t), -1, dtype=torch.long),
             "mask": torch.zeros(b, t, dtype=torch.bool),
             "agent_count": torch.tensor([r["agent_count"] for r in rows])}
    for i, row in enumerate(rows):
        n = row["length"]
        for name in ("content", "agent_semantic", "numeric", "agent_index", "raw_index"):
            batch[name][i, :n] = torch.from_numpy(row[name])
        batch["mask"][i, :n] = True
    if "step_target" in rows[0]:
        batch["step_target"] = torch.tensor([r["step_target"] for r in rows])
        batch["agent_target"] = torch.tensor([r["agent_target"] for r in rows])
    return {k: v.to(device) for k, v in batch.items()}


class ContextualRanker(nn.Module):
    def __init__(self, embedding_dim: int = 256, numeric_dim: int = 32,
                 hidden: int = 64, dropout: float = 0.2, temporal: bool = True):
        super().__init__()
        self.config = dict(embedding_dim=embedding_dim, numeric_dim=numeric_dim,
                           hidden=hidden, dropout=dropout, temporal=temporal)
        self.content = nn.Linear(embedding_dim, hidden)
        self.question = nn.Linear(embedding_dim, hidden)
        self.numeric = nn.Linear(numeric_dim, hidden // 2)
        self.role = nn.Linear(embedding_dim, hidden // 2)
        self.input = nn.Sequential(nn.Linear(hidden * 4, hidden * 2),
                                   nn.LayerNorm(hidden * 2), nn.GELU(), nn.Dropout(dropout))
        self.temporal = temporal
        if temporal:
            self.gru = nn.GRU(hidden * 2, hidden, bidirectional=True, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden * 2, 2, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden * 2)
        self.step_head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(),
                                       nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.agent_head = nn.Sequential(nn.Linear(hidden * 4, hidden), nn.GELU(),
                                        nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        mask, index = batch["mask"], batch["agent_index"]
        c = self.content(batch["content"])
        q = self.question(batch["question"]).unsqueeze(1).expand_as(c)
        previous = F.pad(c[:, :-1], (0, 0, 1, 0))
        x = self.input(torch.cat([c, c * q, c - previous,
                                 self.numeric(batch["numeric"]),
                                 self.role(batch.get("agent_semantic", torch.zeros_like(batch["content"])))], dim=-1))
        x = x * mask.unsqueeze(-1)
        if self.temporal:
            packed = nn.utils.rnn.pack_padded_sequence(x, mask.sum(1).cpu(),
                                                       batch_first=True, enforce_sorted=False)
            packed, _ = self.gru(packed)
            context, _ = nn.utils.rnn.pad_packed_sequence(packed, batch_first=True,
                                                         total_length=x.shape[1])
            x = x + context
        attended, _ = self.attention(x, x, x, key_padding_mask=~mask, need_weights=False)
        x = self.norm(x + attended) * mask.unsqueeze(-1)
        step = self.step_head(x).squeeze(-1).masked_fill(~mask, float("-inf"))
        count = int(batch["agent_count"].max())
        pooled = []
        for a in range(count):
            authored = (index == a) & mask
            mean = (x * authored.unsqueeze(-1)).sum(1) / authored.sum(1).clamp(min=1).unsqueeze(-1)
            maximum = x.masked_fill(~authored.unsqueeze(-1), float("-inf")).amax(1)
            maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
            pooled.append(torch.cat([mean, maximum], dim=-1))
        agents = self.agent_head(torch.stack(pooled, dim=1)).squeeze(-1)
        valid = torch.arange(count, device=x.device)[None] < batch["agent_count"][:, None]
        return step, agents.masked_fill(~valid, float("-inf"))


def loss_for_batch(step: torch.Tensor, agents: torch.Tensor, batch: dict,
                   sigma: float = 0, agent_weight: float = 1) -> torch.Tensor:
    target = batch["step_target"]
    loss = F.cross_entropy(step, target) + agent_weight * F.cross_entropy(agents, batch["agent_target"])
    if sigma > 0:
        raw = batch["raw_index"]
        location = raw.gather(1, target[:, None])
        soft = torch.exp(-0.5 * ((raw - location) / sigma).float().square()) * batch["mask"]
        soft /= soft.sum(1, keepdim=True)
        logp = F.log_softmax(step, dim=-1).masked_fill(~batch["mask"], 0)
        loss = loss + 0.15 * -(soft * logp).sum(1).mean()
    return loss


def predict(model: ContextualRanker, rows: list[dict], device: str = "cpu",
            batch_size: int = 32) -> list[dict]:
    model.to(device).eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            group = rows[start:start + batch_size]
            s, a = model(collate(group, device))
            sp, ap = s.softmax(-1).cpu().numpy(), a.softmax(-1).cpu().numpy()
            for i, row in enumerate(group):
                record = row["record"]
                outputs.append({"predicted_step": int(row["raw_index"][sp[i, :row['length']].argmax()]),
                                "predicted_agent": record["agents"][ap[i, :row['agent_count']].argmax()],
                                "step_probabilities": sp[i, :row["length"]].astype(float).tolist(),
                                "agent_probabilities": ap[i, :row["agent_count"]].astype(float).tolist()})
    return outputs


def fit(rows: list[dict], config: dict, seed: int, epochs: int, device: str,
        validation: list[dict] | None = None, checkpoints: tuple[int, ...] = ()):
    if not rows or any(not 0 <= r["step_target"] < r["length"] for r in rows):
        raise ValueError("Training needs an observed target step for every trace")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = ContextualRanker(**config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("lr", 0.001), weight_decay=0.01)
    rng = random.Random(seed)
    predictions = {}
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(rows)))
        rng.shuffle(order)
        for start in range(0, len(order), 32):
            group = [rows[i] for i in order[start:start + 32]]
            batch = collate(group, device)
            s, a = model(batch)
            loss = loss_for_batch(s, a, batch, sigma=config.get("sigma", 0),
                                  agent_weight=config.get("agent_weight", 1))
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        if validation is not None and epoch in checkpoints:
            predictions[epoch] = predict(model, validation, device)
    return model, predictions
