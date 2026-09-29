"""Parse Who&When traces without importing code or results from the older project."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


ERROR = re.compile(r"\b(error|traceback|exception|failed|failure|invalid|timeout|404|500)\b", re.I)
UNCERTAIN = re.compile(r"\b(maybe|perhaps|uncertain|unsure|cannot|unable|not sure|could not)\b", re.I)
CORRECT = re.compile(r"\b(correct|correction|retry|recheck|verify|mistake|instead)\b", re.I)
FEATURES = ("error", "uncertain", "correction")
ROLE_FEATURES = ("role_validator", "role_tool")


def normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def speaker(message: dict) -> str | None:
    name = str(message.get("name") or message.get("role") or "").strip()
    if not name:
        return None
    low = name.lower()
    if low in {"human", "user"} or "thought" in low:
        return None
    name = re.sub(r"\s*\([^)]*\)", "", name).strip()
    return name or None


def flags(content: str) -> dict[str, int]:
    s = str(content)
    return {"error": int(bool(ERROR.search(s))),
            "uncertain": int(bool(UNCERTAIN.search(s))),
            "correction": int(bool(CORRECT.search(s)))}


def role_flags(agent: str) -> dict[str, int]:
    low = agent.lower()
    return {"role_validator": int(bool(re.search(r"valid|verif|review|check|quality", low))),
            "role_tool": int(bool(re.search(r"computer|terminal|python|code|file|web", low)))}


def parse_one(path: Path, root: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    history = raw.get("history")
    if not isinstance(history, list):
        raise ValueError("history is not a list")
    messages = []
    agents = []
    for raw_index, item in enumerate(history):
        if not isinstance(item, dict):
            continue
        agent = speaker(item)
        if agent is None:
            continue
        if agent not in agents:
            agents.append(agent)
        content = str(item.get("content") or "")
        messages.append({"raw_index": raw_index, "agent": agent,
                         "content": content, "flags": flags(content)})
    if not agents:
        raise ValueError("no agent messages")
    culprit_raw = str(raw.get("mistake_agent") or "")
    matches = [i for i, a in enumerate(agents) if normalize(a) == normalize(culprit_raw)]
    if len(matches) != 1:
        raise ValueError(f"culprit does not uniquely match agents: {culprit_raw!r}")
    try:
        mistake_step = int(raw["mistake_step"])
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError("invalid mistake_step") from exc
    if not 0 <= mistake_step < len(history):
        raise ValueError("mistake_step outside history")
    label_message_matches = speaker(history[mistake_step]) == agents[matches[0]] if isinstance(history[mistake_step], dict) else False
    file_id = path.relative_to(root).as_posix()
    qid = str(raw.get("question_ID") or "")
    if not qid:
        qid = hashlib.sha256(str(raw.get("question", file_id)).encode()).hexdigest()
    return {"id": file_id, "subset": path.parent.name, "question_id": qid,
            "question": str(raw.get("question") or ""),
            "agents": agents, "culprit": matches[0], "culprit_name": agents[matches[0]],
            "mistake_step": mistake_step, "label_message_matches": label_message_matches,
            "raw_message_count": len(history), "messages": messages}


def parse_unlabeled(path: Path) -> dict:
    """Read a new trace without requiring or reading its attribution label."""
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    history = raw.get("history")
    if not isinstance(history, list):
        raise ValueError("history must be a list")
    agents, messages = [], []
    for raw_index, item in enumerate(history):
        if not isinstance(item, dict):
            continue
        agent = speaker(item)
        if agent is None:
            continue
        if agent not in agents:
            agents.append(agent)
        content = str(item.get("content") or "")
        messages.append({"raw_index": raw_index, "agent": agent,
                         "content": content, "flags": flags(content)})
    if not agents:
        raise ValueError("trace has no agent messages")
    evidence = [({f: int(any(m["flags"][f] for m in messages if m["agent"] == agent))
                  for f in FEATURES} | role_flags(agent)) for agent in agents]
    return {"agents": agents, "messages": messages, "evidence": evidence,
            "n_agents": len(agents), "raw_message_count": len(history)}


def load_records(root: Path) -> tuple[list[dict], list[dict]]:
    records, exclusions = [], []
    for path in sorted(root.rglob("*.json")):
        try:
            records.append(parse_one(path, root))
        except Exception as exc:
            exclusions.append({"id": path.relative_to(root).as_posix(), "reason": str(exc)})
    return records, exclusions


def split_records(records: list[dict], seed: int = 20260929) -> dict[str, list[str]]:
    """Group repeated questions and approximately stratify by subset/agent count."""
    groups = defaultdict(list)
    for r in records:
        groups[r["question_id"]].append(r)
    strata = defaultdict(list)
    for qid, group in groups.items():
        signature = (group[0]["subset"], len(group[0]["agents"]))
        strata[signature].append((qid, group))
    rng = random.Random(seed)
    splits = {"train": [], "validation": [], "test": []}
    for signature in sorted(strata):
        group_list = sorted(strata[signature])
        rng.shuffle(group_list)
        for idx, (_, group) in enumerate(group_list):
            # Use group index, not trace index: related questions stay together.
            fraction = (idx + 0.5) / len(group_list)
            part = "train" if fraction < 0.70 else "validation" if fraction < 0.85 else "test"
            splits[part].extend(r["id"] for r in group)
    return {k: sorted(v) for k, v in splits.items()}


def feature_rows(records: list[dict]) -> list[dict]:
    rows = []
    for r in records:
        evidence = []
        for agent in r["agents"]:
            authored = [m for m in r["messages"] if m["agent"] == agent]
            evidence.append({f: int(any(m["flags"][f] for m in authored)) for f in FEATURES}
                            | role_flags(agent))
        rows.append({"id": r["id"], "subset": r["subset"], "n_agents": len(r["agents"]),
                     "agents": r["agents"], "culprit": r["culprit"], "evidence": evidence})
    return rows


def audit(records: list[dict], exclusions: list[dict]) -> dict:
    return {"source_files": len(records) + len(exclusions), "parsed": len(records),
            "excluded": exclusions,
            "subsets": dict(Counter(r["subset"] for r in records)),
            "agent_counts": dict(sorted(Counter(len(r["agents"]) for r in records).items())),
            "label_step_matches_agent": sum(r["label_message_matches"] for r in records),
            "duplicate_question_groups": sum(n > 1 for n in Counter(r["question_id"] for r in records).values())}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
