#!/usr/bin/env python3
"""
Phase 1: Data Parsing — Who&When multi-agent LLM failure benchmark
==================================================================
Recursively parses every JSON trace in the dataset, extracts the conversation
history, cleans role names, derives Nodes/Edges (interaction topology), and
pulls the ground-truth answer keys (question, mistake_agent, mistake_step).

Outputs (in ./phase1_output/):
  - Phase1_Results.txt           one block per JSON file + dataset summary
  - phase1_parsed_records.json   machine-readable records (bonus artifact)
  - topology_*.png               representative topology diagrams
  - summary_topology_frequency.png / summary_most_common_edges.png

Role-cleaning rules (applied in order):
  1. If a message carries a non-empty `name` field, that is the agent
     (Algorithm-Generated schema: role is just "assistant"/"user").
  2. Otherwise use `role`; ignore messages whose role is "human"/"user"
     (case-insensitive) or contains "thought".
  3. Strip parenthetical annotations: "Orchestrator (-> WebSurfer)" ->
     "Orchestrator"; "Orchestrator (termination condition)" -> "Orchestrator".
Graph construction:
  - Consecutive messages from the same agent are collapsed into one turn.
  - Nodes = unique agents that spoke (order of first appearance).
  - Edges = unique directed arrows A -> B between consecutive speakers
    (transition counts are kept for the summary statistics).
"""

import collections
import json
import re
import textwrap
from pathlib import Path

import networkx as nx

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DATASET_ROOT = Path("Agents_Failure_Attribution/Who&When")
OUT_DIR = Path("phase1_output")
RESULTS_TXT = OUT_DIR / "Phase1_Results.txt"
RECORDS_JSON = OUT_DIR / "phase1_parsed_records.json"

HUMAN_ROLES = {"human", "user"}          # ignored (case-insensitive)
IGNORE_SUBSTRINGS = ("thought",)          # ignored if contained in role

NODE_COLOR = "#4C72B0"                    # steel blue
MISTAKE_COLOR = "#C44E52"                 # crimson
ARROW_COLOR = "#555555"


def clean_role(msg: dict):
    """Return the cleaned agent name for a history message, or None to ignore."""
    name = msg.get("name")
    if name is not None and str(name).strip():
        return str(name).strip()                     # Algorithm-Generated schema

    role = str(msg.get("role", "")).strip()
    low = role.lower()
    if low in HUMAN_ROLES:                           # human / user -> ignore
        return None
    if any(s in low for s in IGNORE_SUBSTRINGS):     # (thought) etc. -> ignore
        return None
    cleaned = re.sub(r"\s*\([^)]*\)", "", role).strip()   # drop "(-> X)" etc.
    return cleaned or None


def build_speaker_sequence(history: list):
    """Cleaned speakers in order, consecutive duplicates collapsed."""
    seq = []
    for msg in history:
        agent = clean_role(msg) if isinstance(msg, dict) else None
        if agent and (not seq or seq[-1] != agent):
            seq.append(agent)
    return seq


def extract_topology(seq: list):
    """Nodes (first-appearance order) and unique directed edges + transitions."""
    nodes = list(dict.fromkeys(seq))
    edges = list(dict.fromkeys(zip(seq, seq[1:]))) if len(seq) > 1 else []
    transitions = max(len(seq) - 1, 0)
    return nodes, edges, transitions


def build_graph(nodes: list, edges: list):
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G


def classify_topology(G: nx.DiGraph):
    """Structural label so heterogeneous agent-naming schemas are comparable."""
    n = G.number_of_nodes()
    if n == 0:
        return "empty (no agent messages)"
    if n == 1:
        return "1-agent (no interaction)"
    m = G.number_of_edges()
    indegs = [G.in_degree(v) for v in G.nodes]
    outdegs = [G.out_degree(v) for v in G.nodes]
    if (m == n - 1 and nx.is_weakly_connected(G) and nx.is_directed_acyclic_graph(G)
            and indegs.count(0) == 1 and outdegs.count(0) == 1
            and max(indegs) <= 1 and max(outdegs) <= 1):
        return f"{n}-agent chain"
    if (m == n and max(indegs) <= 1 and max(outdegs) <= 1
            and nx.is_strongly_connected(G)):
        return f"{n}-agent cycle"
    return f"{n}-agent other"


def sorted_json_files(root: Path):
    """All *.json under root, Hand-Crafted first, numeric-aware filename sort."""
    def sort_key(p: Path):
        subset = 0 if "Hand-Crafted" in p.parts else 1
        stem = p.stem
        num = int(stem) if stem.isdigit() else float("inf")
        return (subset, str(p.parent.name), num, stem)

    files = [p for p in root.rglob("*.json") if p.is_file()]
    return sorted(files, key=sort_key)


def wrap_label(s: str, width: int = 14):
    return "\n".join(textwrap.wrap(s, width)) or s


# --------------------------------------------------------------------------- #
# Parsing pass
# --------------------------------------------------------------------------- #
def parse_all():
    records = []
    parse_failures = []
    for path in sorted_json_files(DATASET_ROOT):
        rel = path.relative_to(DATASET_ROOT).as_posix()
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception as exc:                          # pragma: no cover
            parse_failures.append((rel, str(exc)))
            continue

        history = data.get("history") or []
        seq = build_speaker_sequence(history)
        nodes, edges, transitions = extract_topology(seq)
        G = build_graph(nodes, edges)

        records.append({
            "file_name": rel,
            "subset": rel.split("/")[0],
            "question": (data.get("question") or "").strip(),
            "nodes": nodes,
            "num_nodes": len(nodes),
            "edges": [f"{a} -> {b}" for a, b in edges],
            "num_edges": len(edges),
            "speaking_transitions": transitions,
            "topology": classify_topology(G),
            "mistake_agent": data.get("mistake_agent", "N/A"),
            "mistake_step": data.get("mistake_step", "N/A"),
            "mistake_reason": data.get("mistake_reason", ""),
            "ground_truth": data.get("ground_truth", ""),
            "message_count_raw": len(history),
        })
    return records, parse_failures


# --------------------------------------------------------------------------- #
# Results text file
# --------------------------------------------------------------------------- #
def write_results_txt(records, parse_failures):
    W = 100
    lines = []
    ap = lines.append

    ap("=" * W)
    ap("PHASE 1: DATA PARSING RESULTS — Who&When Multi-Agent LLM Failure Benchmark")
    ap("=" * W)
    ap(f"Dataset root          : {DATASET_ROOT}")
    ap(f"JSON files found      : {len(records) + len(parse_failures)} "
       f"(Hand-Crafted: {sum(1 for r in records if r['subset'] == 'Hand-Crafted')}, "
       f"Algorithm-Generated: {sum(1 for r in records if r['subset'] == 'Algorithm-Generated')})")
    ap(f"Successfully parsed   : {len(records)}")
    ap(f"Failed to parse       : {len(parse_failures)}")
    ap("Role-cleaning rules   : agent = `name` field if present, else `role`; "
       "ignored 'human'/'user' and roles containing 'thought';")
    ap("                        parenthetical annotations stripped "
       "(e.g. 'Orchestrator (-> WebSurfer)' -> 'Orchestrator').")
    ap("Graph construction    : consecutive same-agent turns collapsed; "
       "Nodes = unique speakers; Edges = unique directed arrows between")
    ap("                        consecutive speakers (A -> B means A spoke, then B spoke next).")
    ap("=" * W)
    ap("")

    for i, r in enumerate(records, 1):
        ap(f"{'#' * W}")
        ap(f"BLOCK {i} / {len(records)}")
        ap(f"{'#' * W}")
        ap(f"File Name                 : {r['file_name']}")
        ap("Question Asked            :")
        for ln in textwrap.wrap(r["question"] or "(missing)", W - 29):
            ap(f"    {ln}")
        ap(f"Nodes (Agents involved)   : {', '.join(r['nodes']) if r['nodes'] else '(none)'}")
        ap(f"Number of Nodes           : {r['num_nodes']}")
        ap("Edges (directed)          :")
        if r["edges"]:
            for e in r["edges"]:
                ap(f"    {e}")
        else:
            ap("    (none)")
        ap(f"Number of Edges           : {r['num_edges']}")
        ap(f"Topology Class            : {r['topology']}")
        ap(f"Ground Truth Mistake Agent: {r['mistake_agent']}")
        ap(f"Ground Truth Mistake Step : {r['mistake_step']}")
        ap("")

    # ---------------------------- summary section --------------------------- #
    ap("=" * W)
    ap("DATASET-LEVEL SUMMARY")
    ap("=" * W)

    topo_counts = collections.Counter(r["topology"] for r in records)
    ap("\nTopology class frequency (most common first):")
    for topo, cnt in topo_counts.most_common():
        ap(f"    {topo:<32} {cnt:>4} files ({cnt / len(records):5.1%})")

    node_dist = collections.Counter(r["num_nodes"] for r in records)
    ap("\nDistribution of number of agent nodes per trace:")
    for k in sorted(node_dist):
        ap(f"    {k} agents: {node_dist[k]:>4} files")

    edge_counter = collections.Counter()
    for r in records:
        edge_counter.update(r["edges"])
    ap("\nTop 15 most common directed edges across the dataset:")
    for e, cnt in edge_counter.most_common(15):
        ap(f"    {e:<70} {cnt:>4}")

    agent_counter = collections.Counter()
    for r in records:
        agent_counter.update(r["nodes"])
    ap("\nTop 15 agents by number of traces they appear in:")
    for a, cnt in agent_counter.most_common(15):
        ap(f"    {a:<55} {cnt:>4} traces")

    ma = collections.Counter(str(r["mistake_agent"]) for r in records)
    ap("\nGround-truth mistake agent distribution (top 15):")
    for a, cnt in ma.most_common(15):
        ap(f"    {a:<55} {cnt:>4}")

    if parse_failures:
        ap("\nFiles that failed to parse:")
        for rel, err in parse_failures:
            ap(f"    {rel}: {err}")

    ap("\n" + "=" * W)
    ap(f"END OF REPORT — {len(records)} file blocks written.")
    ap("=" * W)

    RESULTS_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Diagrams
# --------------------------------------------------------------------------- #
def _layout(G, topo):
    nodes = list(G.nodes)
    if "chain" in topo:
        start = next(v for v in nodes if G.in_degree(v) == 0)
        path, seen = [start], {start}
        while len(path) < len(nodes):
            nxt = next((w for w in G.successors(path[-1]) if w not in seen), None)
            if nxt is None:
                break
            path.append(nxt)
            seen.add(nxt)
        pos = {v: (i, 0) for i, v in enumerate(path)}
        for v in nodes:
            pos.setdefault(v, (len(path), 1))
    elif "cycle" in topo:
        pos = nx.circular_layout(G)
    else:
        pos = nx.kamada_kawai_layout(G) if len(nodes) <= 8 else nx.spring_layout(G, seed=42, k=1.8)
    return pos


def draw_topology(G, topo, mistake_agent, title, subtitle, out_path):
    pos = _layout(G, topo)
    fig_w = max(8.0, 2.1 * G.number_of_nodes())
    fig, ax = plt.subplots(figsize=(fig_w, 4.2))

    colors = [MISTAKE_COLOR if v == mistake_agent else NODE_COLOR for v in G.nodes]
    nx.draw_networkx_edges(
        G, pos, ax=ax, edge_color=ARROW_COLOR, width=1.8, arrows=True,
        arrowsize=22, arrowstyle="-|>", connectionstyle="arc3,rad=0.12",
        min_source_margin=18, min_target_margin=18,
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=colors, node_size=3400,
        edgecolors="white", linewidths=2.0, margins=0.18,
    )
    labels = {v: wrap_label(v) for v in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=8.5, font_weight="bold")

    ax.set_title(f"{title}\n{subtitle}", fontsize=11)
    legend = [Patch(facecolor=NODE_COLOR, label="Agent"),
              Patch(facecolor=MISTAKE_COLOR, label="Ground-truth mistake agent")]
    if mistake_agent not in G.nodes:
        legend.append(Patch(facecolor="white", edgecolor="gray",
                            label=f"'{mistake_agent}' not among speaking nodes"))
    ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.22),
              ncol=len(legend), fontsize=8.5, frameon=False)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_diagrams(records):
    by_topo = collections.defaultdict(list)
    for r in records:
        by_topo[r["topology"]].append(r)

    # Most common structural topologies -> one real representative each
    common = [t for t, _ in collections.Counter(
        r["topology"] for r in records).most_common() if not t.startswith(("empty", "1-agent"))][:4]

    diagram_paths = []
    for topo in common:
        rep = by_topo[topo][0]
        G = build_graph(rep["nodes"], [tuple(e.split(" -> ")) for e in rep["edges"]])
        n_files = len(by_topo[topo])
        safe = topo.replace(" ", "_").replace("(", "").replace(")", "")
        out = OUT_DIR / f"topology_{safe}__example_{Path(rep['file_name']).stem}.png"
        draw_topology(
            G, topo, str(rep["mistake_agent"]),
            f"Most-common topology: {topo}  ({n_files} of {len(records)} files, "
            f"{n_files / len(records):.1%})",
            f"Representative trace: {rep['file_name']}",
            out,
        )
        diagram_paths.append(out)

    # Explicit chain examples (user-requested): one representative per chain class
    chain_topos = sorted(t for t in by_topo if "chain" in t and t not in common)
    for topo in chain_topos:
        rep = by_topo[topo][0]
        G = build_graph(rep["nodes"], [tuple(e.split(" -> ")) for e in rep["edges"]])
        safe = topo.replace(" ", "_")
        out = OUT_DIR / f"topology_{safe}__example_{Path(rep['file_name']).stem}.png"
        draw_topology(
            G, topo, str(rep["mistake_agent"]),
            f"Chain topology example: {topo}  ({len(by_topo[topo])} of {len(records)} files)",
            f"Representative trace: {rep['file_name']}",
            out,
        )
        diagram_paths.append(out)

    # Extra: even-frequency summary charts
    # 1) topology frequency bar chart
    topo_counts = collections.Counter(r["topology"] for r in records)
    labels, vals = zip(*topo_counts.most_common())
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(labels) + 1.6))
    ax.barh(range(len(labels)), vals, color=NODE_COLOR)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Number of JSON traces")
    ax.set_title("Agent-topology frequency across the Who&When dataset")
    for i, v in enumerate(vals):
        ax.text(v, i, f"  {v}", va="center", fontsize=9)
    fig.tight_layout()
    p1 = OUT_DIR / "summary_topology_frequency.png"
    fig.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close(fig)
    diagram_paths.append(p1)

    # 2) most common directed edges bar chart
    edge_counter = collections.Counter()
    for r in records:
        edge_counter.update(r["edges"])
    top = edge_counter.most_common(15)
    labels = [e for e, _ in top][::-1]
    vals = [c for _, c in top][::-1]
    fig, ax = plt.subplots(figsize=(10, 0.38 * len(labels) + 1.6))
    ax.barh(range(len(labels)), vals, color="#55A868")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Occurrences across all traces")
    ax.set_title("Top 15 most common directed agent-to-agent edges")
    for i, v in enumerate(vals):
        ax.text(v, i, f"  {v}", va="center", fontsize=9)
    fig.tight_layout()
    p2 = OUT_DIR / "summary_most_common_edges.png"
    fig.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close(fig)
    diagram_paths.append(p2)

    return diagram_paths


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    OUT_DIR.mkdir(exist_ok=True)

    records, parse_failures = parse_all()
    print(f"Parsed {len(records)} files ({len(parse_failures)} failures).")

    write_results_txt(records, parse_failures)
    print(f"Wrote {RESULTS_TXT}")

    RECORDS_JSON.write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {RECORDS_JSON}")

    diagram_paths = make_diagrams(records)
    for p in diagram_paths:
        print(f"Wrote {p}")

    blocks = sum(
        1 for ln in RESULTS_TXT.read_text(encoding="utf-8").splitlines()
        if ln.startswith("BLOCK ")
    )
    print(f"Sanity check: {blocks} BLOCK entries in results file "
          f"(expected {len(records)}).")


if __name__ == "__main__":
    main()
