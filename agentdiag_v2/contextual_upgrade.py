"""Reproducible external-validation selection for contextual attribution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .benchmark_step import exact_metrics, bootstrap_interval
from .src.data import load_records, parse_unlabeled, write_json, write_jsonl
from .src import contextual_sparse
from .src.step_ranker import predict as baseline_predict

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "contextual_upgrade"
DEFAULT_CACHE = HERE.parent.parent / "agentdiag-model-cache"
DEFAULT_EXTERNAL = HERE.parent.parent / "StepFinder/data"
L2_GRID = (0.5, 2.0)
SEEDS = (13, 29)
EPOCHS = (8, 16, 24)
CONFIGS = {
    "temporal_exact": {"model": {"hidden": 64, "dropout": 0.25, "temporal": True}, "sigma": 0},
    "temporal_distance": {"model": {"hidden": 64, "dropout": 0.25, "temporal": True}, "sigma": 1},
    "attention_exact": {"model": {"hidden": 64, "dropout": 0.25, "temporal": False}, "sigma": 0},
}


def external_data(path: Path):
    manifest = json.loads((HERE / "data/stepfinder_safe_questions.json").read_text())
    safe = set(manifest["safe_question_sha256"])
    rows, _ = load_records(path)
    rows = [r for r in rows if "/train/" in r["id"] and r["label_message_matches"]
            and 2 <= len(r["agents"]) <= 4
            and hashlib.sha256(r["question"].strip().encode()).hexdigest() in safe]
    if len(rows) != 2004 or len({r["question"].strip() for r in rows}) != 58:
        raise ValueError("External snapshot differs from the audited source")
    folds = json.loads((HERE / "benchmark_step/external_training_folds.json").read_text())
    mapping = folds["question_to_fold"]
    if not all(r["question_id"] in mapping for r in rows):
        raise ValueError("External fold manifest no longer matches data")
    return rows, mapping, manifest


def serialize(rows, predictions):
    return [{"id": r["id"], "subset": r["id"].split("/", 1)[0],
             "question_id": r["question_id"], "actual_agent": r["culprit_name"],
             "actual_step": r["mistake_step"], **p} for r, p in zip(rows, predictions)]


def sparse_phase(path: Path, output: Path):
    rows, mapping, _ = external_data(path)
    all_results = {}
    for l2 in L2_GRID:
        pairs = {}
        for fold in range(5):
            training = [r for r in rows if mapping[r["question_id"]] != fold]
            validation = [r for r in rows if mapping[r["question_id"]] == fold]
            model = contextual_sparse.fit(training, l2)
            predictions = contextual_sparse.predict(model, validation)
            pairs.update({r["id"]: p for r, p in zip(validation, predictions)})
            print(f"Sparse L2={l2} fold={fold}: {exact_metrics(validation, predictions)}", flush=True)
        predictions = [pairs[r["id"]] for r in rows]
        name = f"sparse_{l2}"
        write_jsonl(output / f"cv_{name}.jsonl", serialize(rows, predictions))
        all_results[name] = exact_metrics(rows, predictions)
        write_json(output / "sparse_cv_metrics.json", all_results)
    return all_results


def embeddings_for(rows, cache: Path):
    from .src.semantic_encoder import SemanticEncoder, record_texts
    texts = [text for r in rows for text in record_texts(r)]
    return SemanticEncoder(cache).encode(texts)


def neural_phase(path: Path, output: Path, cache: Path):
    import torch
    from .src.contextual_ranker import prepare, fit
    torch.set_num_threads(4)
    rows, mapping, _ = external_data(path)
    embeddings = embeddings_for(rows, cache)
    prepared = {r["id"]: prepare(r, embeddings) for r in rows}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics_file = output / "neural_cv_metrics.json"
    all_metrics = json.loads(metrics_file.read_text()) if metrics_file.exists() else {}
    for name, config in CONFIGS.items():
        for seed in SEEDS:
            paths = [output / f"cv_{name}_{seed}_{epoch}.jsonl" for epoch in EPOCHS]
            if all(p.exists() for p in paths):
                print(f"Using completed CV {name} seed={seed}", flush=True)
                continue
            pairs = {epoch: {} for epoch in EPOCHS}
            for fold in range(5):
                training = [prepared[r["id"]] for r in rows if mapping[r["question_id"]] != fold]
                validation_rows = [r for r in rows if mapping[r["question_id"]] == fold]
                validation = [prepared[r["id"]] for r in validation_rows]
                _, checkpoints = fit(training, config, seed, max(EPOCHS), device,
                                     validation=validation, checkpoints=EPOCHS)
                for epoch, outputs in checkpoints.items():
                    pairs[epoch].update({r["id"]: p for r, p in zip(validation_rows, outputs)})
                    fold_metrics = exact_metrics(validation_rows, outputs)
                    algorithm = fold_metrics["Algorithm-Generated"]
                    progress = {"config": name, "seed": seed, "fold": fold, "epoch": epoch,
                                "metrics": fold_metrics}
                    with (output / "training_progress.jsonl").open("a", encoding="utf-8") as file:
                        file.write(json.dumps(progress) + "\n")
                    print(f"{name} seed={seed} fold={fold} epoch={epoch}: "
                          f"Algorithm agent={algorithm['agent_accuracy']:.3f} "
                          f"step={algorithm['step_accuracy']:.3f}", flush=True)
            for epoch in EPOCHS:
                outputs = [pairs[epoch][r["id"]] for r in rows]
                key = f"{name}_{seed}_{epoch}"
                write_jsonl(output / f"cv_{key}.jsonl", serialize(rows, outputs))
                all_metrics[key] = exact_metrics(rows, outputs)
            write_json(output / "neural_cv_metrics.json", all_metrics)
    return all_metrics


def encoding_manifest(rows, cache: Path, output: Path):
    from .src.semantic_encoder import MODEL, REVISION, record_texts, cache_key
    texts = sorted({text for r in rows for text in record_texts(r)})
    missing = [t for t in texts if not (cache / "vectors" / (cache_key(t) + ".npy")).exists()]
    if missing:
        raise ValueError(f"Encoding cache is incomplete: {len(missing)} missing texts")
    metadata = cache / "encoder/.cache/huggingface/download/model.safetensors.metadata"
    weight_metadata = metadata.read_text().splitlines() if metadata.exists() else []
    write_json(output / "encoding_manifest.json", {"model": MODEL, "revision": REVISION,
               "downloaded_weights_etag": weight_metadata[1] if len(weight_metadata) > 1 else None,
               "embedding_dimension": 256, "max_tokens_per_text": 384,
               "long_text_policy": "First 288 and last 96 encoder tokens. Lexical comparator reads up to 1,000 word tokens per text block.",
               "unique_observed_texts": len(texts), "training_traces": len(rows),
               "text_keys_sha256": hashlib.sha256("\n".join(cache_key(t) for t in texts).encode()).hexdigest(),
               "ground_truth_answers_used": False, "attribution_labels_encoded": False})


def read_cv(path: Path, rows):
    values = {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}
    if set(values) != {r["id"] for r in rows}:
        raise ValueError(f"CV prediction IDs do not match: {path}")
    return [values[r["id"]] for r in rows]


def blend(rows, first, second, weight: float, head: str):
    out = []
    key = head + "_probabilities"
    for r, a, b in zip(rows, first, second):
        p = (1 - weight) * np.asarray(a[key]) + weight * np.asarray(b[key])
        value = dict(a)
        value[key] = p.tolist()
        value["predicted_" + head] = (r["agents"][int(p.argmax())] if head == "agent"
                                      else r["messages"][int(p.argmax())]["raw_index"])
        out.append(value)
    return out


def select_phase(path: Path, output: Path):
    rows, _, source = external_data(path)
    baseline = read_cv(HERE / "benchmark_step/external_cv_predictions_l2_0_45.jsonl", rows)
    candidates = {"baseline": baseline}
    for l2 in L2_GRID:
        p = output / f"cv_sparse_{l2}.jsonl"
        if p.exists():
            candidates[f"sparse_{l2}"] = read_cv(p, rows)
    for name in CONFIGS:
        for epoch in EPOCHS:
            paths = [output / f"cv_{name}_{seed}_{epoch}.jsonl" for seed in SEEDS]
            if all(p.exists() for p in paths):
                pairs = [read_cv(p, rows) for p in paths]
                average = blend(rows, pairs[0], pairs[1], 0.5, "step")
                average = blend(rows, average, pairs[1], 0.5, "agent")
                candidates[f"{name}_{epoch}"] = average
    # A fixed equal lexical/semantic mixture is another predeclared view.
    neural_names = [name for name in candidates if name != "baseline" and not name.startswith("sparse_")]
    for name in neural_names:
        for l2 in L2_GRID:
            lexical = f"sparse_{l2}"
            if lexical in candidates:
                average = blend(rows, candidates[name], candidates[lexical], 0.5, "step")
                average = blend(rows, average, candidates[lexical], 0.5, "agent")
                candidates[f"mix__{name}__{lexical}"] = average
    selected, scores = {}, {}
    for head in ("agent", "step"):
        # Each head is selected independently, using only Algorithm-Generated OOF labels.
        best = None
        for name, outputs in candidates.items():
            for weight in (0, 0.25, 0.5, 0.75, 1):
                mixed = blend(rows, baseline, outputs, weight, head)
                metrics = exact_metrics(rows, mixed)
                key = head + "_accuracy"
                score = (metrics["Algorithm-Generated"][key], metrics["Hand-Crafted"][key], -weight)
                scores[f"{head}:{name}:{weight}"] = {"selection": score, "metrics": metrics}
                if best is None or score > best[0]:
                    best = (score, name, weight)
        selected[head] = {"candidate": best[1], "weight": best[2], "selection_score": best[0]}
    selected_outputs = [dict(p) for p in baseline]
    for head, selection in selected.items():
        outputs = blend(rows, baseline, candidates[selection["candidate"]], selection["weight"], head)
        for final, prediction in zip(selected_outputs, outputs):
            final[head + "_probabilities"] = prediction[head + "_probabilities"]
            final["predicted_" + head] = prediction["predicted_" + head]
    write_jsonl(output / "cv_selected.jsonl", serialize(rows, selected_outputs))
    freeze = {"selection": selected, "candidate_names": list(candidates),
              "selected_external_cv_metrics": exact_metrics(rows, selected_outputs),
              "external_source": source, "seeds": SEEDS, "epochs": EPOCHS,
              "configs": CONFIGS, "training_trace_ids_sha256": hashlib.sha256(
                  "\n".join(sorted(r["id"] for r in rows)).encode()).hexdigest(),
              "protocol": "Head-specific blending selected only on external question-group OOF accuracy. Algorithm-Generated primary, Hand-Crafted tie breaker. No Who&When target labels selected this configuration."}
    freeze["cv_file_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(output.glob("cv_*.jsonl"))}
    freeze["baseline_cv_sha256"] = hashlib.sha256(
        (HERE / "benchmark_step/external_cv_predictions_l2_0_45.jsonl").read_bytes()).hexdigest()
    freeze["fold_manifest_sha256"] = hashlib.sha256(
        (HERE / "benchmark_step/external_training_folds.json").read_bytes()).hexdigest()
    freeze["source_code_sha256"] = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        for name in ("src/data.py", "src/contextual_ranker.py", "src/contextual_sparse.py", "src/semantic_encoder.py")}
    write_json(output / "selection.json", freeze)
    write_json(output / "selection_scores.json", scores)
    print(json.dumps(selected, indent=2), flush=True)
    return freeze


def selected_components(frozen: dict) -> set[str]:
    names = set()
    for selection in frozen["selection"].values():
        if selection["weight"] > 0:
            name = selection["candidate"]
            names.update(name.split("__")[1:] if name.startswith("mix__") else [name])
    return names


def train_selected(path: Path, output: Path, cache: Path):
    import torch
    from .src.contextual_ranker import prepare, fit
    torch.set_num_threads(4)
    frozen = json.loads((output / "selection.json").read_text())
    rows, _, _ = external_data(path)
    names = selected_components(frozen)
    for name in names:
        if name.startswith("sparse_"):
            write_json(output / f"model_{name}.json", contextual_sparse.fit(rows, float(name.split("_")[1])))
    neural = [name for name in names if name != "baseline" and not name.startswith("sparse_")]
    if neural:
        embeddings = embeddings_for(rows, cache)
        prepared = [prepare(r, embeddings) for r in rows]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for name in neural:
            config_name, epoch = name.rsplit("_", 1)
            for seed in SEEDS:
                model, _ = fit(prepared, CONFIGS[config_name], seed, int(epoch), device)
                torch.save({"state_dict": model.cpu().state_dict(), "config": model.config,
                            "training": {"external_only": True, "seed": seed, "epochs": int(epoch)}},
                           output / f"model_{name}_{seed}.pt")


def predict_selected(rows, output: Path, cache: Path):
    frozen = json.loads((output / "selection.json").read_text())
    baseline_model = json.loads((HERE / "benchmark_step/step_model.json").read_text())
    baseline = baseline_predict(baseline_model, rows)
    candidates = {"baseline": baseline}
    names = selected_components(frozen)
    for name in names:
        if name.startswith("sparse_"):
            model = json.loads((output / f"model_{name}.json").read_text())
            candidates[name] = contextual_sparse.predict(model, rows)
    neural = [name for name in names if name != "baseline" and not name.startswith("sparse_")]
    if neural:
        import torch
        from .src.contextual_ranker import ContextualRanker, prepare, predict
        embeddings = embeddings_for(rows, cache)
        prepared = [prepare(r, embeddings) for r in rows]
        for name in neural:
            results = []
            for seed in SEEDS:
                saved = torch.load(output / f"model_{name}_{seed}.pt", map_location="cpu", weights_only=True)
                model = ContextualRanker(**saved["config"])
                model.load_state_dict(saved["state_dict"])
                results.append(predict(model, prepared))
            average = blend(rows, results[0], results[1], 0.5, "step")
            candidates[name] = blend(rows, average, results[1], 0.5, "agent")
    for selection in frozen["selection"].values():
        name = selection["candidate"]
        if selection["weight"] > 0 and name.startswith("mix__"):
            first, second = name.split("__")[1:]
            average = blend(rows, candidates[first], candidates[second], 0.5, "step")
            candidates[name] = blend(rows, average, candidates[second], 0.5, "agent")
    result = baseline
    for head, selection in frozen["selection"].items():
        mixed = blend(rows, baseline, candidates.get(selection["candidate"], baseline), selection["weight"], head)
        for final, p in zip(result, mixed):
            final[head + "_probabilities"] = p[head + "_probabilities"]
            final["predicted_" + head] = p["predicted_" + head]
    return result


def evaluate_phase(output: Path, cache: Path):
    # Selection and final checkpoints must exist before original label loading here.
    selected_file = output / "selection.json"
    selection_hash = hashlib.sha256(selected_file.read_bytes()).hexdigest()
    original, exclusions = load_records(HERE / "data/Who&When")
    if len(original) != 184 or exclusions:
        raise ValueError("Original benchmark changed")
    unlabeled = [parse_unlabeled(HERE / "data/Who&When" / r["id"]) for r in original]
    predictions = predict_selected(unlabeled, output, cache)
    old = read_cv(HERE / "benchmark_step/full_benchmark_predictions.jsonl", original)
    metrics, previous = exact_metrics(original, predictions), exact_metrics(original, old)
    changes = {}
    for subset in ("Algorithm-Generated", "Hand-Crafted"):
        group = [(r, p, b) for r, p, b in zip(original, predictions, old) if r["subset"] == subset]
        changes[subset] = {}
        for head in ("agent", "step"):
            target = "culprit_name" if head == "agent" else "mistake_step"
            gains = sum(p["predicted_" + head] == r[target] and b["predicted_" + head] != r[target] for r, p, b in group)
            losses = sum(p["predicted_" + head] != r[target] and b["predicted_" + head] == r[target] for r, p, b in group)
            changes[subset][head] = {"gains": gains, "losses": losses}
            metrics[subset][head + "_95pct_interval"] = bootstrap_interval(
                [r for r, _, _ in group], [p for _, p, _ in group], head, 103)
    write_jsonl(output / "benchmark_predictions.jsonl", serialize(original, predictions))
    results = {"old": previous, "new": metrics, "paired_changes": changes,
               "selection_sha256": selection_hash,
               "limitation": "Previously inspected original corpus; exploratory, not an independent paper-matched win. Strict exact names and raw steps, no ground-truth answers at inference."}
    write_json(output / "metrics.json", results)
    print(json.dumps(results, indent=2), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("sparse", "encode", "neural", "select", "train", "evaluate"), required=True)
    parser.add_argument("--stepfinder-data", type=Path, default=DEFAULT_EXTERNAL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase == "sparse":
        sparse_phase(args.stepfinder_data, args.output)
    elif args.phase == "encode":
        rows, _, _ = external_data(args.stepfinder_data)
        embeddings_for(rows, args.cache)
        encoding_manifest(rows, args.cache, args.output)
    elif args.phase == "neural":
        neural_phase(args.stepfinder_data, args.output, args.cache)
    elif args.phase == "select":
        select_phase(args.stepfinder_data, args.output)
    elif args.phase == "train":
        train_selected(args.stepfinder_data, args.output, args.cache)
    else:
        evaluate_phase(args.output, args.cache)


if __name__ == "__main__":
    main()
