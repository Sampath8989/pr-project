"""Pinned local semantic encoder with a label-free, content-addressed cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

MODEL = "Qwen/Qwen3-Embedding-0.6B"
REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"


def cache_key(text: str, dimension: int = 256, max_tokens: int = 384) -> str:
    config = [MODEL, REVISION, dimension, max_tokens, "head-tail-v1", text]
    return hashlib.sha256(json.dumps(config, ensure_ascii=False).encode()).hexdigest()


class SemanticEncoder:
    def __init__(self, cache: Path, dimension: int = 256, max_tokens: int = 384,
                 device: str | None = None):
        if not 32 <= dimension <= 1024 or max_tokens < 32:
            raise ValueError("Invalid Qwen embedding dimension or token budget")
        self.cache, self.dimension, self.max_tokens = Path(cache), dimension, max_tokens
        self.cache.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.model = self.tokenizer = None

    def _load(self):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        local = self.cache / "encoder"
        if not (local / "model.safetensors").is_file():
            from huggingface_hub import snapshot_download
            snapshot_download(MODEL, revision=REVISION, local_dir=str(local), allow_patterns=[
                "config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                "vocab.json", "merges.txt"])
        source = str(local)
        self.tokenizer = AutoTokenizer.from_pretrained(
            source, revision=REVISION, cache_dir=str(self.cache / "hub"),
            padding_side="left", trust_remote_code=False)
        self.model = AutoModel.from_pretrained(
            source, revision=REVISION, cache_dir=str(self.cache / "hub"),
            torch_dtype=dtype, attn_implementation="sdpa", trust_remote_code=False)
        self.model.to(self.device).eval()

    def encode(self, texts: list[str], batch_size: int = 8) -> dict[str, np.ndarray]:
        if not texts:
            return {}
        import torch
        unique = list(dict.fromkeys(texts))
        result, missing = {}, []
        packed_path = self.cache / f"packed-{REVISION}-{self.dimension}-{self.max_tokens}.npz"
        packed = {}
        if packed_path.exists():
            with np.load(packed_path, allow_pickle=False) as saved:
                keys, values = saved["keys"], saved["values"]
                if values.shape != (len(keys), self.dimension) or not np.isfinite(values).all():
                    raise ValueError("Invalid packed embedding cache")
                packed = dict(zip(keys.tolist(), values))
        vectors = self.cache / "vectors"
        vectors.mkdir(exist_ok=True)
        for text in unique:
            key = cache_key(text, self.dimension, self.max_tokens)
            path = vectors / (key + ".npy")
            if not text.strip():
                result[text] = np.zeros(self.dimension, dtype=np.float32)
                np.save(path, result[text], allow_pickle=False)
            elif key in packed:
                result[text] = packed[key]
            elif path.exists():
                value = np.load(path, allow_pickle=False)
                if value.shape != (self.dimension,) or not np.isfinite(value).all():
                    raise ValueError(f"Invalid cached embedding: {path}")
                result[text] = value
            else:
                missing.append(text)
        if missing and self.model is None:
            self._load()
        # Sort by length to reduce padding. The output dictionary preserves lookup by text.
        missing.sort(key=len)
        for start in range(0, len(missing), batch_size):
            batch = missing[start:start + batch_size]
            tokens = self.tokenizer(batch, add_special_tokens=True, truncation=False)
            items = []
            for ids in tokens["input_ids"]:
                if len(ids) > self.max_tokens:
                    head = self.max_tokens * 3 // 4
                    ids = ids[:head] + ids[-(self.max_tokens - head):]
                items.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
            inputs = self.tokenizer.pad(items, padding=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                hidden = self.model(**inputs).last_hidden_state[:, -1, :self.dimension].float()
                embeddings = torch.nn.functional.normalize(hidden, p=2, dim=-1).cpu().numpy()
            for text, value in zip(batch, embeddings):
                result[text] = value
                np.save(vectors / (cache_key(text, self.dimension, self.max_tokens) + ".npy"),
                        value, allow_pickle=False)
            if start % (batch_size * 32) == 0 or start + batch_size >= len(missing):
                print(f"Encoded {min(start + batch_size, len(missing))}/{len(missing)} new texts", flush=True)
        updates = {cache_key(t, self.dimension, self.max_tokens): v for t, v in result.items()}
        if not packed_path.exists() or any(key not in packed for key in updates):
            packed.update(updates)
            keys = sorted(packed)
            temporary = packed_path.with_suffix(".tmp")
            with temporary.open("wb") as file:
                np.savez_compressed(file, keys=np.asarray(keys), values=np.stack([packed[k] for k in keys]))
            temporary.replace(packed_path)
        return result


def record_texts(record: dict) -> list[str]:
    return [record.get("question", "")] + list(record["agents"]) + [
        message["content"] for message in record["messages"]]
