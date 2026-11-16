"""Embeddings. Uses a local multilingual model when available, otherwise a
character n-gram hashing embedder that needs no download (fine for demos/tests)."""

from __future__ import annotations

import hashlib
import os

import numpy as np


class HashEmbedder:
    """Bag of character 1-3 grams hashed into a fixed vector. Crude but works
    offline for Chinese and English; good enough to prove the pipeline."""

    related_threshold = 0.55  # similarity above which another topic counts as "related"

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        t = text.lower()
        for n in (1, 2, 3):
            for i in range(len(t) - n + 1):
                gram = t[i : i + n]
                if gram.strip():
                    h = int(hashlib.md5(gram.encode()).hexdigest(), 16)
                    v[h % self.dim] += 1.0 if n == 1 else 2.0
        norm = np.linalg.norm(v)
        return v / norm if norm else v


class STEmbedder:
    """sentence-transformers model, e.g. intfloat/multilingual-e5-small."""

    # e5-style models score almost everything 0.7+, so "related" needs a high bar
    related_threshold = 0.86

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # lazy import

        self.model = SentenceTransformer(model_name)
        self.prefix = "query: " if "e5" in model_name else ""

    def embed(self, text: str) -> np.ndarray:
        v = self.model.encode(self.prefix + text, normalize_embeddings=True)
        return np.asarray(v, dtype=np.float32)


def make_embedder():
    name = os.environ.get("EMBED_MODEL", "intfloat/multilingual-e5-small")
    if name == "hash":
        return HashEmbedder()
    try:
        return STEmbedder(name)
    except Exception as e:  # no package or no network: fall back so the app still runs
        print(f"[onethread] embedding model '{name}' unavailable ({e}); using hash embedder")
        return HashEmbedder()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or a.shape != b.shape:
        return 0.0
    return float(np.dot(a, b))
