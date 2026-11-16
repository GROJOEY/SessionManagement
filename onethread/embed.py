"""Embeddings. Uses a local multilingual model when available, otherwise a
character n-gram hashing embedder that needs no download (fine for demos/tests)."""

from __future__ import annotations

import hashlib
import os
import re
import sys

import numpy as np


# Function characters that match everywhere and say nothing about the topic.
STOP_CHARS = set("的了是在有和与及或也就都而但我你他她它们这那个一不没很太把被给到对从为以于之其等"
                 "年月日号吗呢吧啊呀么着过上下中里后前时会要能可想说做用好还再又才只已比更最")


class HashEmbedder:
    """Bag of character 1-3 grams hashed into a fixed vector. No download, works offline
    for Chinese and English. Digits, punctuation and function characters are dropped so
    that '80 公斤' and '2026 年' don't make unrelated notes look alike."""

    # lexical overlap only, so the bar is low; Claude reads the topic index for the semantics
    recall_floor = 0.03
    name = "hash-v2"

    def __init__(self, dim: int = 8192):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        t = re.sub(r"[\d\s\W_]+", " ", text.lower())
        for n in (1, 2, 3):
            for i in range(len(t) - n + 1):
                gram = t[i : i + n]
                if " " in gram:
                    continue
                if n == 1 and (gram in STOP_CHARS or gram.isascii()):
                    continue
                h = int(hashlib.md5(gram.encode()).hexdigest(), 16)
                v[h % self.dim] += 0.8 if n == 1 else 1.0
        norm = np.linalg.norm(v)
        return v / norm if norm else v


class STEmbedder:
    """sentence-transformers model, e.g. intfloat/multilingual-e5-small."""

    # e5-style models score almost everything 0.7+, so the floor sits higher
    recall_floor = 0.8

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # lazy import

        self.model = SentenceTransformer(model_name)
        self.prefix = "query: " if "e5" in model_name else ""
        self.name = model_name

    def embed(self, text: str) -> np.ndarray:
        v = self.model.encode(self.prefix + text, normalize_embeddings=True)
        return np.asarray(v, dtype=np.float32)


def make_embedder():
    """EMBED_MODEL=auto (default): use the multilingual model if sentence-transformers is
    installed, otherwise the built-in hash embedder. 'hash' or a model name forces one."""
    name = os.environ.get("EMBED_MODEL", "auto")
    if name == "hash":
        return HashEmbedder()
    if name == "auto":
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            return HashEmbedder()
        name = "intfloat/multilingual-e5-small"
    try:
        return STEmbedder(name)
    except Exception as e:  # no package or no network: fall back so the app still runs
        print(f"[onethread] embedding model '{name}' unavailable ({e}); using hash embedder", file=sys.stderr)
        return HashEmbedder()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or a.shape != b.shape:
        return 0.0
    return float(np.dot(a, b))
