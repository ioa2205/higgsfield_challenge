"""Embedding backends.

The product uses ``FastEmbedEmbedder`` (BAAI/bge-small-en-v1.5, 384-dim) loaded
fully offline from a cache dir baked into the image at build time. The test
suite uses ``FakeEmbedder`` -- a deterministic, dependency-free hashing
embedder -- so pytest needs no model download and no network. Both produce
``EMBED_DIM``-length unit vectors, so the DB column and cosine math are
identical either way.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

from .. import config

_TOKEN = re.compile(r"[a-z0-9]+")


class BaseEmbedder:
    dim = config.EMBED_DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class FakeEmbedder(BaseEmbedder):
    """Deterministic bag-of-words hashing embedder (offline, no deps).

    Shared tokens between two texts produce overlapping non-zero dimensions, so
    cosine similarity is meaningful enough for contract/recall tests without a
    real model.
    """

    backend = "fake"

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            for token in _TOKEN.findall((text or "").lower()):
                idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
                vec[idx] += 1.0
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec /= norm
            out.append(vec.tolist())
        return out


class FastEmbedEmbedder(BaseEmbedder):
    """Real bge-small embedder via fastembed (ONNX), loaded offline."""

    backend = "fastembed"

    def __init__(self) -> None:
        from fastembed import TextEmbedding

        self.model = TextEmbedding(
            model_name=config.EMBED_MODEL,
            cache_dir=config.EMBED_CACHE_DIR,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self.model.embed(list(texts))]


def get_embedder() -> BaseEmbedder:
    if config.EMBED_BACKEND == "fake":
        return FakeEmbedder()
    return FastEmbedEmbedder()
