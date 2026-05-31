"""Reciprocal Rank Fusion (RRF).

Two retrievers — pgvector (semantic) and Postgres full-text (keyword) — return
ranked lists whose raw scores live on different, uncalibrated scales (cosine in
[-1, 1] vs ts_rank, an unbounded relevance number). Averaging them is
meaningless. RRF sidesteps calibration entirely by fusing on RANK:

    rrf(memory) = Σ_sources 1 / (RRF_K + rank_in_source)

``RRF_K`` (~60, tunable in config) damps the contribution of low ranks so the
head of each list dominates. A memory near the top of either retriever scores
high; one near the top of BOTH scores highest. We also carry each source's raw
score so the caller can apply the noise gate (keyword present? cosine ≥ floor?).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Fused:
    memory: dict[str, Any]
    rrf_score: float = 0.0
    cosine: float | None = None      # best vector cosine, if vector-retrieved
    ts_rank: float | None = None     # best ts_rank, if keyword-retrieved
    sources: set[str] = field(default_factory=set)

    @property
    def keyword_hit(self) -> bool:
        return self.ts_rank is not None and self.ts_rank > 0


def rrf_fuse(
    semantic: list[dict],
    keyword: list[dict],
    k: int,
) -> list[Fused]:
    """Fuse two ranked memory lists (each a list of row-dicts with ``id`` and
    ``score``) into one list ordered by descending RRF score.

    ``semantic`` rows carry cosine in ``score``; ``keyword`` rows carry ts_rank
    in ``score``. Rank is 1-based, in the order each list was returned by SQL.
    """
    fused: dict[Any, Fused] = {}

    def _entry(row: dict) -> Fused:
        mid = row["id"]
        if mid not in fused:
            fused[mid] = Fused(memory=row)
        return fused[mid]

    for rank, row in enumerate(semantic, start=1):
        e = _entry(row)
        e.rrf_score += 1.0 / (k + rank)
        e.cosine = float(row["score"])
        e.sources.add("semantic")

    for rank, row in enumerate(keyword, start=1):
        e = _entry(row)
        e.rrf_score += 1.0 / (k + rank)
        e.ts_rank = float(row["score"])
        e.sources.add("keyword")

    return sorted(fused.values(), key=lambda f: f.rrf_score, reverse=True)
