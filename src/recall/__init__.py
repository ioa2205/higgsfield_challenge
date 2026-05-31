"""Recall: query → assembled context + citations.

Phase 3 ships HYBRID retrieval (pgvector cosine + Postgres full-text) fused by
RRF (``fusion``), then a tiered, token-budgeted assembly (``assembly``) with a
noise gate. ``service.run_recall`` is the orchestrator the API calls;
``retrieval`` pulls the candidate sets from the single Postgres store.
"""
from .service import run_recall

__all__ = ["run_recall"]
