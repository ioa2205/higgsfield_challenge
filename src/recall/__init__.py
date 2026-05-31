"""Recall: turn a query into assembled context + citations.

Phase 1 ships naive cosine top-k. Phases 2+ extend this package (hybrid
retrieval, RRF, priority-tier assembly) in place.
"""
from .naive import assemble_context

__all__ = ["assemble_context"]
