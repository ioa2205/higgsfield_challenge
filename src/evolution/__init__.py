"""Fact evolution: slot-based supersession (Phase 4).

When a newly ingested memory describes the same slot as an existing ACTIVE
memory for the same user, the old memory is marked inactive and the new one's
``supersedes`` pointer is set to it — keeping the history intact without
deleting any rows.

Two triggers:

1. **Exact key match** (primary, deterministic): same canonical slot key
   (``employment``, ``location``, etc.) for the same user.  Always fires when
   the extraction layer used the canonical vocabulary (Phase 2+).

2. **Fuzzy embedding match** (safety net): same memory *type* AND cosine
   similarity ≥ ``SUPERSESSION_SIM_THRESHOLD`` — for cases where the LLM
   produced a slightly variant key or the rule extractor mapped to a generic
   key.  Threshold is tuned high (default 0.92) to avoid false merges.

Opinion arcs — "opinion.*" keys — are handled by the SAME mechanism.  Each
new opinion for the same subject supersedes the previous; the chain is
preserved so Tier-1 can render "updated; previously …" and the history is
inspectable via /users/{id}/memories.

Entry point: ``apply_supersession(conn, draft, new_memory_id, user_id, embedding)``.
"""
from .supersede import apply_supersession, is_unchanged_active_memory

__all__ = ["apply_supersession", "is_unchanged_active_memory"]
