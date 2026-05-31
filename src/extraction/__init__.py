"""Memory extraction from conversation turns.

Phase 2 ships HYBRID extraction: a provider-agnostic LLM path (``llm.py``) with
a deterministic regex fallback (``rules.py``), orchestrated synchronously by
``pipeline.extract``. Both paths emit the same typed ``MemoryDraft`` shape
(``draft.py``). The legacy single-``event`` ``naive.py`` is gone.
"""
from .draft import MemoryDraft
from .pipeline import extract

# Backwards-compatible alias: routes historically imported ``extract_memories``.
extract_memories = extract

__all__ = ["MemoryDraft", "extract", "extract_memories"]
