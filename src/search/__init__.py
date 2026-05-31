"""Search: raw semantic results, no context assembly.

Phase 1 ships naive cosine. Phases 2+ extend this package (full-text + hybrid)
in place.
"""
from .naive import format_results

__all__ = ["format_results"]
