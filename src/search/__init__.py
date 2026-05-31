"""Search: structured hybrid lookup for the agent-invoked /search tool.

Phase 3 ships hybrid retrieval (full-text + vector) scoped by user/session.
``run_search`` is the orchestrator the API calls; ``format_results`` shapes the
§3 result rows.
"""
from .search import format_results, run_search

__all__ = ["run_search", "format_results"]
