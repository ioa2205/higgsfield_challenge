"""Entity population: extract and persist entities from memories (Phase 4).

Populates the ``entities`` and ``memory_entities`` tables from the slot keys
and values that the extraction layer already produces.  Entity types mirror the
four types the schema already defines: ``user``, ``pet``, ``employer``, ``city``.

Entry point: ``populate_entities(conn, memory_id, draft, user_id)``.
"""
from .populate import populate_entities

__all__ = ["populate_entities"]
