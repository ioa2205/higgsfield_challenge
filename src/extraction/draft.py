"""The normalized memory shape every extraction path emits.

Both the LLM path (``llm.py``) and the deterministic rule path (``rules.py``)
produce ``MemoryDraft``s with the SAME fields, so ``pipeline.py`` and the
persistence layer never need to know which path produced a memory. Provenance
(``provenance``) records *which path* a memory came from — the per-path half of
the §4 provenance requirement (the other half, ``source_session`` /
``source_turn``, is stamped at insert time from the request).

Canonical slot keys
-------------------
A memory's ``key`` is a *canonical slot* so a later phase can supersede two
memories about the same topic (e.g. two ``employment`` facts). The vocabulary is
small and documented here so extraction and supersession agree on names:

    employment            current employer / role        (fact)
    location              where the user currently lives  (fact)
    origin                where they moved from / are from (fact)
    pet.name              a pet's name                    (fact, often implicit)
    pet.species           a pet's species                 (fact)
    family.<relation>     spouse/child/parent/sibling …   (fact)
    diet                  vegetarian / vegan / …          (fact)
    allergy               an allergy                      (fact)
    preference.favorite.<thing>   "favourite X is Y"      (preference)
    preference.answer_style       desired answer style    (preference)
    preference            any other stated preference     (preference)
    opinion.<subject>     a sentiment about a subject     (opinion)
    event                 a recent-conversation fallback  (event)

Keys are stable per-topic on purpose: ``opinion.typescript`` stays the same key
as an opinion evolves, which is what lets the (later) supersession layer model
an opinion *arc* rather than a pile of unrelated rows.
"""
from __future__ import annotations

from dataclasses import dataclass

# The four memory types are also enforced by the DB CHECK constraint.
MEMORY_TYPES = ("fact", "preference", "opinion", "event")


@dataclass
class MemoryDraft:
    type: str
    key: str | None
    value: str
    confidence: float
    # Which extraction path produced this: "llm:gemini", "rule", "rule:event", …
    provenance: str = "rule"


def clamp_confidence(value: object, default: float = 0.7) -> float:
    """Coerce arbitrary model output into a confidence in [0, 1]."""
    try:
        c = float(value)
    except (TypeError, ValueError):
        return default
    if c != c:  # NaN
        return default
    return max(0.0, min(1.0, c))
