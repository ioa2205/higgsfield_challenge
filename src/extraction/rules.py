"""Deterministic, dependency-free rule extraction.

This is the safety net the pipeline uses when no LLM is configured, when the
LLM call errors/hangs, or when the LLM returns nothing usable. It runs entirely
offline (regex only) and is also what the test suite and the offline default
deployment rely on, so it is built to be *genuinely useful*, not a stub.

It recognises the §4 categories — personal facts (employment, location, family,
pets), preferences/opinions, corrections, and IMPLICIT facts ("walking Biscuit
this morning" → ``pet.name``) — and normalises each into a typed ``MemoryDraft``
with a canonical slot key (see ``draft.py``). Values are written as short, human
-readable statements that *preserve the salient proper noun* (Berlin, Notion,
Biscuit) so they survive both keyword and vector recall.
"""
from __future__ import annotations

import re

from .draft import MemoryDraft

# Trailing temporal / filler words to trim off a captured entity so
# "Notion now" → "Notion" and "Berlin last month" → "Berlin".
_TRAILERS = re.compile(
    r"\s+(?:now|today|currently|recently|these days|this (?:year|month|week)|"
    r"last (?:year|month|week)|since .*|as of .*)\s*$",
    re.IGNORECASE,
)
# A correction cue anywhere in the message raises confidence and tags the draft;
# the *new* value is still captured by the normal slot patterns below.
_CORRECTION = re.compile(
    r"\b(?:actually|sorry|correction|i meant|not\b.*?—|no,\s)", re.IGNORECASE
)


def _clean(entity: str) -> str:
    """Trim punctuation, articles and trailing temporal filler from a capture."""
    entity = entity.strip().strip(".,;:!?\"'()")
    entity = re.sub(r"^(?:the|a|an)\s+", "", entity, flags=re.IGNORECASE)
    prev = None
    while prev != entity:  # strip possibly-stacked trailers ("now", "this year")
        prev = entity
        entity = _TRAILERS.sub("", entity).strip().strip(".,;:!?")
    return entity


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _user_text(messages: list[dict]) -> str:
    """Concatenate the user's utterances in the turn (extraction targets what the
    *user* said, not the assistant's reply)."""
    parts = [
        (m.get("content") or "").strip()
        for m in messages
        if m.get("role") == "user" and (m.get("content") or "").strip()
    ]
    return "  ".join(parts)


# Each rule: (compiled pattern, builder(match, correction) -> MemoryDraft|None).
# Order matters only for readability; all rules run and all hits are emitted,
# then de-duplicated by (type, key, value).
def _employment(m, corr):
    company_raw = m.group("company")
    role = m.groupdict().get("role")
    # "Stripe as a backend engineer" → company=Stripe, role=backend engineer.
    if role is None and re.search(r"\sas\s", company_raw, re.IGNORECASE):
        company_raw, role = re.split(r"\sas\s", company_raw, 1, flags=re.IGNORECASE)
    company = _clean(company_raw)
    if not company:
        return None
    value = f"Works at {company}"
    if role and _clean(role):
        value += f" as {_clean(role)}"
    return MemoryDraft("fact", "employment", value, 0.9 if corr else 0.8)


def _employment_retired(m, corr):
    company = _clean(m.group("company"))
    if not company:
        return None
    return MemoryDraft("fact", "employment", f"No longer works at {company}", 0.9)


def _location(m, corr):
    city = _clean(m.group("city"))
    if not city:
        return None
    origin = m.groupdict().get("origin")
    value = f"Lives in {city}"
    if origin:
        value += f" (moved from {_clean(origin)})"
    return MemoryDraft("fact", "location", value, 0.9 if corr else 0.85)


def _origin(m, corr):
    place = _clean(m.group("place"))
    if not place:
        return None
    return MemoryDraft("fact", "origin", f"Originally from {place}", 0.75)


def _pet_explicit(m, corr):
    species = m.group("species").lower()
    name = _clean(m.group("name"))
    if not name or name.lower() in _NOT_NAMES:
        return None
    return MemoryDraft("fact", "pet.name", f"Has a {species} named {name}", 0.85)


def _pet_implicit(m, corr):
    # "walking Biscuit this morning" / "fed Biscuit" → infer a named pet.
    name = _clean(m.group("name"))
    if not name or name.lower() in _NOT_NAMES:
        return None
    return MemoryDraft("fact", "pet.name", f"Has a pet named {name}", 0.6)


def _family(m, corr):
    rel = m.group("rel").lower()
    rel = {"mom": "mother", "dad": "father", "kid": "child"}.get(rel, rel)
    name = m.groupdict().get("name")
    value = f"Has a {rel}"
    if name:
        value += f" named {_clean(name)}"
    return MemoryDraft("fact", f"family.{rel}", value, 0.75)


def _diet(m, corr):
    diet = _clean(m.group("diet")).capitalize()
    return MemoryDraft("fact", "diet", diet, 0.85)


def _allergy(m, corr):
    what = _clean(m.group("what"))
    if not what:
        return None
    return MemoryDraft("fact", "allergy", f"Allergic to {what}", 0.85)


def _fav(m, corr):
    thing = _clean(m.group("thing"))
    val = _clean(m.group("val"))
    if not thing or not val:
        return None
    return MemoryDraft(
        "preference", f"preference.favorite.{_slug(thing)}",
        f"Favourite {thing} is {val}", 0.8,
    )


def _answer_style(m, corr):
    style = _clean(m.group("style"))
    if not style:
        return None
    return MemoryDraft(
        "preference", "preference.answer_style",
        f"Prefers {style} answers", 0.8,
    )


def _preference(m, corr):
    what = _clean(m.group("what"))
    if not what:
        return None
    return MemoryDraft("preference", "preference", f"Prefers {what}", 0.7)


_SENTIMENT_VERB = {
    "love": "Loves", "loves": "Loves", "adore": "Loves", "like": "Likes",
    "enjoy": "Enjoys", "hate": "Dislikes", "dislike": "Dislikes",
    "can't stand": "Dislikes", "cant stand": "Dislikes",
}


def _opinion_verb(m, corr):
    verb = m.group("verb").lower()
    subj = _clean(m.group("subj"))
    if not subj:
        return None
    lead = _SENTIMENT_VERB.get(verb, "Feels about")
    return MemoryDraft(
        "opinion", f"opinion.{_slug(subj.split()[0])}",
        f"{lead} {subj}", 0.7,
    )


def _opinion_adj(m, corr):
    subj = _clean(m.group("subj"))
    adj = m.group("adj").lower()
    if not subj:
        return None
    return MemoryDraft(
        "opinion", f"opinion.{_slug(subj.split()[0])}",
        f"Thinks {subj} is {adj}", 0.7,
    )


# Words that look like a capitalised name after a verb but are not pet names.
_NOT_NAMES = {
    "i", "the", "my", "this", "that", "today", "tomorrow", "monday", "tuesday",
    "wednesday", "thursday", "friday", "saturday", "sunday",
}

_RULES: list[tuple[re.Pattern, object]] = [
    # employment ----------------------------------------------------------
    (re.compile(
        r"\bI\s+(?:no\s+longer|don't|do\s+not|stopped)\s+(?:work|working)\s+(?:at|for)\s+(?P<company>[^.,;!?\n]+)",
        re.IGNORECASE), _employment_retired),
    # The leading "I (am)" and an intervening adverb ("actually") are optional so
    # "...and work at Acme" / "I actually work at Notion" both match on "work at X".
    (re.compile(
        r"\b(?:I(?:'m| am)?\s+)?(?:\w+ly\s+)?(?:an?\s+(?P<role>[\w ]+?)\s+)?(?<!no longer )(?<!don't )(?<!not )(?<!stopped )(?:works?|working|employed)\s+(?:at|for)\s+(?P<company>[^.,;!?\n]+)",
        re.IGNORECASE), _employment),
    (re.compile(
        r"\bI\s+(?:just\s+)?(?:joined|started\s+(?:at|working\s+at)?|got\s+a\s+job\s+at|moved\s+to\s+a\s+(?:job|role)\s+at)\s+(?P<company>[^.,;!?\n]+)",
        re.IGNORECASE), _employment),
    # location ------------------------------------------------------------
    (re.compile(
        r"\bI\s+(?:just\s+|recently\s+)?(?:moved|relocated)\s+to\s+(?P<city>[A-Za-z][A-Za-z .'-]*?)"
        r"(?:\s+from\s+(?P<origin>[A-Za-z][A-Za-z .'-]*?))?"
        r"(?=[.,;!?\n]|\s+(?:from|last|this|a\s+few|recently|for|because|where|so|and)\b|$)",
        re.IGNORECASE), _location),
    (re.compile(
        r"\bI(?:'m| am)?\s+(?:still\s+|currently\s+)?(?:live|living|based|relocating|relocated)\s+in\s+(?P<city>[^.,;!?\n]+?)"
        r"(?=\s+and\s+(?:I\s+)?(?:work|working|am\s+employed)\b|[.,;!?\n]|$)",
        re.IGNORECASE), _location),
    (re.compile(
        r"\bnot\s+[A-Za-z][A-Za-z .'-]*?,?\s+I\s+meant\s+(?P<city>[A-Za-z][A-Za-z .'-]*?)(?=[.,;!?\n]|$)",
        re.IGNORECASE), _location),
    (re.compile(
        r"\bI(?:'m| am)\s+from\s+(?P<place>[^.,;!?\n]+)", re.IGNORECASE), _origin),
    # pets ----------------------------------------------------------------
    # Set-valued slot: this broad form captures every "dog named X, cat named
    # Y" mention. Identical pets are deduplicated at write time; different pet
    # names intentionally remain active together.
    (re.compile(
        r"\b(?P<species>dog|cat|puppy|kitten|hamster|rabbit|parrot|bird|pet)\s+(?:is\s+)?(?:named|called)\s+(?P<name>[A-Z][\w-]*)",
        re.IGNORECASE), _pet_explicit),
    (re.compile(
        r"\bmy\s+(?P<species>dog|cat|puppy|kitten|hamster|rabbit|parrot|bird|pet)\s+(?:is\s+)?(?:named|called)\s+(?P<name>[A-Z][\w-]*)",
        re.IGNORECASE), _pet_explicit),
    (re.compile(
        r"\bI\s+have\s+(?:a|an|two|three|\d+)\s+(?P<species>dog|cat|puppy|kitten|hamster|rabbit|parrot|bird|pet)s?\s+(?:named|called)\s+(?P<name>[A-Z][\w-]*)",
        re.IGNORECASE), _pet_explicit),
    # "my dog Biscuit" — name directly after species, capitalised (no "named").
    # Keyword/species are case-insensitive (sentence-initial "My"); the NAME group
    # stays case-sensitive so we only capture a capitalised proper noun.
    (re.compile(
        r"\b(?i:my)\s+(?P<species>(?i:dog|cat|puppy|kitten|hamster|rabbit|parrot|bird))\s+(?P<name>[A-Z][a-z]+)\b",
        ), _pet_explicit),
    (re.compile(
        r"\b(?i:walking|walked|feeding|fed|grooming|groomed|took)\s+(?P<name>[A-Z][a-z]+)\b",
        ), _pet_implicit),
    # family --------------------------------------------------------------
    (re.compile(
        r"\bmy\s+(?P<rel>wife|husband|spouse|partner|son|daughter|kid|child|mother|father|mom|dad|brother|sister)\b(?:[^.,;!?\n]*?(?:named|called)\s+(?P<name>[A-Z][\w-]*))?",
        re.IGNORECASE), _family),
    # diet / allergy ------------------------------------------------------
    (re.compile(
        r"\bI(?:'m| am)\s+(?:a\s+)?(?P<diet>vegetarian|vegan|pescatarian|carnivore|gluten-free|lactose intolerant)\b",
        re.IGNORECASE), _diet),
    (re.compile(
        r"\b(?:I(?:'m| am)\s+)?allergic\s+to\s+(?P<what>[^.,;!?\n]+)", re.IGNORECASE), _allergy),
    # preferences ---------------------------------------------------------
    (re.compile(
        r"\bmy\s+favou?rite\s+(?P<thing>[\w ]+?)\s+is\s+(?P<val>[^.,;!?\n]+)",
        re.IGNORECASE), _fav),
    (re.compile(
        r"\bI\s+(?:prefer|like|want)\s+(?P<style>[\w ,]+?)\s+(?:answers|responses|replies)\b",
        re.IGNORECASE), _answer_style),
    (re.compile(
        r"\bI\s+prefer\s+(?P<what>[^.,;!?\n]+)", re.IGNORECASE), _preference),
    # opinions ------------------------------------------------------------
    (re.compile(
        r"\bI\s+(?P<verb>love|loves|adore|like|enjoy|hate|dislike|can'?t stand)\s+(?P<subj>[^.,;!?\n]+)",
        re.IGNORECASE), _opinion_verb),
    (re.compile(
        r"\b(?P<subj>[A-Z][\w ]*?)\s+(?:is|are)\s+(?:getting\s+|kind of\s+|really\s+|pretty\s+)?(?P<adj>annoying|great|terrible|awesome|fine|overrated|underrated|amazing|frustrating|fantastic|painful|solid)\b",
        ), _opinion_adj),
]


def rule_extract(messages: list[dict]) -> list[MemoryDraft]:
    """Extract typed memories from a turn using regex rules. May return []."""
    text = _user_text(messages)
    if not text:
        return []

    corr = bool(_CORRECTION.search(text))
    drafts: list[MemoryDraft] = []
    for pattern, build in _RULES:
        for match in pattern.finditer(text):
            try:
                draft = build(match, corr)
            except (IndexError, AttributeError):
                draft = None
            if draft and draft.value.strip():
                draft.provenance = "rule:correction" if corr else "rule"
                drafts.append(draft)

    # De-duplicate identical (type, key, value); keep the highest confidence.
    best: dict[tuple, MemoryDraft] = {}
    for d in drafts:
        k = (d.type, d.key, d.value.lower())
        if k not in best or d.confidence > best[k].confidence:
            best[k] = d
    return list(best.values())
