from __future__ import annotations

from stalerag.retrieval.bm25 import rank_passages
from stalerag.schema import ROLE_CURRENT, ROLE_STALE, Example, Passage

CLOSED_BOOK = "closed_book"
STALE_ONLY = "stale_only"
CURRENT_ONLY = "current_only"
MIXED_STALE_FIRST = "mixed_stale_first"
MIXED_CURRENT_FIRST = "mixed_current_first"

CONDITIONS = (CLOSED_BOOK, STALE_ONLY, CURRENT_ONLY, MIXED_STALE_FIRST, MIXED_CURRENT_FIRST)


def build_context(example: Example, condition: str) -> list[Passage]:
    stale = example.passage(ROLE_STALE)
    current = example.passage(ROLE_CURRENT)
    distractors = example.distractors()
    if condition == CLOSED_BOOK:
        return []
    if condition == STALE_ONLY:
        return [stale, *distractors]
    if condition == CURRENT_ONLY:
        return [current, *distractors]
    if condition == MIXED_STALE_FIRST:
        return [stale, current, *distractors]
    if condition == MIXED_CURRENT_FIRST:
        return [current, stale, *distractors]
    raise ValueError(f"unknown retrieval condition: {condition!r}")


def vanilla_context(example: Example) -> list[Passage]:
    return rank_passages(example.question, list(example.passages))
