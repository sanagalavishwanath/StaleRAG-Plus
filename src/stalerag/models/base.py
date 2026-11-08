from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from stalerag.schema import Example, Passage

_ARTICLES = {"a", "an", "the"}


@dataclass(frozen=True)
class ReaderOutput:
    answer: str
    confidence: float


class Reader(Protocol):
    name: str

    def answer(self, example: Example, context: list[Passage]) -> ReaderOutput: ...


def normalize_answer(text: str | None) -> str:
    if not text:
        return ""
    tokens = re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()
    return " ".join(t for t in tokens if t not in _ARTICLES)


def contains_phrase(text: str, phrase: str) -> bool:
    normalized_text, normalized_phrase = normalize_answer(text), normalize_answer(phrase)
    if not normalized_phrase:
        return False
    return f" {normalized_phrase} " in f" {normalized_text} "


def classify_answer(prediction: str | None, example: Example) -> str:
    has_current = contains_phrase(prediction or "", example.current_answer)
    has_stale = contains_phrase(prediction or "", example.stale_answer)
    if has_current and has_stale:
        return "both"
    if has_current:
        return "current"
    if has_stale:
        return "stale"
    return "other"
