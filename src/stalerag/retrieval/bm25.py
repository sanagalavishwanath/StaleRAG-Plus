from __future__ import annotations

import math
import re
from collections import Counter

from stalerag.schema import Passage

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def bm25_scores(query: str, docs: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    if not docs:
        return []
    query_terms = set(tokenize(query))
    tokenized = [tokenize(doc) for doc in docs]
    count = len(docs)
    average = sum(len(tokens) for tokens in tokenized) / count or 1.0
    document_frequency: Counter[str] = Counter()
    for tokens in tokenized:
        document_frequency.update(set(tokens))
    scores = []
    for tokens in tokenized:
        frequency = Counter(tokens)
        score = 0.0
        for term in query_terms:
            if term not in frequency:
                continue
            df = document_frequency[term]
            idf = math.log(1 + (count - df + 0.5) / (df + 0.5))
            denominator = frequency[term] + k1 * (1 - b + b * len(tokens) / average)
            score += idf * frequency[term] * (k1 + 1) / denominator
        scores.append(score)
    return scores


def rank_passages(question: str, passages: list[Passage]) -> list[Passage]:
    scores = bm25_scores(question, [p.text for p in passages])
    order = sorted(range(len(passages)), key=lambda i: (-scores[i], passages[i].pid))
    return [passages[i] for i in order]
