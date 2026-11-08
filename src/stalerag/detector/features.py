from __future__ import annotations

import datetime as dt
import re

import numpy as np

from stalerag.nli.scorer import NLIScorer
from stalerag.retrieval.bm25 import bm25_scores, tokenize
from stalerag.schema import Example, Passage

PAST_CUES = re.compile(
    r"\b(former|formerly|previously|until|stepped down|resigned|retired|replaced by|"
    r"succeeded by|left)\b|\bex-",
    re.IGNORECASE,
)
PRESENT_CUES = re.compile(
    r"\b(current|currently|incumbent|since|appointed|succeeding|took over|assumed)\b",
    re.IGNORECASE,
)
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
CAPS = re.compile(r"\b[A-Z][a-z]+\b")

FEATURE_GROUPS: dict[str, list[str]] = {
    "temporal": [
        "has_date",
        "date_rank",
        "days_behind_newest",
        "year_gap",
        "has_year",
        "cue_past",
        "cue_present",
    ],
    "nli": [
        "nli_contra_max",
        "nli_contra_mean",
        "nli_entail_max",
        "nli_neutral_mean",
        "nli_contra_newer",
        "nli_contra_older",
    ],
    "lexical": ["q_overlap", "bm25_norm", "claim_diff_mean", "claim_diff_max"],
}
FEATURE_NAMES = [name for group in FEATURE_GROUPS.values() for name in group]
_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def group_columns(groups: list[str]) -> list[int]:
    return [_INDEX[name] for group in groups for name in FEATURE_GROUPS[group]]


def is_relevant(example: Example, passage: Passage) -> bool:
    return example.entity in passage.text


def _parse_date(value: str | None) -> dt.date | None:
    if not value:
        return None
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        return None


def _jaccard_distance(left: set[str], right: set[str]) -> float:
    union = left | right
    return 1.0 - len(left & right) / len(union) if union else 0.0


def _max_or_nan(values: list[float]) -> float:
    return max(values) if values else float("nan")


def _mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _pairwise_nli(texts: list[str], nli: NLIScorer) -> np.ndarray:
    size = len(texts)
    matrix = np.zeros((3, size, size))
    pairs, index = [], []
    for i in range(size):
        for j in range(i + 1, size):
            pairs += [(texts[i], texts[j]), (texts[j], texts[i])]
            index.append((i, j))
    if not pairs:
        return matrix
    probs = nli.probs(pairs)
    for k, (i, j) in enumerate(index):
        symmetric = (probs[2 * k] + probs[2 * k + 1]) / 2
        for channel in range(3):
            matrix[channel, i, j] = matrix[channel, j, i] = symmetric[channel]
    return matrix


def featurize(example: Example, passages: list[Passage], nli: NLIScorer) -> np.ndarray:
    out = np.full((len(passages), len(FEATURE_NAMES)), np.nan)
    relevant_idx = [i for i, p in enumerate(passages) if is_relevant(example, p)]
    if not relevant_idx:
        return out
    relevant = [passages[i] for i in relevant_idx]
    size = len(relevant)
    dates = [_parse_date(p.date) for p in relevant]
    dated = [d for d in dates if d is not None]
    newest = max(dated) if dated else None
    years = [[int(y) for y in YEAR.findall(p.text)] for p in relevant]
    set_max_year = max((max(y) for y in years if y), default=None)
    bm25 = bm25_scores(example.question, [p.text for p in relevant])
    bm25_peak = max(bm25) or 1.0
    question_tokens = set(tokenize(example.question))
    entity_caps = set(CAPS.findall(example.entity))
    cap_sets = [set(CAPS.findall(p.text)) - entity_caps for p in relevant]
    nli_matrix = _pairwise_nli([p.text for p in relevant], nli)
    contra, entail, neutral = nli_matrix

    for i, passage in enumerate(relevant):
        row = {name: float("nan") for name in FEATURE_NAMES}
        others = [j for j in range(size) if j != i]
        date = dates[i]
        row["has_date"] = 1.0 if date else 0.0
        if date and newest:
            row["days_behind_newest"] = (newest - date).days / 365.0
            if len(dated) >= 2:
                less = sum(d < date for d in dated)
                equal = sum(d == date for d in dated) - 1
                row["date_rank"] = (less + equal / 2) / (len(dated) - 1)
        row["has_year"] = 1.0 if years[i] else 0.0
        if years[i] and set_max_year is not None:
            row["year_gap"] = float(max(years[i]) - set_max_year)
        row["cue_past"] = float(len(PAST_CUES.findall(passage.text)))
        row["cue_present"] = float(len(PRESENT_CUES.findall(passage.text)))
        if others:
            row["nli_contra_max"] = float(max(contra[i, j] for j in others))
            row["nli_contra_mean"] = float(np.mean([contra[i, j] for j in others]))
            row["nli_entail_max"] = float(max(entail[i, j] for j in others))
            row["nli_neutral_mean"] = float(np.mean([neutral[i, j] for j in others]))
            newer = [j for j in others if date and dates[j] and dates[j] > date]
            older = [j for j in others if date and dates[j] and dates[j] < date]
            row["nli_contra_newer"] = _max_or_nan([float(contra[i, j]) for j in newer])
            row["nli_contra_older"] = _max_or_nan([float(contra[i, j]) for j in older])
            diffs = [_jaccard_distance(cap_sets[i], cap_sets[j]) for j in others]
            row["claim_diff_mean"] = _mean_or_nan(diffs)
            row["claim_diff_max"] = _max_or_nan(diffs)
        passage_tokens = set(tokenize(passage.text))
        union = question_tokens | passage_tokens
        row["q_overlap"] = len(question_tokens & passage_tokens) / len(union) if union else 0.0
        row["bm25_norm"] = bm25[i] / bm25_peak
        out[relevant_idx[i]] = [row[name] for name in FEATURE_NAMES]
    return out
