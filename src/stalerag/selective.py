from __future__ import annotations

import numpy as np

from stalerag.models.base import contains_phrase
from stalerag.schema import Passage


def combined_confidence(
    reader_confidence: float, answer: str, context: list[Passage], reliability: dict[str, float]
) -> float:
    supports = [
        reliability[p.pid]
        for p in context
        if p.pid in reliability and contains_phrase(p.text, answer)
    ]
    return reader_confidence * (min(supports) if supports else 0.5)


def threshold_for_coverage(confidence: np.ndarray, coverage: float) -> float:
    values = np.asarray(confidence, dtype=float)
    if values.size == 0:
        raise ValueError("confidence must be non-empty")
    if not 0.0 < coverage <= 1.0:
        raise ValueError("coverage must be in (0, 1]")
    return float(np.quantile(values, round(1.0 - coverage, 12), method="lower"))


def selective_summary(
    correct: np.ndarray, stale: np.ndarray, confidence: np.ndarray, threshold: float
) -> dict[str, float]:
    answered = np.asarray(confidence) >= threshold
    correct = np.asarray(correct, dtype=bool)
    stale = np.asarray(stale, dtype=bool)
    count = int(answered.sum())
    return {
        "coverage": float(answered.mean()),
        "accuracy_answered": float(correct[answered].mean()) if count else float("nan"),
        "stale_rate_answered": float(stale[answered].mean()) if count else float("nan"),
        "stale_rate_all": float((stale & answered).mean()),
        "correct_rate_all": float((correct & answered).mean()),
    }


def bootstrap_selective(
    correct: np.ndarray,
    stale: np.ndarray,
    answered: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, tuple[float, float]]:
    correct, stale, answered = (np.asarray(a, dtype=bool) for a in (correct, stale, answered))
    if correct.ndim != 2 or correct.shape != stale.shape or correct.shape != answered.shape:
        raise ValueError("correct, stale and answered must share a [seeds, questions] shape")
    questions = correct.shape[1]
    rng = np.random.default_rng(seed)
    coverage, accuracy, stale_all = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, questions, questions)
        ans = answered[:, idx]
        total = ans.sum()
        coverage.append(ans.mean())
        stale_all.append((stale[:, idx] & ans).mean())
        if total:
            accuracy.append((correct[:, idx] & ans).sum() / total)

    def interval(values: list[float]) -> tuple[float, float]:
        if not values:
            return float("nan"), float("nan")
        return float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2))

    return {
        "coverage": interval(coverage),
        "accuracy_answered": interval(accuracy),
        "stale_rate_all": interval(stale_all),
    }


def risk_coverage_curve(
    correct: np.ndarray, confidence: np.ndarray, points: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    correct = np.asarray(correct, dtype=bool)
    confidence = np.asarray(confidence, dtype=float)
    if correct.size == 0 or correct.size != confidence.size:
        raise ValueError("correct and confidence must be non-empty and equally sized")
    order = np.argsort(-confidence, kind="stable")
    errors = (~correct[order]).cumsum()
    counts = np.arange(1, correct.size + 1)
    risk = errors / counts
    picks = np.unique(np.linspace(0, correct.size - 1, num=min(points, correct.size)).astype(int))
    return counts[picks] / correct.size, risk[picks]


def area_under_risk_coverage(correct: np.ndarray, confidence: np.ndarray) -> float:
    correct = np.asarray(correct, dtype=bool)
    confidence = np.asarray(confidence, dtype=float)
    order = np.argsort(-confidence, kind="stable")
    errors = (~correct[order]).cumsum()
    return float(np.mean(errors / np.arange(1, correct.size + 1)))
