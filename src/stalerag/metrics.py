from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("cannot bootstrap an empty array")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(n_boot, array.size))
    means = array[indices].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.size == 0 or y_true.size != y_score.size:
        raise ValueError("y_true and y_score must be non-empty and equally sized")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, y_true.size, y_true.size)
        sample = y_true[idx]
        if sample.min() == sample.max():
            continue
        values.append(metric(sample, y_score[idx]))
    if not values:
        raise ValueError("every bootstrap resample contained a single class")
    return float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2))


def bootstrap_relative_reduction(
    baseline: np.ndarray,
    method: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    baseline = np.asarray(baseline, dtype=float)
    method = np.asarray(method, dtype=float)
    if baseline.size == 0 or baseline.size != method.size:
        raise ValueError("baseline and method must be non-empty and equally sized")

    def reduction(base_mean: float, method_mean: float) -> float:
        return float("nan") if base_mean <= 0 else 1.0 - method_mean / base_mean

    point = reduction(baseline.mean(), method.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, baseline.size, size=(n_boot, baseline.size))
    base_means = baseline[idx].mean(axis=1)
    method_means = method[idx].mean(axis=1)
    values = [reduction(b, m) for b, m in zip(base_means, method_means, strict=True)]
    values = [v for v in values if not np.isnan(v)]
    if not values:
        return point, float("nan"), float("nan")
    return point, float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2))


def order_flip_rate(first: list[str], second: list[str]) -> float:
    if len(first) != len(second):
        raise ValueError("answer lists must have equal length")
    if not first:
        raise ValueError("answer lists must be non-empty")
    return float(np.mean([a != b for a, b in zip(first, second, strict=True)]))


def fast_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positives = np.asarray(y_true).astype(bool)
    n_pos = int(positives.sum())
    n_neg = positives.size - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("both classes are required")
    ranks = rankdata(y_score)
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fast_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positives = np.asarray(y_true).astype(bool)
    scores = np.asarray(y_score, dtype=float)
    n_pos = int(positives.sum())
    if n_pos == 0:
        raise ValueError("at least one positive is required")
    order = np.argsort(-scores, kind="stable")
    sorted_scores, sorted_pos = scores[order], positives[order]
    true_pos = np.cumsum(sorted_pos)
    false_pos = np.cumsum(~sorted_pos)
    edges = np.r_[np.flatnonzero(np.diff(sorted_scores)), positives.size - 1]
    precision = true_pos[edges] / (true_pos[edges] + false_pos[edges])
    recall = true_pos[edges] / n_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def fast_f1(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> float:
    positives = np.asarray(y_true).astype(bool)
    predicted = np.asarray(y_score) >= threshold
    true_pos = int((positives & predicted).sum())
    false_pos = int((~positives & predicted).sum())
    false_neg = int((positives & ~predicted).sum())
    denominator = 2 * true_pos + false_pos + false_neg
    return 2 * true_pos / denominator if denominator else 0.0


def detector_metrics(
    y_stale: np.ndarray,
    stale_score: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    threshold: float = 0.5,
) -> dict[str, dict[str, float]]:
    y_stale = np.asarray(y_stale)
    stale_score = np.asarray(stale_score)

    def sklearn_f1(y: np.ndarray, s: np.ndarray) -> float:
        return float(f1_score(y, s >= threshold, zero_division=0))

    def bootstrap_f1(y: np.ndarray, s: np.ndarray) -> float:
        return fast_f1(y, s, threshold)

    table = {}
    for name, point_fn, boot_fn in (
        ("auroc", roc_auc_score, fast_auroc),
        ("auprc", average_precision_score, fast_auprc),
        ("f1", sklearn_f1, bootstrap_f1),
    ):
        low, high = bootstrap_metric_ci(y_stale, stale_score, boot_fn, n_boot, seed)
        value = float(point_fn(y_stale, stale_score))
        table[name] = {"value": value, "ci_low": low, "ci_high": high}
    return table
