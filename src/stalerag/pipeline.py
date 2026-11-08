from __future__ import annotations

import numpy as np

from stalerag.detector.model import ReliabilityDetector
from stalerag.schema import Passage


def reliability_map(
    passages: list[Passage], features: np.ndarray, detector: ReliabilityDetector
) -> dict[str, float]:
    scored = np.flatnonzero(~np.isnan(features).all(axis=1))
    if scored.size == 0:
        return {}
    values = detector.reliability(features[scored])
    return {passages[i].pid: float(v) for i, v in zip(scored, values, strict=True)}


def reorder(
    passages: list[Passage], reliability: dict[str, float], drop_below: float | None = None
) -> list[Passage]:
    kept = list(passages)
    if drop_below is not None:
        kept = [p for p in passages if reliability.get(p.pid, -1.0) >= drop_below]
        if not kept and reliability:
            scored = (p for p in passages if p.pid in reliability)
            best = max(scored, key=lambda p: reliability[p.pid])
            kept = [best]
    return sorted(
        kept, key=lambda p: (p.pid not in reliability, -reliability.get(p.pid, 0.0), p.pid)
    )
