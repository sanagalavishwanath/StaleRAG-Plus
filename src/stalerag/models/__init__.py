from __future__ import annotations

from typing import Any

from stalerag.models.base import Reader


def make_reader(spec: dict[str, Any], seed: int) -> Reader:
    options = dict(spec)
    kind = options.pop("kind", None)
    if kind == "simulated":
        from stalerag.models.simulated import SimulatedReader

        return SimulatedReader(seed=seed, **options)
    if kind == "hf":
        from stalerag.models.hf import HFReader

        return HFReader(**options)
    raise ValueError(f"unknown reader kind: {kind!r}")
