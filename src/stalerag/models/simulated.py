from __future__ import annotations

import hashlib
import random
import re
from functools import lru_cache

from stalerag.models.base import ReaderOutput
from stalerag.schema import Example, Passage


@lru_cache(maxsize=8192)
def _claim_pattern(prop: str, entity: str) -> re.Pattern[str]:
    return re.compile(
        r"(?P<v>[A-Z][a-z]+(?: [A-Z][a-z]+)+),? (?:is|serves as|has been|became)"
        r"(?: currently| now)? the " + re.escape(prop) + " of " + re.escape(entity)
    )


def extract_claim(text: str, prop: str, entity: str) -> str | None:
    match = _claim_pattern(prop, entity).search(text)
    return match.group("v") if match else None


class SimulatedReader:
    def __init__(
        self,
        name: str,
        seed: int,
        first_passage_bias: float = 0.7,
        recency_use: float = 0.3,
        memory_stale: float = 0.4,
        memory_current: float = 0.3,
        noise: float = 0.08,
    ) -> None:
        for label, value in (
            ("first_passage_bias", first_passage_bias),
            ("recency_use", recency_use),
            ("memory_stale", memory_stale),
            ("memory_current", memory_current),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must be in [0, 1]")
        if memory_stale + memory_current > 1.0:
            raise ValueError("memory_stale + memory_current must not exceed 1")
        if noise < 0:
            raise ValueError("noise must be non-negative")
        self.name = name
        self.seed = seed
        self.first_passage_bias = first_passage_bias
        self.recency_use = recency_use
        self.memory_stale = memory_stale
        self.memory_current = memory_current
        self.noise = noise

    def _rng(self, example: Example, context: list[Passage]) -> random.Random:
        key = f"{self.seed}|{self.name}|{example.qid}|{'/'.join(p.pid for p in context)}"
        return random.Random(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big"))

    def _confidence(self, rng: random.Random, centre: float) -> float:
        return min(0.99, max(0.01, centre + rng.gauss(0.0, self.noise)))

    def answer(self, example: Example, context: list[Passage]) -> ReaderOutput:
        rng = self._rng(example, context)
        claims = []
        for passage in context:
            value = extract_claim(passage.text, example.prop, example.entity)
            if value:
                claims.append((passage, value))
        if not claims:
            draw = rng.random()
            if draw < self.memory_current:
                answer = example.current_answer
            elif draw < self.memory_current + self.memory_stale:
                answer = example.stale_answer
            else:
                answer = "Unknown"
            return ReaderOutput(answer, self._confidence(rng, 0.45))
        distinct = {value for _, value in claims}
        if len(distinct) == 1:
            return ReaderOutput(claims[0][1], self._confidence(rng, 0.92))
        dated = [(p.date, v) for p, v in claims if p.date]
        if len(dated) == len(claims) and rng.random() < self.recency_use:
            answer = max(dated)[1]
        elif rng.random() < self.first_passage_bias:
            answer = claims[0][1]
        else:
            answer = claims[-1][1]
        return ReaderOutput(answer, self._confidence(rng, 0.60))
