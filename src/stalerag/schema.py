from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

ROLE_STALE = "stale"
ROLE_CURRENT = "current"
ROLE_DISTRACTOR = "distractor"
ROLES = (ROLE_STALE, ROLE_CURRENT, ROLE_DISTRACTOR)


@dataclass(frozen=True)
class Passage:
    pid: str
    text: str
    date: str | None
    role: str

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"unknown passage role: {self.role!r}")


@dataclass
class Example:
    qid: str
    question: str
    entity: str
    prop: str
    current_answer: str
    stale_answer: str
    passages: list[Passage]
    source: str = "unknown"

    def __post_init__(self) -> None:
        if self.current_answer.strip().lower() == self.stale_answer.strip().lower():
            raise ValueError(f"{self.qid}: current and stale answers must differ")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Example:
        payload = dict(data)
        payload["passages"] = [Passage(**p) for p in payload["passages"]]
        return cls(**payload)

    def passage(self, role: str) -> Passage:
        for passage in self.passages:
            if passage.role == role:
                return passage
        raise KeyError(f"{self.qid}: no passage with role {role!r}")

    def distractors(self) -> list[Passage]:
        return [p for p in self.passages if p.role == ROLE_DISTRACTOR]
