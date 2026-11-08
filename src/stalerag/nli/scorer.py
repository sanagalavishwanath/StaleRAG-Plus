from __future__ import annotations

import re
from typing import Protocol

import numpy as np

CONTRADICTION, ENTAILMENT, NEUTRAL = 0, 1, 2

_CAPS = re.compile(r"\b[A-Z][a-z]+\b")
_WORD = re.compile(r"[a-z]+")
_STOP = {
    "the", "and", "for", "has", "was", "were", "with", "that", "this", "from", "been", "being",
    "after", "since", "until", "its", "his", "her", "their", "who",
}


class NLIScorer(Protocol):
    def probs(self, pairs: list[tuple[str, str]]) -> np.ndarray: ...


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 3 and w not in _STOP}


class LexicalNLI:
    def probs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        scores = [self._score(premise, hypothesis) for premise, hypothesis in pairs]
        return np.array(scores).reshape(-1, 3)

    @staticmethod
    def _score(premise: str, hypothesis: str) -> list[float]:
        premise_caps, hypothesis_caps = set(_CAPS.findall(premise)), set(_CAPS.findall(hypothesis))
        premise_words, hypothesis_words = _content_words(premise), _content_words(hypothesis)
        overlap = len(premise_words & hypothesis_words) / max(len(hypothesis_words), 1)
        if overlap < 0.3:
            return [0.10, 0.10, 0.80]
        new_in_hypothesis = hypothesis_caps - premise_caps
        new_in_premise = premise_caps - hypothesis_caps
        if new_in_hypothesis and new_in_premise:
            return [0.80, 0.05, 0.15]
        if not new_in_hypothesis:
            return [0.15, 0.70, 0.15]
        return [0.20, 0.20, 0.60]


def label_index(id2label: dict[int, str]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for index, label in id2label.items():
        lowered = label.lower()
        if lowered.startswith("contradict"):
            mapping[CONTRADICTION] = int(index)
        elif lowered.startswith("entail"):
            mapping[ENTAILMENT] = int(index)
        elif lowered.startswith("neutral"):
            mapping[NEUTRAL] = int(index)
    if len(mapping) != 3:
        raise ValueError(f"cannot map NLI labels from {id2label!r}")
    return mapping


class HFNLI:
    def __init__(self, model_id: str, batch_size: int = 16) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.eval()
        self.mapping = label_index(dict(self.model.config.id2label))

    def probs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        torch = self._torch
        rows = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            encoded = self.tokenizer(
                [p for p, _ in batch],
                [h for _, h in batch],
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = self.model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
            for row in probabilities:
                rows.append([row[self.mapping[k]] for k in (CONTRADICTION, ENTAILMENT, NEUTRAL)])
        return np.array(rows).reshape(-1, 3)
