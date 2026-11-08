from __future__ import annotations

from typing import Any

from stalerag.nli.scorer import NLIScorer


def make_nli(spec: dict[str, Any]) -> NLIScorer:
    options = dict(spec)
    kind = options.pop("kind", None)
    if kind == "lexical":
        from stalerag.nli.scorer import LexicalNLI

        return LexicalNLI()
    if kind == "hf":
        from stalerag.nli.scorer import HFNLI

        return HFNLI(**options)
    raise ValueError(f"unknown NLI kind: {kind!r}")
