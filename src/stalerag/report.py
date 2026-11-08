from __future__ import annotations

from typing import Any

SIMULATION_NOTICE = (
    "**Simulated run.** Readers in this run are rule-based simulations with hand-set parameters "
    "and the NLI scorer is a lexical heuristic on synthetic data. These numbers verify that the "
    "pipeline works end to end. They are not measurements of Llama-3.1-8B, Qwen2.5-7B or "
    "Mistral-7B and must not be quoted as project results."
)


def _fmt(entry: dict[str, float | None], digits: int = 3) -> str:
    value = entry["value"]
    if value is None:
        return "n/a"
    low, high = entry.get("ci_low"), entry.get("ci_high")
    if low is None or high is None:
        return f"{value:.{digits}f}"
    return f"{value:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def render_markdown(results: dict[str, Any]) -> str:
    meta, detector, readers = results["meta"], results["detector"], results["readers"]
    simulated = any(kind == "simulated" for kind in meta["reader_kinds"].values())
    out = ["# StaleRAG++ experiment summary", ""]
    if simulated or meta["data_source"] == "synthetic":
        out += [SIMULATION_NOTICE, ""]
    out += [
        f"- data source: {meta['data_source']}; examples: {meta['n_examples']} "
        f"(train {meta['n_train']}, val {meta['n_val']}, test {meta['n_test']}, split by entity)",
        f"- seeds: {meta['seeds']}; bootstrap resamples: {meta['n_boot']}; "
        f"coverage target: {meta['coverage_target']}",
        f"- detector backend: {detector['backend']}; NLI scorer: {meta['nli']}",
        f"- wall time (s): { {k: round(v, 1) for k, v in meta['timings_seconds'].items()} }",
        "",
        "## Evidence reliability detector (positive class = stale)",
        "",
    ]
    main = detector["main"]
    out += [
        _table(
            ["metric", "value [95% bootstrap CI]"],
            [[name.upper(), _fmt(main[name])] for name in ("auroc", "f1", "auprc")],
        ),
        "",
        f"AUROC across seeds: {main['auroc_per_seed']['mean']:.3f} "
        f"+/- {main['auroc_per_seed']['std']:.4f}",
        "",
        "### Feature ablation",
        "",
        _table(
            ["variant", "AUROC", "F1", "AUPRC"],
            [
                [name, _fmt(m["auroc"]), _fmt(m["f1"]), _fmt(m["auprc"])]
                for name, m in detector["ablation"].items()
            ],
        ),
        "",
        "### Generalization",
        "",
    ]
    gen = detector["generalization"]
    rows = [
        [
            "unseen entity (main split)",
            _fmt(gen["unseen_entity"]["auroc"]),
            _fmt(gen["unseen_entity"]["f1"]),
        ],
        ["random split", _fmt(gen["random_split"]["auroc"]), _fmt(gen["random_split"]["f1"])],
    ]
    rows += [
        [f"held-out property: {prop}", _fmt(m["auroc"]), _fmt(m["f1"])]
        for prop, m in gen["unseen_property"].items()
    ]
    out += [_table(["setting", "AUROC", "F1"], rows), ""]
    for name, block in readers.items():
        reader_kind = meta["reader_kinds"][name]
        out += [f"## Reader: {name} ({reader_kind})", "", "### Retrieval conditions", ""]
        out += [
            _table(
                ["condition", "stale rate", "current accuracy", "other"],
                [
                    [key, _fmt(e["stale_rate"]), _fmt(e["current_acc"]), _fmt(e["other_rate"])]
                    for key, e in block["conditions"].items()
                ],
            ),
            "",
            "### Passage-order sensitivity (answer flip rate, stale-first vs current-first)",
            "",
        ]
        rows = []
        for method, e in block["order_sensitivity"].items():
            reduction = e.get("relative_reduction_vs_vanilla")
            rows.append(
                [
                    method,
                    _fmt(e["flip_rate"]),
                    f"{e['accuracy_gap']:.3f}",
                    _fmt(reduction) if reduction else "-",
                ]
            )
        out += [
            _table(["method", "flip rate", "accuracy gap", "reduction vs vanilla"], rows),
            "",
            "### Selective prediction",
            "",
        ]
        rows = []
        for method, e in block["selective"].items():
            rows.append(
                [
                    method,
                    _fmt(e["coverage"]),
                    _fmt(e["accuracy_answered"]),
                    _fmt(e["stale_rate_all"]),
                    f"{e['no_abstain_accuracy']:.3f}",
                    f"{e['aurc']:.3f}",
                ]
            )
        out += [
            _table(
                [
                    "method",
                    "coverage",
                    "accuracy on answered",
                    "stale rate (all questions)",
                    "accuracy w/o abstention",
                    "AURC",
                ],
                rows,
            ),
            "",
            "### Relative stale-rate reduction vs vanilla RAG (no abstention)",
            "",
            _table(
                ["comparison", "reduction [95% CI]"],
                [[k, _fmt(v)] for k, v in block["stale_rate_reduction"].items()],
            ),
            "",
        ]
    return "\n".join(out)
