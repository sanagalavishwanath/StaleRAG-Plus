from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from stalerag.data.io import group_split, holdout_property, random_split, read_jsonl
from stalerag.data.synthetic import generate
from stalerag.detector.features import FEATURE_GROUPS, featurize, group_columns
from stalerag.detector.model import ReliabilityDetector
from stalerag.metrics import (
    bootstrap_ci,
    bootstrap_relative_reduction,
    detector_metrics,
)
from stalerag.models import make_reader
from stalerag.models.base import classify_answer, normalize_answer
from stalerag.nli import make_nli
from stalerag.pipeline import reliability_map, reorder
from stalerag.retrieval.conditions import (
    CONDITIONS,
    MIXED_CURRENT_FIRST,
    MIXED_STALE_FIRST,
    build_context,
    vanilla_context,
)
from stalerag.selective import (
    area_under_risk_coverage,
    bootstrap_selective,
    combined_confidence,
    risk_coverage_curve,
    selective_summary,
    threshold_for_coverage,
)
from stalerag.schema import Example

log = logging.getLogger(__name__)

METHOD_CONTEXT = {"vanilla": "vanilla", "rerank": "rerank_sf", "filter": "filter_sf"}
ORDER_PAIRS = {
    "vanilla": (MIXED_STALE_FIRST, MIXED_CURRENT_FIRST),
    "rerank": ("rerank_sf", "rerank_cf"),
    "filter": ("filter_sf", "filter_cf"),
}
REPORTED_KEYS = (*CONDITIONS, "vanilla", "rerank_sf", "filter_sf")


def load_examples(data_cfg: dict[str, Any]) -> list[Example]:
    source = data_cfg["source"]
    if source == "synthetic":
        return generate(
            data_cfg["n"],
            data_cfg["synthetic_seed"],
            data_cfg.get("p_date_missing", 0.35),
            data_cfg.get("p_date_inverted", 0.10),
        )
    if source == "jsonl":
        return read_jsonl(data_cfg["path"])
    raise ValueError(f"unknown data source: {source!r}")


def detector_rows(
    examples: list[Example], feats: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    rows, labels = [], []
    for example in examples:
        matrix = feats[example.qid]
        for index, passage in enumerate(example.passages):
            if passage.role == "distractor":
                continue
            rows.append(matrix[index])
            labels.append(1 if passage.role == "stale" else 0)
    return np.array(rows), np.array(labels)


def fit_ensemble(
    seeds: list[int],
    backend: str,
    columns: list[int] | None,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, list[ReliabilityDetector]]:
    detectors = [
        ReliabilityDetector(seed=s, backend=backend, columns=columns).fit(x_train, y_train)
        for s in seeds
    ]
    return np.mean([d.stale_probability(x_test) for d in detectors], axis=0), detectors


def evaluate_detector(
    train: list[Example],
    test: list[Example],
    everything: list[Example],
    feats: dict[str, np.ndarray],
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], list[ReliabilityDetector]]:
    seeds, backend, n_boot = cfg["seeds"], cfg["detector"]["backend"], cfg["n_boot"]
    boot_seed = cfg["data_seed"]
    x_train, y_train = detector_rows(train, feats)
    x_test, y_test = detector_rows(test, feats)
    scores, detectors = fit_ensemble(seeds, backend, None, x_train, y_train, x_test)
    main = detector_metrics(y_test, scores, n_boot, boot_seed)
    per_seed = [roc_auc_score(y_test, d.stale_probability(x_test)) for d in detectors]
    main["auroc_per_seed"] = {"mean": float(np.mean(per_seed)), "std": float(np.std(per_seed))}

    variants: dict[str, list[int]] = {}
    for group in FEATURE_GROUPS:
        variants[f"without_{group}"] = group_columns([g for g in FEATURE_GROUPS if g != group])
        variants[f"only_{group}"] = group_columns([group])
    ablation = {}
    for name, columns in variants.items():
        variant_scores, _ = fit_ensemble(seeds, backend, columns, x_train, y_train, x_test)
        ablation[name] = detector_metrics(y_test, variant_scores, n_boot, boot_seed)
    ablation["full"] = main

    rand_train, rand_test = random_split(everything, 0.3, cfg["data_seed"])
    xr_train, yr_train = detector_rows(rand_train, feats)
    xr_test, yr_test = detector_rows(rand_test, feats)
    rand_scores, _ = fit_ensemble(seeds, backend, None, xr_train, yr_train, xr_test)
    generalization: dict[str, Any] = {
        "unseen_entity": main,
        "random_split": detector_metrics(yr_test, rand_scores, n_boot, boot_seed),
        "unseen_property": {},
    }
    for prop in sorted({e.prop for e in everything}):
        rest, held = holdout_property(everything, prop)
        xp_train, yp_train = detector_rows(rest, feats)
        xp_test, yp_test = detector_rows(held, feats)
        prop_scores, _ = fit_ensemble(seeds, backend, None, xp_train, yp_train, xp_test)
        generalization["unseen_property"][prop] = detector_metrics(
            yp_test, prop_scores, n_boot, boot_seed
        )
    held_auroc = [m["auroc"]["value"] for m in generalization["unseen_property"].values()]
    generalization["unseen_property_macro_auroc"] = float(np.mean(held_auroc))
    block = {
        "backend": detectors[0].backend_name,
        "n_train_passages": int(len(y_train)),
        "n_test_passages": int(len(y_test)),
        "main": main,
        "ablation": ablation,
        "generalization": generalization,
    }
    return block, detectors


def contexts_for(example: Example, rel: dict[str, float], drop_below: float) -> dict[str, list]:
    stale_first = build_context(example, MIXED_STALE_FIRST)
    current_first = build_context(example, MIXED_CURRENT_FIRST)
    contexts = {condition: build_context(example, condition) for condition in CONDITIONS}
    contexts["vanilla"] = vanilla_context(example)
    contexts["rerank_sf"] = reorder(stale_first, rel)
    contexts["rerank_cf"] = reorder(current_first, rel)
    contexts["filter_sf"] = reorder(stale_first, rel, drop_below)
    contexts["filter_cf"] = reorder(current_first, rel, drop_below)
    return contexts


def run_reader_split(
    reader: Any,
    examples: list[Example],
    rel_maps: dict[str, dict[str, float]],
    drop_below: float,
) -> list[dict[str, Any]]:
    rows = []
    for example in examples:
        rel = rel_maps[example.qid]
        contexts = contexts_for(example, rel, drop_below)
        cache: dict[tuple[str, ...], Any] = {}
        outputs = {}
        for key, context in contexts.items():
            signature = tuple(p.pid for p in context)
            if signature not in cache:
                cache[signature] = reader.answer(example, context)
            outputs[key] = cache[signature]
        rows.append({"example": example, "contexts": contexts, "outputs": outputs, "rel": rel})
    return rows


def method_confidences(rows: list[dict[str, Any]], method: str) -> np.ndarray:
    key = METHOD_CONTEXT[method]
    values = []
    for row in rows:
        output = row["outputs"][key]
        if method == "vanilla":
            values.append(output.confidence)
        else:
            values.append(
                combined_confidence(
                    output.confidence, output.answer, row["contexts"][key], row["rel"]
                )
            )
    return np.array(values)


def class_array(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.array([classify_answer(r["outputs"][key].answer, r["example"]) for r in rows])


def evaluate_reader(
    spec: dict[str, Any],
    val: list[Example],
    test: list[Example],
    rel_val: dict[int, dict[str, dict[str, float]]],
    rel_test: dict[int, dict[str, dict[str, float]]],
    cfg: dict[str, Any],
    reader_cache: dict[str, Any],
) -> dict[str, Any]:
    seeds, n_boot, coverage = cfg["seeds"], cfg["n_boot"], cfg["coverage"]
    drop_below = cfg.get("filter_threshold", 0.5)
    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        if spec["kind"] == "hf":
            reader = reader_cache.setdefault(spec["name"], make_reader(spec, seed))
        else:
            reader = make_reader(spec, seed)
        val_rows = run_reader_split(reader, val, rel_val[seed], drop_below)
        test_rows = run_reader_split(reader, test, rel_test[seed], drop_below)
        record: dict[str, Any] = {"classes": {}, "flips": {}, "methods": {}}
        keys = {*REPORTED_KEYS, *[k for pair in ORDER_PAIRS.values() for k in pair]}
        for key in keys:
            record["classes"][key] = class_array(test_rows, key)
        for method, (first, second) in ORDER_PAIRS.items():
            answers_a = [normalize_answer(r["outputs"][first].answer) for r in test_rows]
            answers_b = [normalize_answer(r["outputs"][second].answer) for r in test_rows]
            record["flips"][method] = np.array([a != b for a, b in zip(answers_a, answers_b)])
        for method, key in METHOD_CONTEXT.items():
            classes = record["classes"][key]
            correct, stale = classes == "current", classes == "stale"
            conf_val = method_confidences(val_rows, method)
            conf_test = method_confidences(test_rows, method)
            val_classes = class_array(val_rows, key)
            threshold = threshold_for_coverage(conf_val, coverage)
            record["methods"][method] = {
                "correct": correct,
                "stale": stale,
                "conf": conf_test,
                "answered": conf_test >= threshold,
                "threshold": threshold,
                "val_accuracy": float((val_classes == "current").mean()),
            }
        per_seed.append(record)
        log.info("reader %s seed %s done", spec["name"], seed)

    def stack(getter: Any) -> np.ndarray:
        return np.stack([getter(r) for r in per_seed])

    out: dict[str, Any] = {"conditions": {}, "order_sensitivity": {}, "selective": {}}
    for key in REPORTED_KEYS:
        is_current = stack(lambda r, k=key: r["classes"][k] == "current").astype(float)
        is_stale = stack(lambda r, k=key: r["classes"][k] == "stale").astype(float)
        is_other = 1.0 - is_current - is_stale
        entry = {}
        for name, arr in (
            ("current_acc", is_current),
            ("stale_rate", is_stale),
            ("other_rate", is_other),
        ):
            low, high = bootstrap_ci(arr.mean(axis=0), n_boot, cfg["data_seed"])
            entry[name] = {
                "value": float(arr.mean()),
                "ci_low": low,
                "ci_high": high,
                "seed_std": float(arr.mean(axis=1).std()),
            }
        out["conditions"][key] = entry

    flips = {m: stack(lambda r, mm=m: r["flips"][mm]).astype(float) for m in ORDER_PAIRS}
    for method, (first, second) in ORDER_PAIRS.items():
        acc_first = stack(lambda r, k=first: r["classes"][k] == "current").mean(axis=1)
        acc_second = stack(lambda r, k=second: r["classes"][k] == "current").mean(axis=1)
        low, high = bootstrap_ci(flips[method].mean(axis=0), n_boot, cfg["data_seed"])
        entry = {
            "flip_rate": {"value": float(flips[method].mean()), "ci_low": low, "ci_high": high},
            "accuracy_gap": float(np.abs(acc_first - acc_second).mean()),
        }
        if method != "vanilla":
            point, low, high = bootstrap_relative_reduction(
                flips["vanilla"].mean(axis=0), flips[method].mean(axis=0), n_boot, cfg["data_seed"]
            )
            entry["relative_reduction_vs_vanilla"] = {
                "value": point,
                "ci_low": low,
                "ci_high": high,
            }
        out["order_sensitivity"][method] = entry

    stale_all = {}
    for method in METHOD_CONTEXT:
        correct = stack(lambda r, m=method: r["methods"][m]["correct"])
        stale = stack(lambda r, m=method: r["methods"][m]["stale"])
        answered = stack(lambda r, m=method: r["methods"][m]["answered"])
        everyone = np.ones_like(answered)
        stale_all[method] = {
            "no_abstain": (stale & everyone).astype(float).mean(axis=0),
            "abstain": (stale & answered).astype(float).mean(axis=0),
        }
        summaries = [
            selective_summary(
                r["methods"][method]["correct"],
                r["methods"][method]["stale"],
                r["methods"][method]["conf"],
                r["methods"][method]["threshold"],
            )
            for r in per_seed
        ]
        intervals = bootstrap_selective(correct, stale, answered, n_boot, cfg["data_seed"])
        entry = {
            name: {
                "value": float(np.nanmean([s[name] for s in summaries])),
                **(
                    {"ci_low": intervals[name][0], "ci_high": intervals[name][1]}
                    if name in intervals
                    else {}
                ),
            }
            for name in ("coverage", "accuracy_answered", "stale_rate_all")
        }
        entry["no_abstain_accuracy"] = float(correct.mean())
        entry["aurc"] = float(
            np.mean(
                [
                    area_under_risk_coverage(
                        r["methods"][method]["correct"], r["methods"][method]["conf"]
                    )
                    for r in per_seed
                ]
            )
        )
        cov, risk = risk_coverage_curve(
            np.concatenate([r["methods"][method]["correct"] for r in per_seed]),
            np.concatenate([r["methods"][method]["conf"] for r in per_seed]),
        )
        entry["risk_coverage"] = {"coverage": cov.tolist(), "risk": risk.tolist()}
        out["selective"][method] = entry

    reductions = {}
    for method in ("rerank", "filter"):
        for mode in ("no_abstain", "abstain"):
            point, low, high = bootstrap_relative_reduction(
                stale_all["vanilla"]["no_abstain"],
                stale_all[method][mode],
                n_boot,
                cfg["data_seed"],
            )
            reductions[f"{method}_{mode}_vs_vanilla"] = {
                "value": point,
                "ci_low": low,
                "ci_high": high,
            }
    point, low, high = bootstrap_relative_reduction(
        stale_all["vanilla"]["no_abstain"],
        stale_all["vanilla"]["abstain"],
        n_boot,
        cfg["data_seed"],
    )
    reductions["vanilla_abstain_vs_vanilla"] = {"value": point, "ci_low": low, "ci_high": high}
    out["stale_rate_reduction"] = reductions
    return out


def run_experiments(cfg: dict[str, Any]) -> dict[str, Any]:
    timings: dict[str, float] = {}
    started = time.perf_counter()
    examples = load_examples(cfg["data"])
    train, val, test = group_split(examples, tuple(cfg["split"]), cfg["data_seed"])
    timings["data"] = time.perf_counter() - started

    mark = time.perf_counter()
    nli = make_nli(cfg["nli"])
    feats = {e.qid: featurize(e, e.passages, nli) for e in examples}
    timings["features"] = time.perf_counter() - mark

    mark = time.perf_counter()
    detector_block, detectors = evaluate_detector(train, test, examples, feats, cfg)
    timings["detector"] = time.perf_counter() - mark
    by_seed = dict(zip(cfg["seeds"], detectors, strict=True))
    rel_val = {
        s: {e.qid: reliability_map(e.passages, feats[e.qid], d) for e in val}
        for s, d in by_seed.items()
    }
    rel_test = {
        s: {e.qid: reliability_map(e.passages, feats[e.qid], d) for e in test}
        for s, d in by_seed.items()
    }

    mark = time.perf_counter()
    reader_cache: dict[str, Any] = {}
    readers = {
        spec["name"]: evaluate_reader(spec, val, test, rel_val, rel_test, cfg, reader_cache)
        for spec in cfg["readers"]
    }
    timings["readers"] = time.perf_counter() - mark
    timings["total"] = time.perf_counter() - started
    return {
        "meta": {
            "data_source": cfg["data"]["source"],
            "n_examples": len(examples),
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
            "seeds": cfg["seeds"],
            "n_boot": cfg["n_boot"],
            "coverage_target": cfg["coverage"],
            "nli": cfg["nli"],
            "reader_kinds": {s["name"]: s["kind"] for s in cfg["readers"]},
            "timings_seconds": timings,
        },
        "detector": detector_block,
        "readers": readers,
    }
