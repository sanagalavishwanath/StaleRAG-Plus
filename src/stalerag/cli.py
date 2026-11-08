from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import yaml

from stalerag.data.io import write_jsonl
from stalerag.data.synthetic import generate


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        return _clean(value.item())
    return value


def cmd_generate(args: argparse.Namespace) -> None:
    examples = generate(args.n, args.seed)
    write_jsonl(args.out, examples)
    print(f"wrote {len(examples)} synthetic examples to {args.out}")


def cmd_build_benchmark(args: argparse.Namespace) -> None:
    from stalerag.data.builder import build_benchmark
    from stalerag.data.revisions import WikipediaClient
    from stalerag.data.wikidata import WikidataClient

    examples = build_benchmark(
        WikidataClient(), WikipediaClient(), target_size=args.target_size, seed=args.seed
    )
    write_jsonl(args.out, examples)
    print(f"wrote {len(examples)} examples to {args.out}")


def cmd_run(args: argparse.Namespace) -> None:
    from stalerag.experiments.run import run_experiments
    from stalerag.plots import plot_ablation, plot_risk_coverage
    from stalerag.report import render_markdown

    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    results = _clean(run_experiments(cfg))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(render_markdown(results), encoding="utf-8")
    plot_risk_coverage(results, out_dir / "risk_coverage.png")
    plot_ablation(results, out_dir / "detector_ablation.png")
    print(render_markdown(results))
    print(f"\nartifacts written to {out_dir}/")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stalerag")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate-synthetic")
    gen.add_argument("--n", type=int, default=2000)
    gen.add_argument("--seed", type=int, default=7)
    gen.add_argument("--out", default="data/synthetic.jsonl")
    gen.set_defaults(func=cmd_generate)
    bench = sub.add_parser("build-benchmark")
    bench.add_argument("--out", default="data/stalerag_benchmark.jsonl")
    bench.add_argument("--target-size", type=int, default=2000)
    bench.add_argument("--seed", type=int, default=0)
    bench.set_defaults(func=cmd_build_benchmark)
    run = sub.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--out-dir", default="results")
    run.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
