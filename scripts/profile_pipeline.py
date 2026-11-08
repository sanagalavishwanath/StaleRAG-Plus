from __future__ import annotations

import argparse
import cProfile
import io
import pstats

import yaml

from stalerag.experiments.run import run_experiments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    profiler = cProfile.Profile()
    profiler.enable()
    results = run_experiments(cfg)
    profiler.disable()
    buffer = io.StringIO()
    stats = pstats.Stats(profiler, stream=buffer).sort_stats("cumulative")
    stats.print_stats(args.top)
    timings = results["meta"]["timings_seconds"]
    report = "stage timings (s): " + str({k: round(v, 2) for k, v in timings.items()}) + "\n\n"
    report += buffer.getvalue()
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(report)
    print(report)


if __name__ == "__main__":
    main()
