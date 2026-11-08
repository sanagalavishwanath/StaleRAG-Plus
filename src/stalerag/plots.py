from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_risk_coverage(results: dict[str, Any], path: str | Path) -> None:
    readers = results["readers"]
    fig, axes = plt.subplots(1, len(readers), figsize=(5 * len(readers), 4), squeeze=False)
    for axis, (name, block) in zip(axes[0], readers.items(), strict=True):
        for method, entry in block["selective"].items():
            curve = entry["risk_coverage"]
            axis.plot(curve["coverage"], curve["risk"], label=method)
        axis.set_title(name)
        axis.set_xlabel("coverage")
        axis.set_ylabel("error rate among answered")
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_ablation(results: dict[str, Any], path: str | Path) -> None:
    ablation = results["detector"]["ablation"]
    names = list(ablation)
    values = [ablation[n]["auroc"]["value"] for n in names]
    errors = [
        [ablation[n]["auroc"]["value"] - ablation[n]["auroc"]["ci_low"] for n in names],
        [ablation[n]["auroc"]["ci_high"] - ablation[n]["auroc"]["value"] for n in names],
    ]
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.bar(names, values, yerr=errors, capsize=3)
    axis.set_ylabel("AUROC (95% bootstrap CI)")
    axis.set_ylim(0.4, 1.02)
    axis.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
