"""
Experiment Utilities
====================

Shared helpers for reproducible experiments: seed setup, metric tracking,
result formatting, and plot saving.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPERIMENTS_DIR = Path(__file__).parent
RESULTS_DIR = EXPERIMENTS_DIR.parent / "results"


def setup_seed(seed: int = 42) -> np.random.Generator:
    """Create a reproducible random generator."""
    return np.random.default_rng(seed)


def ensure_results_dir(experiment_name: str) -> Path:
    """Create and return the results directory for an experiment."""
    path = RESULTS_DIR / experiment_name
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class ExperimentResult:
    """Structured container for experiment results."""

    name: str
    hypothesis: str
    method: str
    metrics: dict[str, Any] = field(default_factory=dict)
    observations: list[str] = field(default_factory=list)
    conclusion: str = ""
    runtime_seconds: float = 0.0

    def summary(self) -> str:
        lines = [
            f"EXPERIMENT: {self.name}",
            f"HYPOTHESIS: {self.hypothesis}",
            f"METHOD: {self.method}",
            "",
        ]
        if self.metrics:
            lines.append("METRICS:")
            for k, v in self.metrics.items():
                if isinstance(v, float):
                    lines.append(f"  {k}: {v:.6e}")
                else:
                    lines.append(f"  {k}: {v}")
            lines.append("")
        if self.observations:
            lines.append("OBSERVATIONS:")
            for obs in self.observations:
                lines.append(f"  - {obs}")
            lines.append("")
        if self.conclusion:
            lines.append(f"CONCLUSION: {self.conclusion}")
        if self.runtime_seconds > 0:
            lines.append(f"RUNTIME: {self.runtime_seconds:.3f}s")
        return "\n".join(lines)

    def save(self, directory: Path | None = None) -> Path:
        """Save experiment result as JSON."""
        if directory is None:
            directory = ensure_results_dir(self.name.lower().replace(" ", "_"))
        path = directory / "result.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "name": self.name,
                    "hypothesis": self.hypothesis,
                    "method": self.method,
                    "metrics": self.metrics,
                    "observations": self.observations,
                    "conclusion": self.conclusion,
                    "runtime_seconds": self.runtime_seconds,
                },
                f,
                indent=2,
                default=str,
            )
        return path


def save_fig(
    fig: plt.Figure,
    name: str,
    experiment_name: str | None = None,
    directory: Path | None = None,
) -> Path:
    """Save a matplotlib figure."""
    if directory is None:
        if experiment_name:
            directory = ensure_results_dir(experiment_name)
        else:
            directory = RESULTS_DIR
    path = directory / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def print_table(headers: list[str], rows: list[list[str]], title: str = "") -> str:
    """Format a simple text table."""
    if title:
        lines = [f"\n  {title}", ""]
    else:
        lines = [""]

    col_widths = [
        max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
        for i, h in enumerate(headers)
    ]

    header_line = " | ".join(f"{h:>{w}}" for h, w in zip(headers, col_widths))
    separator = "-+-".join("-" * w for w in col_widths)

    lines.append(f"  {header_line}")
    lines.append(f"  {separator}")
    for row in rows:
        line = " | ".join(f"{str(c):>{w}}" for c, w in zip(row, col_widths))
        lines.append(f"  {line}")

    return "\n".join(lines)
