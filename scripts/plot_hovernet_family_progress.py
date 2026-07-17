#!/usr/bin/env python3
"""Plot metric-formula-aware progress for a HoVer-Net LR pilot family."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cpath_conic.constants import CLASS_NAMES


def load_runs(root: Path) -> dict[str, list[dict]]:
    runs = {}
    for curve in sorted(root.glob("lr_*/training_curve.json")):
        if re.fullmatch(r"lr_\d+(?:\.\d+)?e[+-]?\d+", curve.parent.name) is None:
            continue
        rows = json.loads(curve.read_text())
        if rows:
            runs[curve.parent.name] = rows
    if not runs:
        raise FileNotFoundError(f"no LR training curves under {root}")
    return runs


def scored(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row.get("val_R2") is not None and row.get("val_mPQ+") is not None]


def learning_rate(rows: list[dict]) -> float:
    return float(rows[0]["learning_rate"])


def matching_run(runs: dict[str, list[dict]], rate: float) -> list[dict] | None:
    for rows in runs.values():
        if np.isclose(learning_rate(rows), rate, rtol=0.0, atol=1e-12):
            return rows
    return None


def row_at_epoch(rows: list[dict], epoch: int) -> dict | None:
    return next((row for row in rows if int(row["epoch"]) == int(epoch)), None)


def plot_family(root: Path, output: Path, title: str, control_root: Path | None = None) -> None:
    runs = load_runs(root)
    control_runs = load_runs(control_root) if control_root is not None else {}
    ordered = sorted(runs.items(), key=lambda item: learning_rate(item[1]))
    fig, axes = plt.subplots(3, 2, figsize=(15, 13), dpi=150, constrained_layout=True)
    panels = [
        (axes[0, 0], "val_R2", "Validation count performance", "macro R²", True),
        (axes[0, 1], "val_mPQ+", "Validation typed panoptic quality", "mPQ+", True),
        (axes[1, 0], "val_mDQ+", "Typed detection quality", "mDQ+", True),
        (axes[1, 1], "val_mSQ+", "Typed matched-mask quality", "mSQ+", True),
        (
            axes[2, 0],
            "val_loss",
            "Recipe-specific validation objective (compare trends, not levels)",
            "loss under each recipe",
            False,
        ),
    ]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for run_index, (label, rows) in enumerate(ordered):
        color = colors[run_index % len(colors)]
        rate = learning_rate(rows)
        legend = f"LR {rate:g} · intervention"
        control_rows = matching_run(control_runs, rate) if control_runs else None
        for axis, field, _, _, score_only in panels:
            subset = scored(rows) if score_only else rows
            if not subset or any(row.get(field) is None for row in subset):
                continue
            axis.plot(
                [int(row["epoch"]) for row in subset],
                [float(row[field]) for row in subset],
                marker="o",
                color=color,
                label=legend,
            )
            if control_rows is not None:
                control_subset = scored(control_rows) if score_only else control_rows
                if control_subset and all(row.get(field) is not None for row in control_subset):
                    axis.plot(
                        [int(row["epoch"]) for row in control_subset],
                        [float(row[field]) for row in control_subset],
                        marker="o",
                        linestyle="--",
                        alpha=0.72,
                        color=color,
                        label=f"LR {rate:g} · ordinary control",
                    )
    for axis, _, panel_title, ylabel, _ in panels:
        axis.axhline(0, color="0.65", linewidth=0.8)
        axis.set_title(panel_title)
        axis.set_xlabel("epoch")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=8)

    bar_axis = axes[2, 1]
    width = 0.8 / max(len(ordered), 1)
    x = np.arange(len(CLASS_NAMES), dtype=np.float64)
    for run_index, (label, rows) in enumerate(ordered):
        available = scored(rows)
        if not available:
            continue
        latest = max(available, key=lambda row: int(row["epoch"]))
        class_field = "val_per_class_PQ" if control_runs else "val_per_class_R2"
        values = [float(latest[class_field][name]) for name in CLASS_NAMES]
        control_rows = matching_run(control_runs, learning_rate(rows)) if control_runs else None
        if control_rows is not None:
            control_latest = row_at_epoch(control_rows, int(latest["epoch"]))
            if control_latest is not None:
                values = [
                    value - float(control_latest[class_field][name])
                    for name, value in zip(CLASS_NAMES, values)
                ]
        offset = (run_index - (len(ordered) - 1) / 2) * width
        rate = learning_rate(rows)
        bar_axis.bar(x + offset, values, width=width, label=f"LR {rate:g} · epoch {latest['epoch']}")
    bar_axis.axhline(0, color="0.55", linewidth=0.9)
    bar_axis.set_xticks(x, CLASS_NAMES, rotation=28, ha="right")
    bar_axis.set_ylabel("Δ PQ" if control_runs else "R²")
    bar_axis.set_title(
        "Per-class intervention − exact-control PQ" if control_runs
        else "Per-class R² at latest scored epoch"
    )
    bar_axis.grid(axis="y", alpha=0.22)
    bar_axis.legend(frameon=False, fontsize=8)

    fig.suptitle(title, fontsize=15)
    fig.patch.set_facecolor("white")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--control-root", type=Path)
    args = parser.parse_args()
    plot_family(args.root, args.out, args.title, args.control_root)


if __name__ == "__main__":
    main()
