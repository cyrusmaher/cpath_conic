#!/usr/bin/env python3
"""Plot validation curves and class drivers for HoVer-Net LR experiments."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cpath_conic.constants import CLASS_NAMES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--r2-reference", action="append", default=[], metavar="LABEL=VALUE")
    parser.add_argument("--mpq-reference", action="append", default=[], metavar="LABEL=VALUE")
    parser.add_argument("--title", default="Leakage-free HoVer-Net learning-rate bracket on our validation fold")
    args = parser.parse_args()
    runs = []
    for path in args.run:
        rows = json.loads((path / "training_curve.json").read_text())
        summary = json.loads((path / "summary.json").read_text()) if (path / "summary.json").exists() else {}
        learning_rate = summary.get("args", {}).get("learning_rate", rows[0]["learning_rate"])
        runs.append((float(learning_rate), rows, path.name))
    runs.sort(key=lambda item: item[0])

    def references(values: list[str]) -> list[tuple[str, float]]:
        parsed = []
        for value in values:
            label, number = value.rsplit("=", 1)
            parsed.append((label, float(number)))
        return parsed

    fig, axes = plt.subplots(3, 2, figsize=(13, 12), dpi=150)

    def mean_component(row: dict, direct_key: str, per_class_key: str) -> float:
        if row.get(direct_key) is not None:
            return float(row[direct_key])
        values = row.get(per_class_key, {})
        return float(np.mean([values[name] for name in CLASS_NAMES])) if values else float("nan")

    for learning_rate, rows, run_name in runs:
        scored = [
            row for row in rows
            if row.get("val_R2") is not None and np.isfinite(float(row["val_R2"]))
        ]
        label = f"{run_name} (LR {learning_rate:g})"
        axes[0, 0].plot([row["epoch"] for row in scored], [row["val_R2"] for row in scored], marker="o", label=label)
        axes[0, 1].plot([row["epoch"] for row in scored], [row["val_mPQ+"] for row in scored], marker="o", label=label)
        axes[1, 0].plot(
            [row["epoch"] for row in scored],
            [mean_component(row, "val_mDQ+", "val_per_class_DQ") for row in scored],
            marker="o", label=label,
        )
        axes[1, 1].plot(
            [row["epoch"] for row in scored],
            [mean_component(row, "val_mSQ+", "val_per_class_SQ") for row in scored],
            marker="o", label=label,
        )
        axes[2, 0].plot([row["epoch"] for row in rows], [row["val_loss"] for row in rows], marker="o", label=label)
        lr_changes = [
            row["epoch"]
            for previous, row in zip(rows, rows[1:])
            if float(row["learning_rate"]) != float(previous["learning_rate"])
        ]
        for epoch in lr_changes:
            for axis in (axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1], axes[2, 0]):
                axis.axvline(epoch, color="#555", linestyle=":", linewidth=1, alpha=0.65)

    width = 0.8 / max(len(runs), 1)
    positions = np.arange(len(CLASS_NAMES))
    for run_index, (learning_rate, rows, run_name) in enumerate(runs):
        scored = [row for row in rows if row.get("val_per_class_R2")]
        if not scored:
            continue
        final = scored[-1]
        values = [final["val_per_class_R2"].get(name, np.nan) for name in CLASS_NAMES]
        axes[2, 1].bar(positions - 0.4 + width / 2 + run_index * width, values, width=width, label=run_name)

    axes[0, 0].set(title="Validation count performance", xlabel="epoch", ylabel="macro R²")
    axes[0, 1].set(title="Validation segmentation/type performance", xlabel="epoch", ylabel="mPQ+")
    axes[1, 0].set(title="Typed detection quality", xlabel="epoch", ylabel="mDQ+")
    axes[1, 1].set(title="Typed segmentation quality", xlabel="epoch", ylabel="mSQ+")
    axes[2, 0].set(title="Official six-term validation loss", xlabel="epoch", ylabel="loss")
    axes[2, 1].set(title="Per-class R² at latest scored epoch", ylabel="R²", xticks=positions, xticklabels=CLASS_NAMES)
    axes[2, 1].tick_params(axis="x", labelrotation=30)
    reference_colors = ["#7b3294", "#008837", "#c51b7d", "#01665e"]
    for index, (label, value) in enumerate(references(args.r2_reference)):
        axes[0, 0].axhline(value, color=reference_colors[index % len(reference_colors)], linestyle="--", linewidth=1.2, label=label)
    for index, (label, value) in enumerate(references(args.mpq_reference)):
        axes[0, 1].axhline(value, color=reference_colors[index % len(reference_colors)], linestyle="--", linewidth=1.2, label=label)
    for axis in axes.flat:
        axis.axhline(0, color="#777", linewidth=0.8, alpha=0.5)
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle(args.title)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
