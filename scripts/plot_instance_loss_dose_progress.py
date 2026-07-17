#!/usr/bin/env python3
"""Plot E42's selected-LR dose response with metric and directional diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PANELS = (
    ("val_R2", "Macro count R²"),
    ("val_mPQ+", "Typed panoptic quality (mPQ+)"),
    ("val_mDQ+", "Typed detection quality (mDQ+)"),
    ("val_mSQ+", "Typed segmentation quality (mSQ+)"),
    ("val_bPQ", "Binary panoptic quality"),
    ("val_binary_DQ", "Binary detection quality"),
    ("val_count_error.mean_signed_error", "Mean signed count error"),
    ("val_count_error.under_error_lt_minus_10_fraction", "Undercount error < −10"),
    ("val_count_error.over_error_gt_10_fraction", "Overcount error > +10"),
)


def nested(row: dict, key: str) -> float | None:
    value: object = row
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    if value is None or not np.isfinite(float(value)):
        return None
    return float(value)


def scored(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        row for row in json.loads(path.read_text())
        if row.get("val_R2") is not None and row.get("val_mPQ+") is not None
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--learning-rate-label", default="lr_1e-4")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    series = [("ordinary loss · blend 0", scored(args.control / "training_curve.json"), "#222222", "--")]
    colors = {"0p25": "#2c7fb8", "0p5": "#f28e2b", "1": "#d62728"}
    labels = {"0p25": "instance-equalized blend 0.25", "0p5": "instance-equalized blend 0.5", "1": "instance-equalized blend 1.0"}
    for blend in ("0p25", "0p5", "1"):
        rows = scored(args.root / f"blend_{blend}" / args.learning_rate_label / "training_curve.json")
        if rows:
            series.append((labels[blend], rows, colors[blend], "-"))

    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    fig.subplots_adjust(top=0.88, bottom=0.08, hspace=0.42, wspace=0.22)
    for axis, (key, title) in zip(axes.flat, PANELS):
        for label, rows, color, linestyle in series:
            points = [(int(row["epoch"]), nested(row, key)) for row in rows]
            points = [(epoch, value) for epoch, value in points if value is not None]
            if not points:
                continue
            epochs, values = zip(*points)
            axis.plot(epochs, values, marker="o", linewidth=2, linestyle=linestyle, color=color, label=label)
            axis.annotate(
                f"{values[-1]:.3f}", (epochs[-1], values[-1]), xytext=(4, 3),
                textcoords="offset points", fontsize=8, color=color,
            )
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.2)
        if key == "val_R2" or "signed_error" in key:
            axis.axhline(0, color="#777777", linewidth=0.8)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.945),
        ncol=max(1, len(legend_labels)), frameon=False,
    )
    fig.suptitle(
        "E42 instance-equalized loss dose response · LR 1e-4 · 711-patch development validation",
        fontsize=16, y=0.985,
    )
    fig.text(
        0.5, 0.02,
        "Error = prediction − ground truth. Metric-specific selection remains validation-only; unscored epochs are omitted.",
        ha="center", fontsize=9, color="#555555",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
