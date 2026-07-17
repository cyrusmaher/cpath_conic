#!/usr/bin/env python3
"""Plot E43 auxiliary-weight trajectories at the validation-selected LR."""

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


def _load(path: Path) -> list[dict]:
    return json.loads(path.read_text()) if path.exists() else []


def _scored(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row.get("val_R2") is not None and row.get("val_mPQ+") is not None]


def _weight(path: Path) -> float:
    match = re.fullmatch(r"weight_(\d+)p(\d+)", path.parent.parent.name)
    if match is None:
        raise ValueError(f"cannot parse auxiliary weight from {path}")
    return float(f"{match.group(1)}.{match.group(2)}")


def plot(root: Path, control_curve: Path, output: Path, learning_rate: float) -> None:
    curves = sorted(
        root.glob("weight_*/lr_*/training_curve.json"),
        key=_weight,
    )
    candidates = [(_weight(path), _load(path)) for path in curves]
    candidates = [
        (weight, rows)
        for weight, rows in candidates
        if rows and np.isclose(float(rows[0]["learning_rate"]), learning_rate, rtol=0.0, atol=1e-12)
    ]
    if not candidates:
        raise FileNotFoundError(f"no E43 dose curves at LR {learning_rate:g} under {root}")
    control = _load(control_curve)
    control_scored = _scored(control)
    control_by_epoch = {int(row["epoch"]): row for row in control_scored}

    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), dpi=150, constrained_layout=True)
    panels = (
        (axes[0, 0], "val_R2", "Count objective", "macro R²"),
        (axes[0, 1], "val_mPQ+", "Typed panoptic quality", "mPQ+"),
        (axes[0, 2], "val_mDQ+", "Typed detection quality", "mDQ+"),
        (axes[1, 0], "val_mSQ+", "Matched-mask quality", "mSQ+"),
        (axes[1, 1], "val_gt_instance_type_nll", "Per-nucleus type calibration", "type NLL · lower is better"),
    )
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for axis, field, title, ylabel in panels:
        if control_scored and all(row.get(field) is not None for row in control_scored):
            axis.plot(
                [int(row["epoch"]) for row in control_scored],
                [float(row[field]) for row in control_scored],
                "o--",
                color="0.25",
                alpha=0.72,
                label="ordinary loss · weight 0",
            )
        for index, (weight, rows) in enumerate(candidates):
            available = _scored(rows)
            if not available or any(row.get(field) is None for row in available):
                continue
            axis.plot(
                [int(row["epoch"]) for row in available],
                [float(row[field]) for row in available],
                "o-",
                color=colors[index % len(colors)],
                label=f"auxiliary weight {weight:g}",
            )
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=8)

    bar_axis = axes[1, 2]
    available_candidates = [(weight, _scored(rows)) for weight, rows in candidates]
    available_candidates = [(weight, rows) for weight, rows in available_candidates if rows]
    width = 0.8 / max(len(available_candidates), 1)
    x = np.arange(len(CLASS_NAMES), dtype=np.float64)
    for index, (weight, rows) in enumerate(available_candidates):
        latest = max(rows, key=lambda row: int(row["epoch"]))
        matched = control_by_epoch.get(int(latest["epoch"]))
        if matched is None:
            continue
        values = [
            float(latest["val_per_class_PQ"][name]) - float(matched["val_per_class_PQ"][name])
            for name in CLASS_NAMES
        ]
        offset = (index - (len(available_candidates) - 1) / 2) * width
        bar_axis.bar(
            x + offset,
            values,
            width=width,
            color=colors[index % len(colors)],
            label=f"weight {weight:g} · epoch {latest['epoch']}",
        )
    bar_axis.axhline(0, color="0.35", linewidth=0.9)
    bar_axis.set_xticks(x, CLASS_NAMES, rotation=28, ha="right")
    bar_axis.set_ylabel("candidate − exact-control PQ")
    bar_axis.set_title("Class mechanism at latest scored checkpoint")
    bar_axis.grid(axis="y", alpha=0.22)
    bar_axis.legend(frameon=False, fontsize=8)

    fig.suptitle(
        f"E43 per-nucleus type-loss dose response · LR {learning_rate:g} · development validation only",
        fontsize=15,
    )
    fig.patch.set_facecolor("white")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--control-curve", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    args = parser.parse_args()
    plot(args.root, args.control_curve, args.out, args.learning_rate)


if __name__ == "__main__":
    main()
