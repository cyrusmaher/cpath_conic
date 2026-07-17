#!/usr/bin/env python3
"""Render a live, validation-only view of a multi-LR HoVer-Net pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = (
    ("val_R2", "Validation count performance", "macro R²"),
    ("val_mPQ+", "Validation segmentation/type performance", "mPQ+"),
    ("val_mDQ+", "Typed detection quality", "mDQ+"),
    ("val_mSQ+", "Typed segmentation quality", "mSQ+"),
)


def load_runs(root: Path) -> list[tuple[str, list[dict]]]:
    runs = []
    for path in sorted(root.glob("lr_*/training_curve.json")):
        rows = json.loads(path.read_text())
        if rows:
            rate = float(rows[0]["learning_rate"])
            runs.append((f"LR {rate:g}", rows))
    return sorted(runs, key=lambda item: float(item[1][0]["learning_rate"]))


def scored(rows: list[dict], field: str) -> list[dict]:
    return [row for row in rows if row.get(field) is not None]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--reference-r2", type=float)
    parser.add_argument("--reference-mpq", type=float)
    args = parser.parse_args()

    runs = load_runs(args.root)
    control_runs = dict(load_runs(args.control_root)) if args.control_root is not None else {}
    if not runs:
        raise FileNotFoundError(f"No training curves under {args.root}")

    fig, axes = plt.subplots(3, 2, figsize=(13, 11), dpi=150)
    references = {"val_R2": args.reference_r2, "val_mPQ+": args.reference_mpq}
    for axis, (field, title, ylabel) in zip(axes.ravel()[:4], METRICS):
        for label, rows in runs:
            points = scored(rows, field)
            if points:
                line = axis.plot(
                    [row["epoch"] for row in points],
                    [row[field] for row in points],
                    marker="o",
                    linewidth=1.8,
                    label=f"E42 {label}" if control_runs else label,
                )[0]
                control_points = scored(control_runs.get(label, []), field)
                if control_points:
                    axis.plot(
                        [row["epoch"] for row in control_points],
                        [row[field] for row in control_points],
                        marker="o",
                        linewidth=1.3,
                        linestyle="--",
                        alpha=.75,
                        color=line.get_color(),
                        label=f"ordinary-loss {label}",
                    )
        reference = references.get(field)
        if reference is not None:
            axis.axhline(reference, color="#7c3aed", linestyle="--", linewidth=1.4,
                         label="selected matched ResNet endpoint")
        axis.axhline(0, color="#111827", linewidth=.6, alpha=.35)
        axis.set(title=title, xlabel="epoch", ylabel=ylabel)
        axis.grid(alpha=.2)
        axis.legend(frameon=False, fontsize=8)

    loss_axis = axes[2, 0]
    for label, rows in runs:
        line = loss_axis.plot(
            [row["epoch"] for row in rows],
            [row["val_loss"] for row in rows],
            marker="o",
            markersize=3,
            label=f"E42 {label}" if control_runs else label,
        )[0]
        control_rows = control_runs.get(label, [])
        if control_rows:
            loss_axis.plot(
                [row["epoch"] for row in control_rows],
                [row["val_loss"] for row in control_rows],
                marker="o",
                markersize=2.5,
                linewidth=1.2,
                linestyle="--",
                alpha=.75,
                color=line.get_color(),
                label=f"ordinary-loss {label}",
            )
    loss_axis.set(title="Official six-term validation loss", xlabel="epoch", ylabel="loss")
    loss_axis.grid(alpha=.2)
    loss_axis.legend(frameon=False, fontsize=8)

    class_axis = axes[2, 1]
    available = []
    for label, rows in runs:
        points = scored(rows, "val_R2")
        if points and points[-1].get("val_per_class_R2"):
            available.append((label, points[-1]))
    if available:
        classes = list(available[0][1]["val_per_class_R2"])
        x = np.arange(len(classes))
        width = .8 / len(available)
        for index, (label, row) in enumerate(available):
            values = [row["val_per_class_R2"][name] for name in classes]
            control_row = next(
                (
                    candidate
                    for candidate in control_runs.get(label, [])
                    if int(candidate["epoch"]) == int(row["epoch"])
                    and candidate.get("val_per_class_R2")
                ),
                None,
            )
            if control_row is not None:
                values = [
                    value - control_row["val_per_class_R2"][name]
                    for name, value in zip(classes, values)
                ]
            class_axis.bar(
                x - .4 + width / 2 + index * width,
                values,
                width,
                label=f"{label} (epoch {row['epoch']})",
            )
        class_axis.set_xticks(x, classes, rotation=28, ha="right")
        class_axis.legend(frameon=False, fontsize=8)
    class_axis.axhline(0, color="#111827", linewidth=.6, alpha=.35)
    if control_runs:
        # Rare-class R² can have very large paired deltas when an exact control
        # collapses. Symlog keeps those outliers visible without flattening all
        # common-class effects around zero.
        class_axis.set_yscale("symlog", linthresh=.05)
    class_axis.set(
        title=(
            "Per-class R² delta vs exact control at latest scored epoch"
            if control_runs
            else "Per-class R² at each LR's latest scored epoch"
        ),
        ylabel="ΔR² vs exact control" if control_runs else "R²",
    )
    class_axis.grid(axis="y", alpha=.2)

    fig.suptitle(args.title, fontsize=14)
    fig.text(.5, .006, "Development validation only · unscored training epochs are omitted from metric panels",
             ha="center", fontsize=9, color="#4b5563")
    fig.tight_layout(rect=(0, .02, 1, .98))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
