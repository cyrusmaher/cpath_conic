#!/usr/bin/env python3
"""Plot metric-specific HoVer-Net learning curves and their class-level drivers."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cpath_conic.constants import CLASS_NAMES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curve", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--r2-reference", type=float)
    parser.add_argument("--mpq-reference", type=float)
    args = parser.parse_args()

    rows = json.loads(args.curve.read_text())
    scored = [row for row in rows if row.get("val_R2") is not None]
    epochs = [row["epoch"] for row in scored]
    colors = dict(zip(CLASS_NAMES, plt.get_cmap("tab10").colors))
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), dpi=150)

    axes[0, 0].plot(epochs, [row["val_R2"] for row in scored], "o-", label="HoVer-Net R²")
    if args.r2_reference is not None:
        axes[0, 0].axhline(args.r2_reference, color="#7b3294", linestyle="--", label="CellViT E28")
    axes[0, 1].plot(epochs, [row["val_mPQ+"] for row in scored], "o-", label="HoVer-Net mPQ+")
    if args.mpq_reference is not None:
        axes[0, 1].axhline(args.mpq_reference, color="#7b3294", linestyle="--", label="CellViT E30")

    for class_name in CLASS_NAMES:
        axes[1, 0].plot(epochs, [row["val_per_class_R2"][class_name] for row in scored], "o-", label=class_name, color=colors[class_name])
        axes[1, 1].plot(epochs, [row["val_per_class_PQ"][class_name] for row in scored], "o-", label=class_name, color=colors[class_name])
        axes[2, 0].plot(epochs, [row["val_per_class_signed_error"][class_name] for row in scored], "o-", label=class_name, color=colors[class_name])
        axes[2, 1].plot(epochs, [row["val_per_class_count_ratio"][class_name] for row in scored], "o-", label=class_name, color=colors[class_name])

    axes[0, 0].set(title="Macro count metric", ylabel="R²")
    axes[0, 1].set(title="Pooled instance/type metric", ylabel="mPQ+")
    axes[1, 0].set(title="R² by cell class", ylabel="R²")
    axes[1, 1].set(title="PQ by cell class", ylabel="PQ")
    axes[2, 0].set(title="Directional count bias by cell class", ylabel="mean(predicted − ground truth)")
    axes[2, 1].set(title="Aggregate count ratio by cell class", ylabel="predicted / ground truth")
    axes[2, 0].axhline(0, color="#333", linewidth=1)
    axes[2, 1].axhline(1, color="#333", linewidth=1)

    learning_rate_changes = [
        row["epoch"]
        for previous, row in zip(rows, rows[1:])
        if float(row["learning_rate"]) != float(previous["learning_rate"])
    ]
    for axis in axes.flat:
        for epoch in learning_rate_changes:
            axis.axvline(epoch, color="#555", linestyle=":", linewidth=1, alpha=0.7)
        axis.set_xlabel("epoch")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False, fontsize=8, ncol=2 if axis.get_subplotspec().rowspan.start else 1)
    fig.suptitle("HoVer-Net validation trajectory: aggregate scores and class-level mechanisms")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
