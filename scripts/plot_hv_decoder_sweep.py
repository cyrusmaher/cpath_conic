#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.sweep.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ["binary_threshold", "edge_threshold", "object_size", "ksize", "opening_size", "min_nucleus_size", "bPQ", "count_ratio"]:
            row[key] = float(row[key])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), dpi=140)
    coarse = [row for row in rows if row["stage"] == "object_kernel"]
    first_object_size = min(row["object_size"] for row in coarse)
    subset = sorted((row for row in coarse if row["object_size"] == first_object_size), key=lambda row: row["ksize"])
    axes[0].plot([row["ksize"] for row in subset], [row["bPQ"] for row in subset], marker="o", label="initial sweep")
    confirmation = sorted((row for row in rows if row["stage"] == "kernel_confirmation"), key=lambda row: row["ksize"])
    if confirmation:
        axes[0].plot(
            [row["ksize"] for row in confirmation],
            [row["bPQ"] for row in confirmation],
            marker="o",
            label="after threshold/morphology tuning",
        )
        selected = max(confirmation, key=lambda row: row["bPQ"])
        axes[0].scatter([selected["ksize"]], [selected["bPQ"]], s=85, facecolors="none", edgecolors="#e45756", linewidths=2, zorder=5)
    axes[0].set(title="Kernel / marker sweep", xlabel="Sobel kernel", ylabel="validation bPQ")
    axes[0].legend(frameon=False, fontsize=8)

    thresholds = [row for row in rows if row["stage"] == "thresholds"]
    binary_values = sorted({row["binary_threshold"] for row in thresholds})
    edge_values = sorted({row["edge_threshold"] for row in thresholds})
    grid = np.asarray([[next(row["bPQ"] for row in thresholds if row["binary_threshold"] == binary and row["edge_threshold"] == edge) for edge in edge_values] for binary in binary_values])
    image = axes[1].imshow(grid, cmap="viridis", vmin=min(row["bPQ"] for row in thresholds), vmax=max(row["bPQ"] for row in thresholds))
    axes[1].set(title="Foreground / edge thresholds", xlabel="edge threshold", ylabel="foreground threshold", xticks=range(len(edge_values)), xticklabels=edge_values, yticks=range(len(binary_values)), yticklabels=binary_values)
    for y in range(len(binary_values)):
        for x in range(len(edge_values)):
            axes[1].text(x, y, f"{grid[y, x]:.3f}", ha="center", va="center", color="white" if grid[y, x] < grid.mean() else "black", fontsize=8)
    fig.colorbar(image, ax=axes[1], fraction=0.046)

    morphology = [row for row in rows if row["stage"] == "morphology"]
    labels = [f"open {int(row['opening_size'])}\nmin {int(row['min_nucleus_size'])}" for row in morphology]
    axes[2].bar(range(len(morphology)), [row["bPQ"] for row in morphology], color="#4c78a8")
    tta = next((row for row in rows if row["stage"] == "flip_tta_locked_decoder"), None)
    if tta:
        axes[2].axhline(tta["bPQ"], color="#e45756", linestyle="--", label=f"flip-TTA {tta['bPQ']:.3f}")
        axes[2].legend(frameon=False, fontsize=8)
    axes[2].set(title="Morphology + locked TTA", ylabel="validation bPQ", xticks=range(len(labels)), xticklabels=labels)
    axes[2].tick_params(axis="x", labelrotation=50, labelsize=7)
    for axis in axes:
        axis.grid(alpha=0.2, axis="y")
    fig.suptitle("CellViT HV decoder selected only on 711 validation patches")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
