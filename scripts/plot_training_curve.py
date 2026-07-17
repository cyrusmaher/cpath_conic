#!/usr/bin/env python
"""Plot the classifier learning-rate sweep saved by run_cellvit_conic.py."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curve", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    with args.curve.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rates = sorted({float(row["learning_rate"]) for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=140)
    for rate in rates:
        subset = [row for row in rows if float(row["learning_rate"]) == rate]
        epochs = [int(row["epoch"]) for row in subset]
        label = f"lr={rate:g}"
        axes[0].plot(epochs, [float(row["train_loss"]) for row in subset], label=f"{label} train")
        axes[0].plot(epochs, [float(row["val_loss"]) for row in subset], linestyle="--", label=f"{label} val")
        axes[1].plot(epochs, [float(row["train_macro_f1"]) for row in subset], alpha=0.45, label=f"{label} train")
        axes[1].plot(epochs, [float(row["val_macro_f1"]) for row in subset], linewidth=2, label=f"{label} val")
    axes[0].set_title("Weighted cross-entropy")
    axes[1].set_title("Macro-F1 (selection metric)")
    for axis in axes:
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, frameon=False)
    fig.suptitle("CellViT++ CoNIC classifier learning-rate sweep")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
