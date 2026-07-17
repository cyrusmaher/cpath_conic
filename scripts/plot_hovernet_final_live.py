#!/usr/bin/env python3
"""Plot the live, validation-only trajectory for the final HoVer-Net fit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curve", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="Final class-weighted replacement HoVer-Net")
    parser.add_argument("--uniform-r2", type=float, default=0.796617)
    parser.add_argument("--uniform-mpq", type=float, default=0.439888)
    args = parser.parse_args()

    rows = json.loads(args.curve.read_text())
    if not rows:
        raise ValueError("training curve is empty")
    scored = [row for row in rows if row.get("val_mPQ+") is not None]
    best = max(scored, key=lambda row: float(row["val_mPQ+"])) if scored else None
    epochs = [int(row["epoch"]) for row in rows]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=150)
    loss, primary, components, classes = axes.ravel()

    loss.plot(epochs, [row["train_loss"] for row in rows], marker=".", label="training loss")
    loss.plot(epochs, [row["val_loss"] for row in rows], marker=".", label="validation loss")
    loss.set(title="Official six-term objective", ylabel="loss")

    if scored:
        scored_epochs = [int(row["epoch"]) for row in scored]
        primary.plot(scored_epochs, [row["val_R2"] for row in scored], "o-", label="R²")
        primary.plot(scored_epochs, [row["val_mPQ+"] for row in scored], "o-", label="mPQ+")
        primary.axhline(args.uniform_r2, color="#2563eb", linestyle="--", alpha=.7,
                        label="uniform HoVer native R²")
        primary.axhline(args.uniform_mpq, color="#dc2626", linestyle="--", alpha=.7,
                        label="uniform HoVer native mPQ+")
        primary.scatter([best["epoch"]], [best["val_mPQ+"]], marker="*", s=150,
                        color="#f59e0b", edgecolor="#111827", zorder=5, label="best mPQ+ so far")
        components.plot(scored_epochs, [row["val_mDQ+"] for row in scored], "o-", label="mDQ+")
        components.plot(scored_epochs, [row["val_mSQ+"] for row in scored], "o-", label="mSQ+")

        class_values = best.get("val_per_class_PQ", {})
        class_names = list(class_values)
        x = np.arange(len(class_names))
        classes.bar(x, [class_values[name] for name in class_names], color=plt.get_cmap("tab10").colors[:len(class_names)])
        classes.set_xticks(x, class_names, rotation=28, ha="right")
        classes.set_title(f"Per-class PQ at best scored epoch {int(best['epoch'])}")
        classes.set_ylabel("PQ")
    else:
        for axis in (primary, components, classes):
            axis.text(.5, .5, "Awaiting first scored checkpoint", ha="center", va="center",
                      transform=axis.transAxes, color="#667085")

    primary.set(title="Primary leaderboard metrics", ylabel="score")
    components.set(title="Why mPQ+ moves", ylabel="typed component")
    for axis in (loss, primary, components):
        axis.axvline(25, color="#475467", linestyle=":", linewidth=1, alpha=.8)
        axis.grid(axis="y", alpha=.2)
        axis.set_xlabel("epoch")
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(frameon=False, fontsize=8)
    classes.grid(axis="y", alpha=.2)

    fig.suptitle(args.title, fontsize=15)
    fig.text(
        .5,
        .008,
        "711-patch source-group-disjoint development validation only · LR 1e-4, scheduled 10× decay after epoch 25 · no locked-test inference",
        ha="center",
        fontsize=9,
        color="#475467",
    )
    fig.tight_layout(rect=(0, .025, 1, .97))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
