#!/usr/bin/env python
"""Plot LR-sweep curves from metric-aligned classifier or LoRA runs."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def values(rows, name):
    return [float(row[name]) for row in rows]


def selected_row(rows: list[dict[str, str]], metric_fields: list[str]) -> dict[str, str]:
    """Select the point recorded by training, falling back to the first metric."""
    if "selection_score" in rows[0]:
        return max(rows, key=lambda row: float(row["selection_score"]))
    if not metric_fields:
        raise ValueError("Curve has no validation metric or selection_score")
    return max(rows, key=lambda row: float(row[metric_fields[0]]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curve", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="CoNIC validation-selected learning-rate sweep")
    args = parser.parse_args()
    with args.curve.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Curve CSV is empty")
    fields = set(rows[0])
    if "threshold" in fields and "class" in fields and "val_class_PQ" in fields:
        fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), dpi=140, sharex=True)
        for axis, class_name in zip(axes.ravel(), sorted({row["class"] for row in rows})):
            subset = sorted((row for row in rows if row["class"] == class_name), key=lambda row: float(row["threshold"]))
            thresholds = values(subset, "threshold")
            scores = values(subset, "val_class_PQ")
            selected = max(subset, key=lambda row: (float(row["val_class_PQ"]), -float(row["threshold"])))
            axis.plot(thresholds, scores)
            axis.scatter([float(selected["threshold"])], [float(selected["val_class_PQ"])], s=60, facecolor="gold", edgecolor="black")
            axis.set_title(class_name)
            axis.set_ylabel("validation class PQ")
            axis.grid(alpha=0.25)
        for axis in axes[-1]:
            axis.set_xlabel("top-1 rejection threshold")
        fig.suptitle(args.title)
        fig.tight_layout()
        fig.patch.set_facecolor("white")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return
    if "alpha_a" in fields:
        if "class" in fields and "val_mPQ+" not in fields:
            fig, axis = plt.subplots(figsize=(8.5, 5.2), dpi=140)
            for class_name in sorted({row["class"] for row in rows}):
                subset = sorted((row for row in rows if row["class"] == class_name), key=lambda row: float(row["alpha_a"]))
                alphas = values(subset, "alpha_a")
                scores = values(subset, "val_R2")
                axis.plot(alphas, scores, marker="o", markersize=3, label=class_name)
                selected = max(subset, key=lambda row: (float(row["val_R2"]), -abs(float(row["alpha_a"]) - 0.5)))
                axis.scatter([float(selected["alpha_a"])], [float(selected["val_R2"])], s=55, edgecolor="black", zorder=3)
            axis.set_xlabel("E17 count weight α (1−α weights E21)")
            axis.set_ylabel("validation per-class R²")
            axis.set_title(args.title)
            axis.grid(alpha=0.25)
            axis.legend(frameon=False, ncol=2)
            fig.tight_layout()
            fig.patch.set_facecolor("white")
            args.out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.out, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            return
        rows.sort(key=lambda row: float(row["alpha_a"]))
        alphas = values(rows, "alpha_a")
        fig, axis = plt.subplots(figsize=(7.5, 4.8), dpi=140)
        for field, label in [("val_R2", "validation R²"), ("val_mPQ+", "validation mPQ+"), ("selection_score", "mean selection score")]:
            axis.plot(alphas, values(rows, field), marker="o", label=label)
        selected = max(rows, key=lambda row: float(row["selection_score"]))
        axis.axvline(float(selected["alpha_a"]), color="black", linestyle="--", alpha=0.55, label=f"selected α={float(selected['alpha_a']):g}")
        axis.set_xlabel("focal-head logit weight α")
        axis.set_ylabel("validation metric")
        axis.set_title(args.title)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
        fig.tight_layout()
        fig.patch.set_facecolor("white")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return
    rates = sorted({float(row["learning_rate"]) for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=140)
    loss_field = "train_total" if "train_total" in fields else "train_loss"
    metric_fields = [name for name in ["val_R2", "val_mPQ+", "val_bPQ", "val_macro_f1"] if name in fields]
    for rate in rates:
        subset = [row for row in rows if float(row["learning_rate"]) == rate]
        epochs = [int(row["epoch"]) for row in subset]
        axes[0].plot(epochs, values(subset, loss_field), label=f"lr={rate:g}")
        for metric_index, metric in enumerate(metric_fields):
            axes[1].plot(
                epochs,
                values(subset, metric),
                linestyle="-" if metric_index == 0 else "--",
                label=f"lr={rate:g} {metric.removeprefix('val_')}",
            )
    if metric_fields:
        selected = selected_row(rows, metric_fields)
        primary_metric = "selection_score" if "selection_score" in fields else metric_fields[0]
        selected_x = int(selected["epoch"])
        selected_y = float(selected[primary_metric])
        axes[1].scatter(
            [selected_x],
            [selected_y],
            s=75,
            facecolor="gold",
            edgecolor="black",
            zorder=5,
            label=f"selected lr={float(selected['learning_rate']):g}, epoch={selected_x} ({primary_metric})",
        )
    axes[0].set_title(loss_field.replace("_", " "))
    axes[1].set_title("validation leaderboard proxies")
    for axis in axes:
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, frameon=False)
    fig.suptitle(args.title)
    fig.tight_layout()
    fig.patch.set_facecolor("white")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
