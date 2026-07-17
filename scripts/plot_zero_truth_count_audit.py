#!/usr/bin/env python3
"""Plot a validation-first audit of false counts where one class is absent."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cpath_conic.constants import CLASS_NAMES, COUNT_COLUMNS


def subset(metadata: pd.DataFrame, split: str, source: str, count_column: str) -> pd.DataFrame:
    return metadata[
        metadata["split"].astype(str).eq(split)
        & metadata["source"].astype(str).eq(source)
        & metadata[count_column].eq(0)
    ].copy()


def threshold_rates(values: np.ndarray) -> np.ndarray:
    return np.asarray([(values > threshold).mean() for threshold in (0, 5, 10, 20)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--first-counts", type=Path, required=True)
    parser.add_argument("--second-counts", type=Path, required=True)
    parser.add_argument("--first-test-counts", type=Path)
    parser.add_argument("--second-test-counts", type=Path)
    parser.add_argument("--first-name", default="Raw HoVer-Net TTA")
    parser.add_argument("--second-name", default="E33 count blend")
    parser.add_argument("--source", default="dpath")
    parser.add_argument("--class-name", choices=CLASS_NAMES, default="epithelial")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    metadata = pd.read_csv(args.prepared / "metadata.csv")
    first = np.load(args.first_counts)
    second = np.load(args.second_counts)
    if first.shape != second.shape or first.ndim != 2 or first.shape[1] != len(CLASS_NAMES):
        raise ValueError("count arrays must be aligned N x 6 arrays")
    if len(metadata) != len(first):
        raise ValueError("count arrays must align to prepared metadata patch IDs")
    first_test = np.load(args.first_test_counts) if args.first_test_counts else first
    second_test = np.load(args.second_test_counts) if args.second_test_counts else second
    if first_test.shape != first.shape or second_test.shape != second.shape:
        raise ValueError("optional test count arrays must have the same full-dataset shape")

    class_index = CLASS_NAMES.index(args.class_name)
    count_column = COUNT_COLUMNS[class_index]
    validation = subset(metadata, "val", args.source, count_column)
    test = subset(metadata, "test", args.source, count_column)
    if validation.empty or test.empty:
        raise ValueError("both validation and test need supported true-zero strata")

    def values(frame: pd.DataFrame, array: np.ndarray) -> np.ndarray:
        return array[frame.patch_id.to_numpy(dtype=np.int64), class_index].astype(np.float64)

    val_first, val_second = values(validation, first), values(validation, second)
    test_first, test_second = values(test, first_test), values(test, second_test)
    colors = ("#315b7d", "#d9772b")
    labels = (args.first_name, args.second_name)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    bins = np.arange(0, max(25, int(max(val_first.max(), val_second.max()))) + 3, 2)
    axes[0, 0].hist(val_first, bins=bins, histtype="step", linewidth=2.5, color=colors[0], label=labels[0])
    axes[0, 0].hist(val_second, bins=bins, histtype="step", linewidth=2.5, color=colors[1], label=labels[1])
    axes[0, 0].set_title(f"Development validation · {len(validation)} {args.source.upper()} patches")
    axes[0, 0].set_xlabel(f"Predicted {args.class_name} count when truth = 0")
    axes[0, 0].set_ylabel("Patches")
    axes[0, 0].legend(frameon=False)

    x = np.arange(4)
    width = 0.36
    axes[0, 1].bar(x - width / 2, threshold_rates(val_first), width, color=colors[0], label=labels[0])
    axes[0, 1].bar(x + width / 2, threshold_rates(val_second), width, color=colors[1], label=labels[1])
    axes[0, 1].set_xticks(x, [">0", ">5", ">10", ">20"])
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_ylabel("Proportion of true-zero patches")
    axes[0, 1].set_title("Validation outlier prevalence · selection-relevant")
    axes[0, 1].legend(frameon=False)

    bins = np.arange(0, max(25, int(max(test_first.max(), test_second.max()))) + 3, 2)
    axes[1, 0].hist(test_first, bins=bins, histtype="step", linewidth=2.5, color=colors[0], label=labels[0])
    axes[1, 0].hist(test_second, bins=bins, histtype="step", linewidth=2.5, color=colors[1], label=labels[1])
    axes[1, 0].set_title(f"Retrospective internal test · {len(test)} patches · descriptive only")
    axes[1, 0].set_xlabel(f"Predicted {args.class_name} count when truth = 0")
    axes[1, 0].set_ylabel("Patches")

    connective_index = CLASS_NAMES.index("connective")
    connective_truth = validation[COUNT_COLUMNS[connective_index]].to_numpy(dtype=np.float64)
    axes[1, 1].scatter(connective_truth, val_first, s=35, alpha=0.75, color=colors[0], label=labels[0])
    axes[1, 1].scatter(connective_truth, val_second, s=35, alpha=0.75, color=colors[1], marker="x", label=labels[1])
    axes[1, 1].axhline(10, color="#a82424", linestyle="--", linewidth=1.5, label="10-cell false-count threshold")
    axes[1, 1].set_xlabel("Ground-truth connective count")
    axes[1, 1].set_ylabel(f"Predicted {args.class_name} count")
    axes[1, 1].set_title("Validation: absent-class false counts versus stromal burden")
    axes[1, 1].legend(frameon=False, fontsize=9)

    fig.suptitle(
        f"True-zero {args.class_name} audit: means can hide false-count prevalence",
        fontsize=15,
        fontweight="bold",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
