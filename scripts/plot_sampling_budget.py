#!/usr/bin/env python
"""Visualize source and class exposure induced by CoNIC patch samplers."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cpath_conic.constants import CLASS_NAMES
from cpath_conic.sampling import effective_sample_size, expected_unique_draws, source_class_patch_weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--train-ids", type=Path, help="Optional exact development-fold training IDs")
    args = parser.parse_args()
    metadata = pd.read_csv(args.prepared / "metadata.csv")
    if args.train_ids:
        train_ids = np.load(args.train_ids).astype(np.int32)
        train = metadata.loc[metadata.patch_id.isin(train_ids)].copy()
    else:
        train = metadata.loc[metadata.split.eq("train")].copy()
    counts = train[[f"count_{name}" for name in CLASS_NAMES]].to_numpy(dtype=np.float64)
    uniform = np.ones(len(train))
    recipes = {
        "uniform · no replacement": {"weights": uniform, "replacement": False},
        "uniform · replacement": {"weights": uniform, "replacement": True},
        "10% class": {
            "weights": source_class_patch_weights(
                train.source.to_numpy(), counts, source_fraction=0.0, class_fraction=0.1
            ),
            "replacement": True,
        },
        "25% class": {
            "weights": source_class_patch_weights(
                train.source.to_numpy(), counts, source_fraction=0.0, class_fraction=0.25
            ),
            "replacement": True,
        },
        "50% class": {
            "weights": source_class_patch_weights(
                train.source.to_numpy(), counts, source_fraction=0.0, class_fraction=0.5
            ),
            "replacement": True,
        },
        "50% source": {
            "weights": source_class_patch_weights(
                train.source.to_numpy(), counts, source_fraction=0.5, class_fraction=0.0
            ),
            "replacement": True,
        },
        "25% source + 25% class": {
            "weights": source_class_patch_weights(
                train.source.to_numpy(), counts, source_fraction=0.25, class_fraction=0.25
            ),
            "replacement": True,
        },
    }
    sources = sorted(train.source.unique())
    fig, axes = plt.subplots(1, 3, figsize=(19, 5), dpi=140)
    x = np.arange(len(recipes))
    bottom = np.zeros(len(recipes))
    for source in sources:
        values = []
        mask = train.source.eq(source).to_numpy()
        for recipe in recipes.values():
            weights = recipe["weights"]
            values.append(100.0 * weights[mask].sum() / weights.sum())
        axes[0].bar(x, values, bottom=bottom, label=source)
        bottom += values
    axes[0].set_xticks(x, recipes, rotation=12, ha="right")
    axes[0].set_ylabel("expected sampled patches (%)")
    axes[0].set_title("Source representation budget")
    axes[0].legend(frameon=False, ncol=2)

    width = 0.8 / len(recipes)
    for recipe_index, (name, recipe) in enumerate(recipes.items()):
        weights = recipe["weights"]
        expected = (weights[:, None] * counts).sum(axis=0) / weights.sum()
        expected = 100.0 * expected / expected.sum()
        offset = (recipe_index - (len(recipes) - 1) / 2) * width
        axes[1].bar(np.arange(len(CLASS_NAMES)) + offset, expected, width, label=name)
    axes[1].set_xticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES, rotation=25, ha="right")
    axes[1].set_ylabel("expected sampled cell mass (%)")
    axes[1].set_title("Cell-type exposure induced by patch sampling")
    axes[1].legend(frameon=False, fontsize=8)

    report = {
        "n_patches": int(len(train)),
        "interpretation": (
            "Uniform replacement is the mechanism control for weighted samplers: it preserves the ordinary "
            "expected source/class prior while matching their duplicate-producing draw mode."
        ),
        "recipes": {},
    }
    coverage = []
    ess = []
    for name, recipe in recipes.items():
        weights = recipe["weights"]
        probabilities = weights / weights.sum()
        replacement = bool(recipe["replacement"])
        unique = expected_unique_draws(weights, len(train)) if replacement else float(len(train))
        effective = effective_sample_size(weights)
        coverage.append(100.0 * unique / len(train))
        ess.append(100.0 * effective / len(train))
        source_mass = {
            str(source): float(probabilities[train.source.eq(source).to_numpy()].sum()) for source in sources
        }
        expected_cells = (probabilities[:, None] * counts).sum(axis=0)
        report["recipes"][name] = {
            "sampling_mode": "with_replacement" if replacement else "without_replacement",
            "source_mass": source_mass,
            "expected_cells_per_draw": dict(zip(CLASS_NAMES, expected_cells.tolist())),
            "expected_unique_patches": unique,
            "expected_unique_fraction": unique / len(train),
            "effective_sample_size": effective,
            "effective_sample_size_fraction": effective / len(train),
            "minimum_weight": float(weights.min()),
            "maximum_weight": float(weights.max()),
        }
    bar_x = np.arange(len(recipes))
    axes[2].bar(bar_x - 0.18, coverage, 0.36, label="expected unique")
    axes[2].bar(bar_x + 0.18, ess, 0.36, label="weight ESS")
    axes[2].set_xticks(bar_x, recipes, rotation=12, ha="right")
    axes[2].set_ylabel("fraction of patch set (%)")
    axes[2].set_title("Per-epoch diversity cost")
    axes[2].set_ylim(0, 105)
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("E37/E45 source/class sampling-budget and replacement-control audit")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    report_path = args.report or args.out.with_suffix(".json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
