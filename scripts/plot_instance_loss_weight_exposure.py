#!/usr/bin/env python3
"""Audit and visualize instance-equalized pixel weights from training labels."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cpath_conic.constants import CLASS_NAMES


def collect(prepared: Path, patch_ids: np.ndarray, blend: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    areas, weights, classes = [], [], []
    for patch_id in patch_ids.astype(int):
        label = np.load(prepared / "labels" / f"{patch_id:05d}.npy", allow_pickle=True).item()
        instance_map, class_map = label["inst_map"], label["class_map"]
        instance_ids, instance_areas = np.unique(instance_map[instance_map > 0], return_counts=True)
        if not len(instance_ids):
            continue
        equal_mass = float(instance_areas.mean())
        for instance_id, area in zip(instance_ids, instance_areas):
            values = class_map[instance_map == instance_id]
            values = values[(values >= 1) & (values <= len(CLASS_NAMES))]
            class_id = int(np.bincount(values).argmax()) if len(values) else 0
            if class_id:
                areas.append(int(area))
                weights.append(1.0 + blend * (equal_mass / float(area) - 1.0))
                classes.append(class_id)
    return np.asarray(areas), np.asarray(weights), np.asarray(classes)


def quantiles(values: np.ndarray) -> dict[str, float]:
    probabilities = (0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1)
    return {f"p{100 * probability:g}": float(np.quantile(values, probability)) for probability in probabilities}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--train-ids", type=Path, required=True)
    parser.add_argument("--blend", type=float, default=0.5)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-plot", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.blend <= 1:
        parser.error("--blend must lie in [0, 1]")
    patch_ids = np.load(args.train_ids).astype(np.int32)
    areas, weights, classes = collect(args.prepared, patch_ids, args.blend)
    report = {
        "protocol": "development-training labels only; one final pixel weight per GT nucleus",
        "train_patches": int(len(patch_ids)),
        "nuclei": int(len(areas)),
        "instance_loss_blend": args.blend,
        "area_pixels": quantiles(areas),
        "pixel_weight": quantiles(weights),
        "weight_tail_fraction": {
            f"greater_than_{threshold:g}": float(np.mean(weights > threshold)) for threshold in (2, 4, 8)
        },
        "by_class": {
            name: {
                "nuclei": int(np.sum(classes == index)),
                "median_area_pixels": float(np.median(areas[classes == index])),
                "mean_pixel_weight": float(np.mean(weights[classes == index])),
                "p99_pixel_weight": float(np.quantile(weights[classes == index], 0.99)),
            }
            for index, name in enumerate(CLASS_NAMES, start=1)
        },
        "interpretation": (
            "Very high weights identify tiny GT components that can dominate local gradients despite preserved "
            "aggregate foreground mass. If uncapped equalization fails, test a capped, mass-renormalized variant."
        ),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=150)
    axes[0].hist(areas, bins=np.geomspace(1, max(int(areas.max()), 2), 55), color="#2563eb", alpha=.85)
    axes[0].set(xscale="log", yscale="log", xlabel="GT nucleus area (pixels)", ylabel="nuclei",
                title="Development-training nucleus sizes")
    axes[0].grid(alpha=.18)
    clipped = np.minimum(weights, 20)
    axes[1].hist(clipped, bins=60, color="#7c3aed", alpha=.85)
    axes[1].axvline(4, color="#dc2626", linestyle="--", label=f">4: {np.mean(weights > 4):.2%}")
    axes[1].set(xlabel="blend-0.5 pixel weight (display clipped at 20)", ylabel="nuclei",
                title="Tiny-instance weight tail")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=.18)
    class_values = [weights[classes == index] for index in range(1, len(CLASS_NAMES) + 1)]
    axes[2].boxplot(class_values, tick_labels=CLASS_NAMES, showfliers=False)
    axes[2].scatter(
        np.arange(1, len(CLASS_NAMES) + 1),
        [np.quantile(values, 0.99) for values in class_values],
        color="#dc2626", marker="D", s=28, zorder=3, label="99th percentile",
    )
    axes[2].set(yscale="log", ylabel="pixel weight", title="Weight distribution by GT class")
    axes[2].tick_params(axis="x", labelrotation=30)
    axes[2].grid(axis="y", alpha=.18)
    axes[2].legend(frameon=False)
    fig.suptitle("E42 instance-equalized loss exposure · training labels only")
    fig.tight_layout()
    args.out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_plot, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
