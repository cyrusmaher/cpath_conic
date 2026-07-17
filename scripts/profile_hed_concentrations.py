#!/usr/bin/env python3
"""Profile empirical H/E concentration pairs without touching locked test data."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_hovernet_our_split import hed_concentration


def summarize(values: np.ndarray) -> dict:
    return {
        "n": int(len(values)),
        "H": {name: float(value) for name, value in zip(
            ("q01", "q05", "median", "q95", "q99"), np.quantile(values[:, 0], (0.01, 0.05, 0.5, 0.95, 0.99))
        )},
        "E": {name: float(value) for name, value in zip(
            ("q01", "q05", "median", "q95", "q99"), np.quantile(values[:, 1], (0.01, 0.05, 0.5, 0.95, 0.99))
        )},
        "log_HE_correlation": float(np.corrcoef(np.log(values).T)[0, 1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Output NPZ target bank")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    args = parser.parse_args()

    metadata = pd.read_csv(args.prepared / "metadata.csv").sort_values("patch_id")
    selected = metadata.loc[metadata.split.isin(["train", "val"])].copy()
    if selected.empty:
        raise RuntimeError("no train/validation patches found")
    concentrations = []
    for offset, row in enumerate(selected.itertuples(index=False), start=1):
        image = np.asarray(
            Image.open(args.prepared / "images" / f"{int(row.patch_id):05d}.png").convert("RGB"),
            dtype=np.uint8,
        )
        concentrations.append(hed_concentration(image))
        if offset % 250 == 0 or offset == len(selected):
            print(f"H/E profile {offset}/{len(selected)}", flush=True)
    values = np.asarray(concentrations, dtype=np.float32)
    patch_ids = selected.patch_id.to_numpy(dtype=np.int32)
    sources = selected.source.astype(str).to_numpy(dtype=str)
    splits = selected.split.astype(str).to_numpy(dtype=str)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        patch_ids=patch_ids,
        concentrations=values,
        sources=sources,
        splits=splits,
        quantile=np.asarray(0.95, dtype=np.float32),
    )

    train_values = values[splits == "train"]
    train_low = np.quantile(train_values, 0.01, axis=0)
    train_high = np.quantile(train_values, 0.99, axis=0)
    validation_values = values[splits == "val"]
    report = {
        "descriptor": "per-patch joint 95th-percentile positive Ruifrok H/E concentrations",
        "locked_test_accessed": False,
        "overall": summarize(values),
        "by_split": {
            split: summarize(values[splits == split]) for split in sorted(np.unique(splits))
        },
        "by_source_and_split": {
            f"{source}/{split}": summarize(values[(sources == source) & (splits == split)])
            for source in sorted(np.unique(sources))
            for split in sorted(np.unique(splits[sources == source]))
        },
        "validation_inside_train_q01_q99_fraction": {
            "H": float(np.mean((validation_values[:, 0] >= train_low[0]) & (validation_values[:, 0] <= train_high[0]))),
            "E": float(np.mean((validation_values[:, 1] >= train_low[1]) & (validation_values[:, 1] <= train_high[1]))),
            "joint": float(np.mean(np.all((validation_values >= train_low) & (validation_values <= train_high), axis=1))),
        },
        "training_use_policy": "Profile contains train+validation for audit; every training run filters the target bank to its own train IDs only.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    colors = plt.get_cmap("tab10")
    for index, source in enumerate(sorted(np.unique(sources))):
        mask = sources == source
        axes[0].scatter(values[mask, 0], values[mask, 1], s=8, alpha=0.35, color=colors(index), label=source)
    axes[0].set_title("Observed joint H/E styles by institution")
    axes[0].legend(frameon=False, markerscale=2)
    for split, marker in (("train", "o"), ("val", "x")):
        mask = splits == split
        axes[1].scatter(values[mask, 0], values[mask, 1], s=8, alpha=0.35, marker=marker, label=split)
    axes[1].set_title("Train/validation concentration support")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("H concentration (patch p95)")
        axis.set_ylabel("E concentration (patch p95)")
        axis.grid(alpha=0.2)
    fig.suptitle("CoNIC development-set stain concentration profile (locked test excluded)")
    fig.tight_layout()
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
