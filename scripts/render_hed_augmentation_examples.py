#!/usr/bin/env python
"""Render deterministic per-source HED augmentation examples for review."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_hovernet_our_split import EmpiricalHEDTargetBank, hed_stain_augmentation_array


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--target-jitter", type=float, default=0.05)
    parser.add_argument("--tail-expansion", type=float, default=0.1)
    args = parser.parse_args()
    metadata = pd.read_csv(args.prepared / "metadata.csv")
    with np.load(args.profile) as profile:
        concentrations = profile["concentrations"].astype(np.float32)
        sources = profile["sources"].astype(str)
        splits = profile["splits"].astype(str)
    bank = EmpiricalHEDTargetBank(
        concentrations[splits == "train"],
        sources[splits == "train"],
        jitter=args.target_jitter,
        tail_expansion=args.tail_expansion,
    )
    examples = metadata.loc[metadata.split.eq("train")].groupby("source", sort=True).head(1)
    fig, axes = plt.subplots(len(examples), 4, figsize=(12, 3 * len(examples)), dpi=140, squeeze=False)
    for row_index, row in enumerate(examples.itertuples(index=False)):
        image = np.asarray(
            Image.open(args.prepared / "images" / f"{int(row.patch_id):05d}.png").convert("RGB"),
            dtype=np.uint8,
        )
        axes[row_index, 0].imshow(image)
        axes[row_index, 0].set_title(f"{row.source} · patch {int(row.patch_id)}\noriginal")
        for variant in range(3):
            rng = np.random.default_rng(1000 + row_index * 10 + variant)
            target, target_source = bank.sample(rng)
            augmented = hed_stain_augmentation_array(
                image,
                rng,
                probability=1.0,
                target_concentration=target,
            )
            axes[row_index, variant + 1].imshow(augmented)
            axes[row_index, variant + 1].set_title(
                f"target {target_source}\nH={target[0]:.3f}, E={target[1]:.3f}"
            )
        for axis in axes[row_index]:
            axis.axis("off")
    fig.suptitle(
        f"E36 empirical H/E target transfer — observed pair + ±{100 * args.target_jitter:g}% jitter",
        fontsize=14,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
