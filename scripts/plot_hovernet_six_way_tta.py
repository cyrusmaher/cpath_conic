#!/usr/bin/env python3
"""Create a self-contained visual explanation of E32 six-view spatial TTA."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--patch-id", type=int, default=3127)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    image = np.asarray(Image.open(args.prepared / "images" / f"{args.patch_id:05d}.png").convert("RGB"))
    views = [
        ("Identity", image),
        ("Horizontal flip", np.fliplr(image)),
        ("Vertical flip", np.flipud(image)),
        ("Rotate 90°", np.rot90(image, 1)),
        ("Rotate 180°", np.rot90(image, 2)),
        ("Rotate 270°", np.rot90(image, 3)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 8.2), dpi=150)
    for axis, (label, view) in zip(axes.flat, views):
        axis.imshow(view)
        axis.set_title(label, loc="left", fontsize=12, fontweight="bold", pad=8)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_linewidth(2)
            spine.set_color("#334155")
    fig.suptitle("E32: six-view HoVer-Net test-time augmentation", fontsize=17, fontweight="bold")
    fig.text(
        0.5,
        0.025,
        "One shared model processes all six views. Predictions are inverse-warped to native coordinates; "
        "NP and type probabilities plus HV vector maps are averaged before a single decoder pass. "
        "HV axes/signs are corrected during inversion.",
        ha="center",
        va="bottom",
        fontsize=10.5,
        wrap=True,
    )
    fig.tight_layout(rect=(0, 0.075, 1, 0.94), h_pad=2.5, w_pad=2.0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
