#!/usr/bin/env python3
"""Summarize learning curves and validation-selected checkpoints across HoVer-Net folds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def best_row(frame: pd.DataFrame, metric: str) -> dict | None:
    valid = frame.loc[np.isfinite(frame[metric])]
    if valid.empty:
        return None
    row = valid.loc[valid[metric].idxmax()]
    return {
        "epoch": int(row.epoch),
        "learning_rate": float(row.learning_rate),
        "value": float(row[metric]),
        "other_metric": float(row["val_R2" if metric == "val_mPQ+" else "val_mPQ+"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold0", type=Path, required=True)
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    args = parser.parse_args()

    run_dirs = [args.fold0] + [args.fold_root / f"fold_{fold}_phase1_lr1e-4" for fold in range(1, 6)]
    frames: dict[int, pd.DataFrame] = {}
    reports = []
    for fold, run_dir in enumerate(run_dirs):
        curve = run_dir / "training_curve.csv"
        if not curve.exists():
            reports.append({"fold": fold, "status": "not_started", "run_dir": str(run_dir)})
            continue
        frame = pd.read_csv(curve)
        frames[fold] = frame
        reports.append(
            {
                "fold": fold,
                "status": "complete" if (run_dir / "summary.json").exists() else "running",
                "run_dir": str(run_dir),
                "epochs_recorded": int(frame.epoch.max()),
                "best_mPQ+": best_row(frame, "val_mPQ+"),
                "best_R2": best_row(frame, "val_R2"),
                "final_train_loss": float(frame.iloc[-1].train_loss),
                "final_val_loss": float(frame.iloc[-1].val_loss),
            }
        )

    completed = [item for item in reports if item["status"] == "complete"]
    report = {
        "folds": reports,
        "completed_folds": len(completed),
        "checkpoint_policy": "Within each fold, best_mPQ and best_R2 are selected independently on that fold's group-disjoint validation partition.",
    }
    if completed:
        for name, key in (("mPQ+", "best_mPQ+"), ("R2", "best_R2")):
            values = [item[key]["value"] for item in completed if item[key] is not None]
            report[f"completed_fold_{name}_mean_std"] = [float(np.mean(values)), float(np.std(values))]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=160, sharex=True)
    colors = plt.get_cmap("tab10")
    for fold, frame in frames.items():
        color = colors(fold)
        suffix = "" if (run_dirs[fold] / "summary.json").exists() else " (running)"
        axes[0, 0].plot(frame.epoch, frame.train_loss, color=color, alpha=0.85, label=f"fold {fold}{suffix}")
        axes[0, 1].plot(frame.epoch, frame.val_loss, color=color, alpha=0.85)
        measured = frame.loc[np.isfinite(frame["val_mPQ+"])]
        axes[1, 0].plot(measured.epoch, measured["val_mPQ+"], marker="o", color=color, alpha=0.9)
        axes[1, 1].plot(measured.epoch, measured.val_R2, marker="o", color=color, alpha=0.9)
    axes[0, 0].set_title("Training loss")
    axes[0, 1].set_title("Validation loss")
    axes[1, 0].set_title("Group-disjoint validation mPQ+")
    axes[1, 1].set_title("Group-disjoint validation macro R²")
    for axis in axes.flat:
        axis.axvline(25.5, color="#555", linestyle="--", linewidth=1, alpha=0.8)
        axis.grid(alpha=0.2)
        axis.set_xlabel("Epoch")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("HoVer-Net six-fold training audit (LR 1e-4 → 1e-5 after epoch 25)")
    fig.tight_layout()
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
