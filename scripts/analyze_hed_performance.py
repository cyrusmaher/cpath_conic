#!/usr/bin/env python3
"""Stratify CoNIC validation performance by training-defined H/E bins.

The analysis is deliberately diagnostic.  A stain bin can correlate with source,
tissue morphology, and count range, so the selected anchor must still pass a
paired stain-TTA validation experiment before it is promoted.
"""
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

from cpath_conic.constants import CLASS_NAMES, COUNT_COLUMNS
from cpath_conic.metrics import multiclass_pq_plus, multiclass_r2


def assign_quantile_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Return zero-based bins, placing values equal to an edge in the upper bin."""
    return np.searchsorted(np.asarray(edges), np.asarray(values), side="right").astype(np.int8)


def load_true_maps(prepared: Path, patch_ids: np.ndarray) -> np.ndarray:
    maps = []
    for patch_id in patch_ids:
        label = np.load(
            prepared / "labels" / f"{int(patch_id):05d}.npy", allow_pickle=True
        ).item()
        maps.append(np.stack([label["inst_map"], label["class_map"]], axis=-1))
    return np.asarray(maps, dtype=np.int32)


def source_composition(sources: pd.Series) -> dict[str, dict[str, float | int]]:
    counts = sources.astype(str).value_counts().sort_index()
    return {
        str(source): {"n": int(count), "fraction": float(count / len(sources))}
        for source, count in counts.items()
    }


def select_anchor(
    train_patch_ids: np.ndarray,
    train_concentrations: np.ndarray,
    h_bins: np.ndarray,
    e_bins: np.ndarray,
    h_bin: int,
    e_bin: int,
) -> dict:
    mask = (h_bins == h_bin) & (e_bins == e_bin)
    values = train_concentrations[mask]
    patch_ids = train_patch_ids[mask]
    if not len(values):
        raise ValueError(f"No training anchor candidates in H{h_bin + 1}/E{e_bin + 1}")
    log_values = np.log(np.maximum(values, 1e-8))
    center = np.median(log_values, axis=0)
    chosen = int(np.argmin(np.linalg.norm(log_values - center, axis=1)))
    return {
        "patch_id": int(patch_ids[chosen]),
        "concentration": [float(value) for value in values[chosen]],
        "bin_log_median": [float(value) for value in center],
        "training_support": int(mask.sum()),
    }


def plot_heatmaps(rows: list[dict], n_bins: int, output: Path) -> None:
    fields = [
        ("mPQ+", "Validation mPQ+", "viridis", None),
        ("R2", "Validation macro R²", "viridis", None),
        ("mean_signed_error", "Mean signed error (pred − GT)", "coolwarm", 0.0),
        ("n_patches", "Validation support", "magma", None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=160)
    for axis, (field, title, cmap, center) in zip(axes.flat, fields):
        matrix = np.full((n_bins, n_bins), np.nan, dtype=float)
        for row in rows:
            matrix[row["e_bin"] - 1, row["h_bin"] - 1] = row[field]
        kwargs = {}
        if center is not None:
            bound = float(np.nanmax(np.abs(matrix)))
            kwargs.update(vmin=-bound, vmax=bound)
        image = axis.imshow(matrix, origin="lower", cmap=cmap, aspect="equal", **kwargs)
        for e_index in range(n_bins):
            for h_index in range(n_bins):
                value = matrix[e_index, h_index]
                if np.isfinite(value):
                    text = f"{int(value)}" if field == "n_patches" else f"{value:.3f}"
                    axis.text(h_index, e_index, text, ha="center", va="center", color="white", fontsize=9)
        axis.set_title(title)
        axis.set_xlabel("H concentration quantile bin")
        axis.set_ylabel("E concentration quantile bin")
        axis.set_xticks(range(n_bins), [f"H{i + 1}" for i in range(n_bins)])
        axis.set_yticks(range(n_bins), [f"E{i + 1}" for i in range(n_bins)])
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle("HoVer-Net validation performance across training-defined stain space")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--bins", type=int, default=3)
    parser.add_argument("--min-bin-size", type=int, default=40)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    args = parser.parse_args()

    if args.bins < 2:
        raise ValueError("--bins must be at least 2")
    metadata = pd.read_csv(args.prepared / "metadata.csv").sort_values("patch_id")
    selected = metadata.loc[metadata.split == args.split].sort_values("patch_id").copy()
    if selected.empty:
        raise ValueError(f"No patches in split {args.split!r}")

    profile = np.load(args.profile, allow_pickle=False)
    profiled = pd.DataFrame(
        {
            "patch_id": profile["patch_ids"].astype(np.int32),
            "H": profile["concentrations"][:, 0],
            "E": profile["concentrations"][:, 1],
            "profile_source": profile["sources"].astype(str),
            "profile_split": profile["splits"].astype(str),
        }
    )
    train_profile = profiled.loc[profiled.profile_split == "train"].copy()
    selected = selected.merge(
        profiled[["patch_id", "H", "E"]], on="patch_id", validate="one_to_one"
    )
    if len(selected) != int((metadata.split == args.split).sum()):
        raise ValueError("The concentration profile does not cover every selected patch")

    quantiles = np.arange(1, args.bins) / args.bins
    h_edges = np.quantile(train_profile.H, quantiles)
    e_edges = np.quantile(train_profile.E, quantiles)
    selected["h_bin"] = assign_quantile_bins(selected.H.to_numpy(), h_edges)
    selected["e_bin"] = assign_quantile_bins(selected.E.to_numpy(), e_edges)
    train_h_bins = assign_quantile_bins(train_profile.H.to_numpy(), h_edges)
    train_e_bins = assign_quantile_bins(train_profile.E.to_numpy(), e_edges)

    predictions = np.load(args.predictions, mmap_mode="r")
    all_counts = np.load(args.counts, mmap_mode="r")
    if predictions.shape[0] != len(metadata) or all_counts.shape != (len(metadata), len(CLASS_NAMES)):
        raise ValueError("Predictions and counts must cover the full prepared dataset")

    true_counts_all = selected[COUNT_COLUMNS].to_numpy(dtype=np.float64)
    pred_counts_all = np.asarray(all_counts[selected.patch_id.to_numpy()], dtype=np.float64)
    residual_all = pred_counts_all - true_counts_all
    global_sst = np.square(true_counts_all - true_counts_all.mean(axis=0, keepdims=True)).sum(axis=0)
    rows = []
    for e_bin in range(args.bins):
        for h_bin in range(args.bins):
            bin_rows = selected.loc[(selected.h_bin == h_bin) & (selected.e_bin == e_bin)]
            if bin_rows.empty:
                continue
            indexes = selected.index.get_indexer(bin_rows.index)
            patch_ids = bin_rows.patch_id.to_numpy(dtype=np.int32)
            truth = load_true_maps(args.prepared, patch_ids)
            predicted = np.asarray(predictions[patch_ids], dtype=np.int32)
            pq = multiclass_pq_plus(truth, predicted)
            true_counts = true_counts_all[indexes]
            pred_counts = pred_counts_all[indexes]
            residual = residual_all[indexes]
            r2 = multiclass_r2(
                pd.DataFrame(true_counts, columns=CLASS_NAMES),
                pd.DataFrame(pred_counts, columns=CLASS_NAMES),
            )
            normalized_sse = np.divide(
                np.square(residual),
                global_sst[None, :],
                out=np.zeros_like(residual),
                where=global_sst[None, :] > 0,
            )
            anchor = select_anchor(
                train_profile.patch_id.to_numpy(dtype=np.int32),
                train_profile[["H", "E"]].to_numpy(dtype=np.float64),
                train_h_bins,
                train_e_bins,
                h_bin,
                e_bin,
            )
            rows.append(
                {
                    "bin": f"H{h_bin + 1}/E{e_bin + 1}",
                    "h_bin": h_bin + 1,
                    "e_bin": e_bin + 1,
                    "n_patches": int(len(bin_rows)),
                    "H_median": float(bin_rows.H.median()),
                    "E_median": float(bin_rows.E.median()),
                    "source_composition": source_composition(bin_rows.source),
                    "mPQ+": float(pq["mPQ+"]),
                    "per_class_pq": pq["per_class"],
                    "R2": float(r2["R2"]),
                    "per_class_R2": r2["per_class"],
                    "mean_signed_error": float(residual.mean()),
                    "mean_absolute_error": float(np.abs(residual).mean()),
                    "under_fraction": float(np.mean(residual < 0)),
                    "over_fraction": float(np.mean(residual > 0)),
                    "outlier_fraction": {
                        str(threshold): float(np.mean(np.abs(residual) > threshold))
                        for threshold in (2, 5, 10, 20)
                    },
                    "global_normalized_sse": float(normalized_sse.sum()),
                    "global_normalized_sse_per_patch": float(normalized_sse.sum() / len(bin_rows)),
                    "training_anchor": anchor,
                }
            )

    eligible = [row for row in rows if row["n_patches"] >= args.min_bin_size]
    if not eligible:
        raise ValueError(f"No joint bins have at least {args.min_bin_size} validation patches")
    best_mpq = max(eligible, key=lambda row: row["mPQ+"])
    finite_r2 = [row for row in eligible if np.isfinite(row["R2"])]
    best_r2 = max(finite_r2, key=lambda row: row["R2"])
    lowest_error = min(eligible, key=lambda row: row["global_normalized_sse_per_patch"])

    overall_r2 = multiclass_r2(
        pd.DataFrame(true_counts_all, columns=CLASS_NAMES),
        pd.DataFrame(pred_counts_all, columns=CLASS_NAMES),
    )
    report = {
        "split": args.split,
        "n_patches": int(len(selected)),
        "model_predictions": str(args.predictions),
        "model_counts": str(args.counts),
        "bin_definition": {
            "fit_split": "train",
            "n_bins_per_axis": args.bins,
            "quantiles": [float(value) for value in quantiles],
            "H_edges": [float(value) for value in h_edges],
            "E_edges": [float(value) for value in e_edges],
            "minimum_selection_support": args.min_bin_size,
        },
        "overall_R2": overall_r2,
        "bins": rows,
        "candidate_anchors": {
            "best_mPQ+": {"bin": best_mpq["bin"], **best_mpq["training_anchor"]},
            "best_within_bin_R2": {"bin": best_r2["bin"], **best_r2["training_anchor"]},
            "lowest_global_normalized_sse_per_patch": {
                "bin": lowest_error["bin"],
                **lowest_error["training_anchor"],
            },
        },
        "interpretation_note": (
            "Bins and anchors use training pixels only; validation labels select among them. "
            "Within-bin R2 depends on the conditional count range, while normalized SSE uses "
            "the full validation class SST. Source composition is reported because stain, tissue, "
            "and institution are confounded. A paired two-view validation run is required."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=True))
    plot_heatmaps(rows, args.bins, args.plot)
    print(
        json.dumps(
            {
                "overall_R2": overall_r2["R2"],
                "candidate_anchors": report["candidate_anchors"],
                "bins": [
                    {
                        key: row[key]
                        for key in (
                            "bin",
                            "n_patches",
                            "mPQ+",
                            "R2",
                            "mean_signed_error",
                            "global_normalized_sse_per_patch",
                        )
                    }
                    for row in rows
                ],
            },
            indent=2,
            allow_nan=True,
        )
    )


if __name__ == "__main__":
    main()
