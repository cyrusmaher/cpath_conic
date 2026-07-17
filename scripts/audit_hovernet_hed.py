#!/usr/bin/env python3
"""Quantitatively sanity-check the exact HoVer-Net HED training transform."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_hovernet_our_split import EmpiricalHEDTargetBank, hed_concentration, hed_stain_augmentation_array


def image_diagnostics(original: np.ndarray, augmented: np.ndarray) -> dict[str, float | bool]:
    original_float = original.astype(np.float32) / 255.0
    augmented_float = augmented.astype(np.float32) / 255.0
    original_luminance = original_float.mean(axis=-1)
    augmented_luminance = augmented_float.mean(axis=-1)
    green = (augmented_float[..., 1] > augmented_float[..., 0] + 0.08) & (
        augmented_float[..., 1] > augmented_float[..., 2] + 0.08
    )
    original_std = float(original_luminance.std())
    return {
        "mean_absolute_rgb_delta": float(np.abs(augmented_float - original_float).mean()),
        "mean_luminance_delta": float(augmented_luminance.mean() - original_luminance.mean()),
        "contrast_ratio": float(augmented_luminance.std() / max(original_std, 1.0e-8)),
        "newly_white_fraction": float(
            (augmented_luminance > 0.95).mean() - (original_luminance > 0.95).mean()
        ),
        "green_fraction": float(green.mean()),
        "unchanged": bool(np.array_equal(original, augmented)),
    }


def summarize(rows: list[dict]) -> dict:
    numeric = [key for key, value in rows[0].items() if isinstance(value, float)]
    result: dict[str, object] = {"n": len(rows)}
    for key in numeric:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        result[key] = {
            "minimum": float(values.min()),
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)),
            "maximum": float(values.max()),
        }
    result["unchanged_fraction"] = float(np.mean([row["unchanged"] for row in rows]))
    return result


def nearest_source(values: np.ndarray, centroids: dict[str, np.ndarray], scale: np.ndarray) -> np.ndarray:
    names = np.asarray(sorted(centroids))
    centers = np.stack([centroids[name] for name in names])
    distances = np.square((values[:, None, :] - centers[None, :, :]) / scale[None, None, :]).sum(axis=2)
    return names[np.argmin(distances, axis=1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--patches-per-source", type=int, default=20)
    parser.add_argument("--variants", type=int, default=5)
    parser.add_argument("--target-jitter", type=float, default=0.05)
    parser.add_argument("--tail-expansion", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()

    metadata = pd.read_csv(args.prepared / "metadata.csv")
    with np.load(args.profile) as profile:
        concentrations = profile["concentrations"].astype(np.float32)
        target_sources = profile["sources"].astype(str)
        splits = profile["splits"].astype(str)
    bank = EmpiricalHEDTargetBank(
        concentrations[splits == "train"],
        target_sources[splits == "train"],
        jitter=args.target_jitter,
        tail_expansion=args.tail_expansion,
    )
    train_log = np.log(concentrations[splits == "train"])
    train_sources = target_sources[splits == "train"]
    centroid_scale = np.maximum(train_log.std(axis=0), 1.0e-6)
    centroids = {
        source: np.median(train_log[train_sources == source], axis=0)
        for source in sorted(np.unique(train_sources))
    }
    selected = (
        metadata.loc[metadata.split.eq("train")]
        .sort_values(["source", "patch_id"])
        .groupby("source", sort=True)
        .head(args.patches_per_source)
    )
    rows: list[dict] = []
    for patch_index, row in enumerate(selected.itertuples(index=False)):
        original = np.asarray(
            Image.open(args.prepared / "images" / f"{int(row.patch_id):05d}.png").convert("RGB"),
            dtype=np.uint8,
        )
        original_concentration = hed_concentration(original)
        for variant in range(args.variants):
            rng = np.random.default_rng(args.seed + 1009 * patch_index + variant)
            target, target_source = bank.sample(rng)
            augmented = hed_stain_augmentation_array(
                original,
                rng,
                probability=1.0,
                target_concentration=target,
            )
            achieved = hed_concentration(augmented)
            before_distance = float(np.linalg.norm(np.log(original_concentration) - np.log(target)))
            after_distance = float(np.linalg.norm(np.log(achieved) - np.log(target)))
            rows.append(
                {
                    "patch_id": int(row.patch_id),
                    "source": str(row.source),
                    "variant": variant,
                    "target_source": target_source,
                    "target_H": float(target[0]),
                    "target_E": float(target[1]),
                    "original_H": float(original_concentration[0]),
                    "original_E": float(original_concentration[1]),
                    "achieved_H": float(achieved[0]),
                    "achieved_E": float(achieved[1]),
                    "target_log_distance_before": before_distance,
                    "target_log_distance_after": after_distance,
                    "moved_toward_target": bool(after_distance < before_distance),
                    **image_diagnostics(original, augmented),
                }
            )

    by_source = {
        str(source): summarize([row for row in rows if row["source"] == source])
        for source in sorted({row["source"] for row in rows})
    }
    report = {
        "transform": "scripts.train_hovernet_our_split.hed_stain_augmentation_array",
        "target_distribution": "observed joint training-patch H/E pairs, source-balanced",
        "target_jitter": args.target_jitter,
        "tail_expansion": args.tail_expansion,
        "patches_per_source": args.patches_per_source,
        "variants_per_patch": args.variants,
        "overall": summarize(rows),
        "by_source": by_source,
        "safety_limits": {
            "minimum_contrast_ratio": 0.5,
            "maximum_absolute_luminance_shift": 0.2,
            "maximum_newly_white_fraction": 0.25,
            "maximum_green_fraction": 0.01,
        },
    }
    original_log = np.log(np.asarray([[row["original_H"], row["original_E"]] for row in rows]))
    achieved_log = np.log(np.asarray([[row["achieved_H"], row["achieved_E"]] for row in rows]))
    input_sources = np.asarray([row["source"] for row in rows])
    sampled_sources = np.asarray([row["target_source"] for row in rows])
    original_prediction = nearest_source(original_log, centroids, centroid_scale)
    achieved_prediction = nearest_source(achieved_log, centroids, centroid_scale)
    report["mechanism"] = {
        "moved_toward_sampled_target_fraction": float(np.mean([row["moved_toward_target"] for row in rows])),
        "median_target_log_distance_before": float(np.median([row["target_log_distance_before"] for row in rows])),
        "median_target_log_distance_after": float(np.median([row["target_log_distance_after"] for row in rows])),
        "nearest_centroid_input_source_accuracy_before": float(np.mean(original_prediction == input_sources)),
        "nearest_centroid_input_source_accuracy_after": float(np.mean(achieved_prediction == input_sources)),
        "nearest_centroid_sampled_target_source_accuracy_after": float(np.mean(achieved_prediction == sampled_sources)),
        "interpretation": "A useful stain augmentation moves toward observed targets and lowers recoverability of the input institution from H/E concentration alone.",
    }
    overall = report["overall"]
    report["passed"] = bool(
        overall["contrast_ratio"]["minimum"] >= 0.5
        and max(
            abs(overall["mean_luminance_delta"]["minimum"]),
            abs(overall["mean_luminance_delta"]["maximum"]),
        ) <= 0.2
        and overall["newly_white_fraction"]["maximum"] <= 0.25
        and overall["green_fraction"]["maximum"] <= 0.01
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"passed": report["passed"], "overall": report["overall"]}, indent=2))


if __name__ == "__main__":
    main()
