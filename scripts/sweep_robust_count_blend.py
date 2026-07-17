#!/usr/bin/env python3
"""Validation-select a mask-supported, asymmetrically capped count blend."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cpath_conic.constants import CLASS_NAMES, COUNT_COLUMNS
from scripts.analyze_hovernet_count_complementarity import (
    directional_error_summary,
    r2_score_1d,
    r2_summary,
    summarize_by_source,
    zero_truth_overcount_summary,
)


def robust_blend(
    first: np.ndarray,
    second: np.ndarray,
    first_weight: np.ndarray | float,
    positive_scale: np.ndarray | float,
    negative_scale: np.ndarray | float,
) -> np.ndarray:
    """Blend counts while bounding additions/removals in sqrt-count units.

    ``first`` is the mask-derived anchor. The ordinary convex correction is
    ``(1 - first_weight) * (second - first)``. Positive and negative
    corrections are capped separately so an auxiliary count stream cannot add
    many cells without corresponding mask support.
    """
    anchor = np.asarray(first, dtype=np.float64)
    auxiliary = np.asarray(second, dtype=np.float64)
    first_weight = np.asarray(first_weight, dtype=np.float64)
    positive_scale = np.asarray(positive_scale, dtype=np.float64)
    negative_scale = np.asarray(negative_scale, dtype=np.float64)
    if anchor.shape != auxiliary.shape:
        raise ValueError("count endpoints must have identical shapes")
    if np.any(anchor < 0) or np.any(auxiliary < 0):
        raise ValueError("count endpoints must be nonnegative")
    correction = (1.0 - first_weight) * (auxiliary - anchor)
    root_scale = np.sqrt(anchor + 1.0)
    lower = -negative_scale * root_scale
    upper = positive_scale * root_scale
    return np.rint(anchor + np.minimum(np.maximum(correction, lower), upper)).astype(np.int32)


def select_first_weight(
    truth: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    grid: np.ndarray,
) -> float:
    candidates = []
    for weight in grid:
        prediction = np.rint(weight * first + (1.0 - weight) * second)
        score = r2_score_1d(truth, prediction)
        candidates.append((score if np.isfinite(score) else -np.inf, -abs(weight - 0.5), weight))
    return float(max(candidates)[-1])


def true_zero_tail_gate(
    truth: np.ndarray,
    anchor: np.ndarray,
    candidate: np.ndarray,
    sources: np.ndarray,
    *,
    minimum_support: int = 20,
) -> dict:
    violations = []
    checked = []
    scopes = [("pooled", truth == 0)] + [
        (str(source), (sources == source) & (truth == 0))
        for source in sorted(np.unique(sources))
    ]
    for scope, mask in scopes:
        support = int(mask.sum())
        if support < minimum_support:
            continue
        row = {"scope": scope, "support": support, "thresholds": {}}
        for threshold, base_tolerance in ((10, 0.02), (20, 0.01)):
            anchor_rate = float((anchor[mask] > threshold).mean())
            candidate_rate = float((candidate[mask] > threshold).mean())
            delta = candidate_rate - anchor_rate
            tolerance = max(base_tolerance, 2.0 / support)
            threshold_row = {
                "anchor_fraction": anchor_rate,
                "candidate_fraction": candidate_rate,
                "delta": delta,
                "tolerance": tolerance,
                "passes": bool(delta <= tolerance + 1e-12),
            }
            row["thresholds"][str(threshold)] = threshold_row
            if not threshold_row["passes"]:
                violations.append({"scope": scope, "support": support, "threshold": threshold, **threshold_row})
        checked.append(row)
    return {"passes": not violations, "checked": checked, "violations": violations}


def select_cap_scales(
    truth: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    sources: np.ndarray,
    first_weight: float,
    scale_grid: np.ndarray,
) -> tuple[float, float, list[dict]]:
    candidates = []
    for positive_scale in scale_grid:
        for negative_scale in scale_grid:
            prediction = robust_blend(first, second, first_weight, positive_scale, negative_scale)
            score = r2_score_1d(truth, prediction)
            gate = true_zero_tail_gate(truth, first, prediction, sources)
            candidates.append(
                {
                    "positive_scale": float(positive_scale),
                    "negative_scale": float(negative_scale),
                    "R2": float(score),
                    "tail_gate_passes": gate["passes"],
                    "tail_gate_violations": len(gate["violations"]),
                }
            )
    passing = [row for row in candidates if row["tail_gate_passes"] and np.isfinite(row["R2"])]
    if not passing:
        raise RuntimeError("no cap-scale candidate passed the true-zero tail gate")

    def selection_key(row: dict) -> tuple[float, float, float]:
        # Prefer fewer unsupported additions, then the simpler smaller removal
        # cap, only when validation R2 ties exactly.
        return (row["R2"], -row["positive_scale"], -row["negative_scale"])

    selected = max(passing, key=selection_key)
    ranked = sorted(candidates, key=selection_key, reverse=True)
    return selected["positive_scale"], selected["negative_scale"], ranked[:20]


def fit_recipe(
    truth: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    sources: np.ndarray,
    weight_grid: np.ndarray,
    scale_grid: np.ndarray,
) -> dict:
    first_weights = []
    positive_scales = []
    negative_scales = []
    class_search = {}
    for index, name in enumerate(CLASS_NAMES):
        weight = select_first_weight(truth[:, index], first[:, index], second[:, index], weight_grid)
        positive, negative, ranked = select_cap_scales(
            truth[:, index], first[:, index], second[:, index], sources, weight, scale_grid
        )
        first_weights.append(weight)
        positive_scales.append(positive)
        negative_scales.append(negative)
        class_search[name] = {"top_candidates": ranked}
    return {
        "first_weights": np.asarray(first_weights, dtype=np.float64),
        "positive_scales": np.asarray(positive_scales, dtype=np.float64),
        "negative_scales": np.asarray(negative_scales, dtype=np.float64),
        "class_search": class_search,
    }


def apply_recipe(first: np.ndarray, second: np.ndarray, recipe: dict) -> np.ndarray:
    return robust_blend(
        first,
        second,
        recipe["first_weights"],
        recipe["positive_scales"],
        recipe["negative_scales"],
    )


def summarize(truth: np.ndarray, prediction: np.ndarray, sources: np.ndarray) -> dict:
    return {
        **r2_summary(truth, prediction),
        "count_error": directional_error_summary(truth, prediction),
        "zero_truth_overcount": zero_truth_overcount_summary(truth, prediction),
        "by_source": summarize_by_source(truth, prediction, sources),
    }


def serializable_recipe(recipe: dict) -> dict:
    return {
        "first_weights": dict(zip(CLASS_NAMES, recipe["first_weights"].tolist())),
        "positive_sqrt_count_caps": dict(zip(CLASS_NAMES, recipe["positive_scales"].tolist())),
        "negative_sqrt_count_caps": dict(zip(CLASS_NAMES, recipe["negative_scales"].tolist())),
        "class_search": recipe["class_search"],
    }


def paired_group_bootstrap_r2(
    truth: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    sources: np.ndarray,
    groups: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict:
    """Resample source groups within source and compare paired macro R2."""
    rng = np.random.default_rng(seed)
    group_rows = {
        str(source): [np.flatnonzero((sources == source) & (groups == group)) for group in sorted(np.unique(groups[sources == source]))]
        for source in sorted(np.unique(sources))
    }
    deltas = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled = []
        for rows in group_rows.values():
            sampled.extend(rows[index] for index in rng.integers(0, len(rows), size=len(rows)))
        indices = np.concatenate(sampled)
        deltas[draw] = (
            r2_summary(truth[indices], candidate[indices])["R2"]
            - r2_summary(truth[indices], reference[indices])["R2"]
        )
    return {
        "draws": draws,
        "resampling_unit": "source_group sampled with replacement within source",
        "point_delta_R2": float(r2_summary(truth, candidate)["R2"] - r2_summary(truth, reference)["R2"]),
        "paired_95_CI": np.quantile(deltas, [0.025, 0.975]).tolist(),
        "bootstrap_median": float(np.median(deltas)),
        "fraction_positive": float((deltas > 0).mean()),
        "fraction_above_practical_0p003": float((deltas > 0.003).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--counts-first", type=Path, required=True)
    parser.add_argument("--counts-second", type=Path, required=True)
    parser.add_argument("--name-first", default="mask-derived anchor")
    parser.add_argument("--name-second", default="auxiliary count stream")
    parser.add_argument("--split", default="val")
    parser.add_argument("--alpha-step", type=float, default=0.05)
    parser.add_argument(
        "--scale-grid", type=float, nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 16.0, float("inf")],
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.split.lower() == "test":
        parser.error("robust count-blend selection may not use the locked test split")
    weight_grid = np.arange(0.0, 1.0 + args.alpha_step / 2.0, args.alpha_step)
    weight_grid = weight_grid[(weight_grid >= 0) & (weight_grid <= 1)]
    scale_grid = np.asarray(sorted(set(args.scale_grid)), dtype=np.float64)
    if np.any(np.isnan(scale_grid)) or np.any(scale_grid < 0):
        parser.error("--scale-grid values must be nonnegative and not NaN")

    metadata = pd.read_csv(args.prepared / "metadata.csv").sort_values("patch_id")
    selected = metadata.loc[metadata.split == args.split].sort_values("patch_id").reset_index(drop=True)
    patch_ids = selected.patch_id.to_numpy(dtype=np.int32)
    truth = selected[COUNT_COLUMNS].to_numpy(dtype=np.float64)
    sources = selected.source.astype(str).str.lower().to_numpy()
    first_full = np.load(args.counts_first)
    second_full = np.load(args.counts_second)
    if first_full.shape != second_full.shape or first_full.ndim != 2 or first_full.shape[1] != len(CLASS_NAMES):
        raise ValueError("count files must be matching full-dataset arrays with six columns")
    first = first_full[patch_ids].astype(np.float64)
    second = second_full[patch_ids].astype(np.float64)

    recipe = fit_recipe(truth, first, second, sources, weight_grid, scale_grid)
    robust = apply_recipe(first, second, recipe)
    ordinary = np.rint(
        recipe["first_weights"] * first + (1.0 - recipe["first_weights"]) * second
    ).astype(np.int32)

    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    robust_oof = np.zeros_like(robust)
    ordinary_oof = np.zeros_like(robust)
    fold_reports = []
    for fold, (train_rows, held_rows) in enumerate(
        splitter.split(np.zeros(len(selected)), sources, groups=selected.source_group.to_numpy()), start=1
    ):
        fold_recipe = fit_recipe(
            truth[train_rows], first[train_rows], second[train_rows], sources[train_rows], weight_grid, scale_grid
        )
        robust_oof[held_rows] = apply_recipe(first[held_rows], second[held_rows], fold_recipe)
        ordinary_oof[held_rows] = np.rint(
            fold_recipe["first_weights"] * first[held_rows]
            + (1.0 - fold_recipe["first_weights"]) * second[held_rows]
        ).astype(np.int32)
        fold_reports.append(
            {
                "fold": fold,
                "patches": int(len(held_rows)),
                "held_sources": selected.iloc[held_rows].source.value_counts().sort_index().to_dict(),
                "recipe": serializable_recipe(fold_recipe),
                "ordinary_blend": summarize(truth[held_rows], ordinary_oof[held_rows], sources[held_rows]),
                "robust_blend": summarize(truth[held_rows], robust_oof[held_rows], sources[held_rows]),
            }
        )

    source_oos_robust = np.zeros_like(robust)
    source_oos_ordinary = np.zeros_like(robust)
    source_reports = []
    for held_source in sorted(np.unique(sources)):
        held_rows = np.flatnonzero(sources == held_source)
        train_rows = np.flatnonzero(sources != held_source)
        source_recipe = fit_recipe(
            truth[train_rows], first[train_rows], second[train_rows], sources[train_rows], weight_grid, scale_grid
        )
        source_oos_robust[held_rows] = apply_recipe(first[held_rows], second[held_rows], source_recipe)
        source_oos_ordinary[held_rows] = np.rint(
            source_recipe["first_weights"] * first[held_rows]
            + (1.0 - source_recipe["first_weights"]) * second[held_rows]
        ).astype(np.int32)
        source_reports.append(
            {
                "held_source": str(held_source),
                "patches": int(len(held_rows)),
                "recipe": serializable_recipe(source_recipe),
                "ordinary_blend": summarize(
                    truth[held_rows], source_oos_ordinary[held_rows], sources[held_rows]
                ),
                "robust_blend": summarize(
                    truth[held_rows], source_oos_robust[held_rows], sources[held_rows]
                ),
            }
        )

    report = {
        "protocol": (
            "Development-validation-only two-stage selection. Each class first selects its ordinary convex "
            "count weight, then separately caps positive and negative auxiliary corrections in sqrt(anchor+1) "
            "units. Supported source/class truth-zero >10/>20 tails must remain within max(base tolerance, "
            "two patches/support) of the mask-derived anchor. Five-fold source-stratified GroupKFold repeats "
            "both stages inside each training fold. No test labels or predictions are evaluated."
        ),
        "evaluation_set": f"{len(selected)}-patch source-group-disjoint development {args.split}",
        "test_evaluated": False,
        "endpoints": {args.name_first: summarize(truth, first, sources), args.name_second: summarize(truth, second, sources)},
        "full_validation": {
            "recipe": serializable_recipe(recipe),
            "ordinary_blend": summarize(truth, ordinary, sources),
            "robust_blend": summarize(truth, robust, sources),
            "robust_true_zero_gate_vs_first": {
                name: true_zero_tail_gate(truth[:, index], first[:, index], robust[:, index], sources)
                for index, name in enumerate(CLASS_NAMES)
            },
        },
        "nested_group_cv": {
            "splitter": f"{args.folds}-fold source-stratified GroupKFold over source_group",
            "ordinary_blend": summarize(truth, ordinary_oof, sources),
            "robust_blend": summarize(truth, robust_oof, sources),
            "paired_group_bootstrap": paired_group_bootstrap_r2(
                truth,
                ordinary_oof,
                robust_oof,
                sources,
                selected.source_group.astype(str).to_numpy(),
                draws=args.bootstrap_draws,
                seed=args.seed + 1,
            ),
            "folds": fold_reports,
        },
        "leave_one_source_out": {
            "interpretation": (
                "Both convex weights and cap scales are selected on all other institutions, then applied "
                "to the held institution. This is stricter than the deployment setting, where the same "
                "institution labels exist in development and test, but detects source-local cap gains."
            ),
            "ordinary_blend": summarize(truth, source_oos_ordinary, sources),
            "robust_blend": summarize(truth, source_oos_robust, sources),
            "held_sources": source_reports,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=True))
    full_robust = np.zeros_like(first_full, dtype=np.int32)
    full_ordinary_oof = np.zeros_like(first_full, dtype=np.int32)
    full_robust_oof = np.zeros_like(first_full, dtype=np.int32)
    full_source_oos = np.zeros_like(first_full, dtype=np.int32)
    full_robust[patch_ids] = robust
    full_ordinary_oof[patch_ids] = ordinary_oof
    full_robust_oof[patch_ids] = robust_oof
    full_source_oos[patch_ids] = source_oos_robust
    np.save(args.out.with_suffix(".npy"), full_robust)
    np.save(args.out.with_name(args.out.stem + "_ordinary_oof.npy"), full_ordinary_oof)
    np.save(args.out.with_name(args.out.stem + "_robust_oof.npy"), full_robust_oof)
    np.save(args.out.with_name(args.out.stem + "_source_oos.npy"), full_source_oos)
    print(
        json.dumps(
            {
                "full_validation_R2": report["full_validation"]["robust_blend"]["R2"],
                "ordinary_oof_R2": report["nested_group_cv"]["ordinary_blend"]["R2"],
                "robust_oof_R2": report["nested_group_cv"]["robust_blend"]["R2"],
                "robust_leave_one_source_out_R2": report["leave_one_source_out"]["robust_blend"]["R2"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
