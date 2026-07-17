#!/usr/bin/env python3
"""Nested group-CV count stacking for the CoNIC R²-only recipe.

The stacker never changes masks or cell types. It learns one small ridge model
per class from two independently produced patch-count vectors. Candidate
feature families and ridge strength are selected inside each outer fold, so the
reported OOF score includes hyperparameter/recipe selection rather than using
the same validation labels twice.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from cpath_conic.constants import CLASS_NAMES, COUNT_COLUMNS
from scripts.sweep_count_ensemble import class_r2


RECIPES = ("global_linear", "global_quadratic", "source_linear", "source_quadratic")


def design_matrix(
    counts_a: np.ndarray,
    counts_b: np.ndarray,
    sources: np.ndarray,
    source_levels: list[str],
    recipe: str,
) -> np.ndarray:
    """Create label-free count/source features for one cell class."""
    a = np.asarray(counts_a, dtype=np.float64)
    b = np.asarray(counts_b, dtype=np.float64)
    linear = np.column_stack([a, b])
    quadratic = np.column_stack([a, b, a * a, b * b, a * b])
    base = linear if recipe in {"global_linear", "source_linear"} else quadratic
    if recipe.startswith("global_"):
        return base
    dummies = np.column_stack([(sources == level).astype(np.float64) for level in source_levels[1:]])
    interactions = np.concatenate([dummies * a[:, None], dummies * b[:, None]], axis=1)
    return np.concatenate([base, dummies, interactions], axis=1)


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> dict:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1.0e-8] = 1.0
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit((x - mean) / scale, y)
    return {
        "mean": mean,
        "scale": scale,
        "coef": model.coef_.astype(np.float64),
        "intercept": float(model.intercept_),
        "alpha": float(alpha),
    }


def predict_ridge(model: dict, x: np.ndarray) -> np.ndarray:
    raw = (x - model["mean"]) / model["scale"] @ model["coef"] + model["intercept"]
    return np.rint(np.clip(raw, 0.0, None)).astype(np.int32)


def select_candidate(
    counts_a: np.ndarray,
    counts_b: np.ndarray,
    truth: np.ndarray,
    sources: np.ndarray,
    groups: np.ndarray,
    source_levels: list[str],
    alphas: list[float],
    folds: int,
) -> tuple[str, float, list[dict]]:
    """Select feature family and ridge strength by inner group-disjoint OOF R²."""
    splitter = GroupKFold(n_splits=min(folds, len(np.unique(groups))))
    candidates: list[dict] = []
    complexity = {name: index for index, name in enumerate(RECIPES)}
    for recipe in RECIPES:
        x = design_matrix(counts_a, counts_b, sources, source_levels, recipe)
        for alpha in alphas:
            prediction = np.zeros(len(truth), dtype=np.int32)
            assigned = np.zeros(len(truth), dtype=bool)
            fold_scores = []
            for train_rows, heldout_rows in splitter.split(x, truth, groups):
                model = fit_ridge(x[train_rows], truth[train_rows], alpha)
                prediction[heldout_rows] = predict_ridge(model, x[heldout_rows])
                assigned[heldout_rows] = True
                score = class_r2(truth[heldout_rows], prediction[heldout_rows])
                if np.isfinite(score):
                    fold_scores.append(float(score))
            if not assigned.all():
                raise RuntimeError("inner CV did not assign every row")
            candidates.append(
                {
                    "recipe": recipe,
                    "alpha": float(alpha),
                    "oof_R2": class_r2(truth, prediction),
                    "mean_fold_R2": float(np.mean(fold_scores)) if fold_scores else float("nan"),
                    "complexity": complexity[recipe],
                }
            )
    best = max(candidates, key=lambda item: (item["oof_R2"], -item["complexity"], item["alpha"]))
    return str(best["recipe"]), float(best["alpha"]), candidates


def macro_r2(truth: np.ndarray, prediction: np.ndarray) -> dict:
    per_class = {
        name: class_r2(truth[:, index], prediction[:, index])
        for index, name in enumerate(CLASS_NAMES)
    }
    finite = [value for value in per_class.values() if np.isfinite(value)]
    return {"R2": float(np.mean(finite)), "per_class": per_class}


def bootstrap_delta(
    truth: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    replicates: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    observed = macro_r2(truth, candidate)["R2"] - macro_r2(truth, baseline)["R2"]
    values = np.empty(replicates, dtype=np.float64)
    rows = np.arange(len(truth))
    for index in range(replicates):
        sample = rng.choice(rows, size=len(rows), replace=True)
        values[index] = macro_r2(truth[sample], candidate[sample])["R2"] - macro_r2(
            truth[sample], baseline[sample]
        )["R2"]
    return {
        "observed_delta": float(observed),
        "paired_bootstrap_95_ci": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "probability_candidate_better": float(np.mean(values > 0)),
        "replicates": int(replicates),
    }


def serialize_model(model: dict, recipe: str, class_name: str) -> dict:
    return {
        "class": class_name,
        "recipe": recipe,
        "alpha": model["alpha"],
        "mean": model["mean"].tolist(),
        "scale": model["scale"].tolist(),
        "coef": model["coef"].tolist(),
        "intercept": model["intercept"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--counts-a", type=Path, required=True)
    parser.add_argument("--counts-b", type=Path, required=True)
    parser.add_argument("--baseline-oof-counts", type=Path, required=True)
    parser.add_argument("--name-a", default="A")
    parser.add_argument("--name-b", default="B")
    parser.add_argument("--split", default="val")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 1.0, 10.0, 100.0, 1000.0])
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--out", type=Path, required=True, help="JSON report; .npy sibling stores full fitted predictions")
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Report test metrics after the nested-OOF gate. Off by default.",
    )
    args = parser.parse_args()

    metadata = pd.read_csv(args.prepared / "metadata.csv").sort_values("patch_id").reset_index(drop=True)
    counts_a_full = np.load(args.counts_a).astype(np.int32)
    counts_b_full = np.load(args.counts_b).astype(np.int32)
    baseline_full = np.load(args.baseline_oof_counts).astype(np.int32)
    expected = (len(metadata), len(CLASS_NAMES))
    for name, values in (("A", counts_a_full), ("B", counts_b_full), ("baseline", baseline_full)):
        if values.shape != expected:
            raise ValueError(f"{name} counts have shape {values.shape}; expected {expected}")

    selected = metadata.loc[metadata.split == args.split].copy().reset_index(drop=True)
    patch_ids = selected.patch_id.to_numpy(dtype=np.int32)
    truth = selected[COUNT_COLUMNS].to_numpy(dtype=np.int32)
    counts_a = counts_a_full[patch_ids]
    counts_b = counts_b_full[patch_ids]
    baseline_oof = baseline_full[patch_ids]
    sources = selected.source.astype(str).to_numpy()
    groups = selected.source_group.astype(str).to_numpy()
    source_levels = sorted(metadata.source.astype(str).unique())
    outer = StratifiedGroupKFold(n_splits=args.outer_folds, shuffle=True, random_state=args.seed)
    oof = np.zeros_like(truth)
    assignments = np.full(len(selected), -1, dtype=np.int16)
    fold_reports = []

    for fold, (train_rows, heldout_rows) in enumerate(
        outer.split(np.zeros(len(selected)), sources, groups=groups), start=1
    ):
        if set(groups[train_rows]) & set(groups[heldout_rows]):
            raise RuntimeError("outer CV group leakage")
        class_reports = []
        for class_index, class_name in enumerate(CLASS_NAMES):
            recipe, alpha, candidates = select_candidate(
                counts_a[train_rows, class_index],
                counts_b[train_rows, class_index],
                truth[train_rows, class_index],
                sources[train_rows],
                groups[train_rows],
                source_levels,
                args.alphas,
                args.inner_folds,
            )
            x_train = design_matrix(
                counts_a[train_rows, class_index], counts_b[train_rows, class_index], sources[train_rows], source_levels, recipe
            )
            x_heldout = design_matrix(
                counts_a[heldout_rows, class_index], counts_b[heldout_rows, class_index], sources[heldout_rows], source_levels, recipe
            )
            model = fit_ridge(x_train, truth[train_rows, class_index], alpha)
            oof[heldout_rows, class_index] = predict_ridge(model, x_heldout)
            class_reports.append(
                {
                    "class": class_name,
                    "recipe": recipe,
                    "alpha": alpha,
                    "heldout_R2": class_r2(truth[heldout_rows, class_index], oof[heldout_rows, class_index]),
                    "inner_best_R2": max(item["oof_R2"] for item in candidates),
                }
            )
        assignments[heldout_rows] = fold
        fold_reports.append(
            {
                "fold": fold,
                "n_train": int(len(train_rows)),
                "n_heldout": int(len(heldout_rows)),
                "heldout_sources": selected.iloc[heldout_rows].source.value_counts().to_dict(),
                "heldout": macro_r2(truth[heldout_rows], oof[heldout_rows]),
                "classes": class_reports,
            }
        )
    if np.any(assignments < 0):
        raise RuntimeError("outer CV did not assign every row")
    if not np.isfinite(oof).all() or np.any(oof < 0):
        raise RuntimeError("stacker produced invalid counts")

    final_models = []
    full_prediction = np.zeros_like(counts_a_full)
    all_sources = metadata.source.astype(str).to_numpy()
    final_selection = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        recipe, alpha, candidates = select_candidate(
            counts_a[:, class_index],
            counts_b[:, class_index],
            truth[:, class_index],
            sources,
            groups,
            source_levels,
            args.alphas,
            args.inner_folds,
        )
        x = design_matrix(counts_a[:, class_index], counts_b[:, class_index], sources, source_levels, recipe)
        x_all = design_matrix(counts_a_full[:, class_index], counts_b_full[:, class_index], all_sources, source_levels, recipe)
        model = fit_ridge(x, truth[:, class_index], alpha)
        full_prediction[:, class_index] = predict_ridge(model, x_all)
        final_models.append(serialize_model(model, recipe, class_name))
        final_selection.append(
            {
                "class": class_name,
                "recipe": recipe,
                "alpha": alpha,
                "inner_oof_R2": max(item["oof_R2"] for item in candidates),
            }
        )

    report = {
        "selection_split": args.split,
        "test_evaluated": bool(args.evaluate_test),
        "method_names": {"a": args.name_a, "b": args.name_b},
        "protocol": f"{args.outer_folds}-fold outer StratifiedGroupKFold(source, source_group); {args.inner_folds}-fold inner GroupKFold(source_group)",
        "recipes": list(RECIPES),
        "ridge_alphas": args.alphas,
        "out_of_fold": macro_r2(truth, oof),
        "baseline_oof": macro_r2(truth, baseline_oof),
        "endpoints": {"a": macro_r2(truth, counts_a), "b": macro_r2(truth, counts_b)},
        "oof_vs_baseline": bootstrap_delta(truth, baseline_oof, oof, args.replicates, args.seed),
        "folds": fold_reports,
        "final_selection": final_selection,
        "models": final_models,
        "labels_accessed_for_fit": args.split,
    }
    if args.evaluate_test:
        test = metadata.loc[metadata.split == "test"].sort_values("patch_id")
        test_ids = test.patch_id.to_numpy(dtype=np.int32)
        report["test"] = macro_r2(test[COUNT_COLUMNS].to_numpy(dtype=np.int32), full_prediction[test_ids])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=True))
    np.save(args.out.with_suffix(".npy"), full_prediction)
    oof_full = np.zeros_like(full_prediction)
    oof_full[patch_ids] = oof
    np.save(args.out.with_name(args.out.stem + "_oof.npy"), oof_full)
    print(json.dumps({key: report[key] for key in ("out_of_fold", "baseline_oof", "oof_vs_baseline", "final_selection")}, indent=2))


if __name__ == "__main__":
    main()
