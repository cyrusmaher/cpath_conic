#!/usr/bin/env python3
"""Group-disjoint cross-validation for per-class convex count blending."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from cpath_conic.constants import CLASS_NAMES, COUNT_COLUMNS
from cpath_conic.metrics import multiclass_r2
from scripts.sweep_count_ensemble import blended_counts, class_r2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--counts-a", type=Path, required=True)
    parser.add_argument("--counts-b", type=Path, required=True)
    parser.add_argument("--name-a", default="A")
    parser.add_argument("--name-b", default="B")
    parser.add_argument("--split", default="val")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--alpha-step", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    metadata = pd.read_csv(args.prepared / "metadata.csv").sort_values("patch_id")
    selected = metadata.loc[metadata.split == args.split].sort_values("patch_id").reset_index(drop=True)
    patch_ids = selected.patch_id.to_numpy(dtype=np.int32)
    truth = selected[COUNT_COLUMNS].to_numpy(dtype=np.int32)
    counts_a = np.load(args.counts_a)[patch_ids].astype(np.int32)
    counts_b = np.load(args.counts_b)[patch_ids].astype(np.int32)
    grid = np.arange(0.0, 1.0 + args.alpha_step / 2.0, args.alpha_step)
    grid = grid[(grid >= 0) & (grid <= 1)]
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof = np.zeros_like(truth)
    assignments = np.full(len(selected), -1, dtype=np.int8)
    fold_reports = []

    for fold, (train_rows, heldout_rows) in enumerate(
        splitter.split(np.zeros(len(selected)), selected.source.to_numpy(), groups=selected.source_group.to_numpy()), start=1
    ):
        weights = np.zeros(len(CLASS_NAMES), dtype=np.float64)
        for class_index in range(len(CLASS_NAMES)):
            candidates = []
            for alpha in grid:
                predicted = np.rint(alpha * counts_a[train_rows, class_index] + (1.0 - alpha) * counts_b[train_rows, class_index])
                candidates.append((class_r2(truth[train_rows, class_index], predicted), float(alpha)))
            weights[class_index] = max(candidates, key=lambda item: (item[0], -abs(item[1] - 0.5)))[1]
        oof[heldout_rows] = blended_counts(counts_a[heldout_rows], counts_b[heldout_rows], weights)
        assignments[heldout_rows] = fold
        fold_metrics = multiclass_r2(
            pd.DataFrame(truth[heldout_rows], columns=CLASS_NAMES),
            pd.DataFrame(oof[heldout_rows], columns=CLASS_NAMES),
        )
        fold_reports.append({
            "fold": fold,
            "n_patches": int(len(heldout_rows)),
            "sources": selected.iloc[heldout_rows].source.value_counts().to_dict(),
            "weights_a": dict(zip(CLASS_NAMES, weights.tolist())),
            "heldout": fold_metrics,
        })
    if np.any(assignments < 0):
        raise RuntimeError("not every validation patch received an out-of-fold prediction")

    def metrics(values: np.ndarray) -> dict:
        return multiclass_r2(pd.DataFrame(truth, columns=CLASS_NAMES), pd.DataFrame(values, columns=CLASS_NAMES))

    report = {
        "split": args.split,
        "test_evaluated": False,
        "cross_validation": f"{args.folds}-fold source-stratified GroupKFold over source_group",
        "method_names": {"a": args.name_a, "b": args.name_b},
        "alpha_step": args.alpha_step,
        "out_of_fold": metrics(oof),
        "endpoints": {"a": metrics(counts_a), "b": metrics(counts_b)},
        "folds": fold_reports,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    full_oof = np.zeros((len(metadata), len(CLASS_NAMES)), dtype=np.int32)
    full_oof[patch_ids] = oof
    np.save(args.out.with_suffix(".npy"), full_oof)
    print(json.dumps({"out_of_fold": report["out_of_fold"], "endpoints": report["endpoints"]}, indent=2))


if __name__ == "__main__":
    main()
