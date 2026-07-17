#!/usr/bin/env python3
"""Create development folds while preserving the original validation fold as fold 0."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def balanced_source_group_folds(rows: pd.DataFrame, folds: int, seed: int) -> list[np.ndarray]:
    """Allocate whole groups within each source, balancing patch counts per fold."""
    rng = np.random.default_rng(seed)
    assigned: list[list[int]] = [[] for _ in range(folds)]
    for source in sorted(rows.source.unique()):
        source_rows = rows.loc[rows.source == source]
        group_sizes = source_rows.groupby("source_group").size()
        if len(group_sizes) < folds:
            raise ValueError(f"source {source} has only {len(group_sizes)} groups for {folds} folds")
        groups = group_sizes.index.to_numpy()
        groups = groups[rng.permutation(len(groups))]
        groups = sorted(groups, key=lambda group: int(group_sizes[group]), reverse=True)
        source_load = np.zeros(folds, dtype=np.int64)
        for group in groups:
            lightest = np.flatnonzero(source_load == source_load.min())
            fold = int(rng.choice(lightest))
            patch_ids = source_rows.loc[source_rows.source_group == group, "patch_id"].astype(int).tolist()
            assigned[fold].extend(patch_ids)
            source_load[fold] += len(patch_ids)
    return [np.asarray(sorted(values), dtype=np.int32) for values in assigned]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()
    if args.folds < 3:
        raise ValueError("at least three folds are required")

    metadata = pd.read_csv(args.prepared / "metadata.csv").sort_values("patch_id")
    original_train = metadata.loc[metadata.split == "train"].copy()
    original_val = metadata.loc[metadata.split == "val"].copy()
    development = metadata.loc[metadata.split.isin(["train", "val"])].copy()
    test_ids = set(metadata.loc[metadata.split == "test", "patch_id"].astype(int))
    validation_folds: list[np.ndarray] = [original_val.patch_id.to_numpy(dtype=np.int32)]
    validation_folds.extend(balanced_source_group_folds(original_train, args.folds - 1, args.seed))

    assigned = np.concatenate(validation_folds)
    if len(np.unique(assigned)) != len(development) or set(assigned) != set(development.patch_id):
        raise RuntimeError("validation folds do not partition the development patches exactly once")
    reports = []
    args.outdir.mkdir(parents=True, exist_ok=True)
    development_ids = development.patch_id.to_numpy(dtype=np.int32)
    group_by_id = development.set_index("patch_id").source_group.to_dict()
    for fold, val_ids in enumerate(validation_folds):
        train_ids = development_ids[~np.isin(development_ids, val_ids)]
        if set(train_ids) & set(val_ids) or set(train_ids) & test_ids or set(val_ids) & test_ids:
            raise RuntimeError(f"patch leakage in fold {fold}")
        train_groups = {group_by_id[int(value)] for value in train_ids}
        val_groups = {group_by_id[int(value)] for value in val_ids}
        if train_groups & val_groups:
            raise RuntimeError(f"source-group leakage in fold {fold}")
        np.save(args.outdir / f"fold_{fold}_train_ids.npy", train_ids)
        np.save(args.outdir / f"fold_{fold}_val_ids.npy", val_ids)
        val_rows = development.loc[development.patch_id.isin(val_ids)]
        reports.append(
            {
                "fold": fold,
                "train_patches": int(len(train_ids)),
                "validation_patches": int(len(val_ids)),
                "train_groups": int(len(train_groups)),
                "validation_groups": int(len(val_groups)),
                "validation_sources": val_rows.source.value_counts().to_dict(),
                "is_original_validation_fold": fold == 0,
                "train_ids": str(args.outdir / f"fold_{fold}_train_ids.npy"),
                "validation_ids": str(args.outdir / f"fold_{fold}_val_ids.npy"),
            }
        )
    manifest = {
        "seed": args.seed,
        "folds": args.folds,
        "policy": "fold 0 is the original locked validation split; original training groups are partitioned across remaining folds",
        "test_patches_accessed": False,
        "development_patches": int(len(development)),
        "fold_reports": reports,
    }
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
