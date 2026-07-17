#!/usr/bin/env python
"""Train dustbin/count/PQ-aligned CellViT++ token classifiers.

This script keeps the CellViT segmentation masks and token features fixed.  It
supports a seventh reject class, a differentiable central-patch count loss, and
a soft pooled-PQ surrogate.  Every learning-rate candidate is selected only on
the validation split.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CELLVIT_ROOT = ROOT / "third_party" / "CellViT-plus-plus"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CELLVIT_ROOT))

from cpath_conic.constants import CLASS_NAMES, COUNT_COLUMNS
from cpath_conic.experiment_metrics import evaluate_fixed_masks, load_or_build_instance_cache


def patch_index_ranges(patch_ids: np.ndarray) -> dict[int, np.ndarray]:
    order = np.argsort(patch_ids, kind="stable")
    sorted_ids = patch_ids[order]
    unique, starts, counts = np.unique(sorted_ids, return_index=True, return_counts=True)
    return {int(patch): order[start : start + count] for patch, start, count in zip(unique, starts, counts)}


def infer_assignments(model, features: np.ndarray, device: str, num_classes: int, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    import torch

    model.eval()
    assignments = np.zeros(len(features), dtype=np.int8)
    probabilities = np.zeros((len(features), num_classes), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            stop = min(start + batch_size, len(features))
            logits = model(torch.from_numpy(features[start:stop]).float().to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
            probabilities[start:stop] = probs
            if num_classes == 7:
                assignments[start:stop] = probs.argmax(axis=1).astype(np.int8)
            else:
                assignments[start:stop] = probs.argmax(axis=1).astype(np.int8) + 1
    return assignments, probabilities


def train(args: argparse.Namespace) -> None:
    import torch
    import torch.nn as nn
    from cellvit.models.classifier.linear_classifier import LinearClassifier

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing an accidental CPU training run")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    feature_data = np.load(args.features)
    features = feature_data["features"].astype(np.float32)
    patch_ids = feature_data["patch_ids"].astype(np.int32)
    instance_ids = feature_data["instance_ids"].astype(np.int32)
    labels = feature_data["labels"].astype(np.int8)
    ious = feature_data["ious"].astype(np.float32)
    metadata = pd.read_csv(args.prepared / "metadata.csv").sort_values("patch_id")
    split_by_patch = metadata.set_index("patch_id").split.to_dict()
    train_patch_ids = metadata.loc[metadata.split == "train", "patch_id"].to_numpy(dtype=np.int32)
    val_patch_ids = metadata.loc[metadata.split == "val", "patch_id"].to_numpy(dtype=np.int32)
    train_record_mask = np.isin(patch_ids, train_patch_ids)
    val_record_mask = np.isin(patch_ids, val_patch_ids)

    cache = load_or_build_instance_cache(
        args.cache,
        args.prepared,
        patch_ids,
        instance_ids,
        args.instance_maps,
    )
    central = cache["central"].astype(bool)
    gt_full_counts = cache["gt_full_counts"].astype(np.float32)
    metadata_counts = metadata.set_index("patch_id")[COUNT_COLUMNS].to_numpy(dtype=np.float32)

    num_classes = 7 if args.dustbin else 6
    targets = labels.astype(np.int64)
    matched = (labels > 0) & (ious > args.match_iou)
    if args.dustbin:
        targets = np.where(matched, labels, 0).astype(np.int64)
        supervision_mask = train_record_mask
    else:
        targets = labels.astype(np.int64) - 1
        supervision_mask = train_record_mask & matched

    train_target_counts = np.bincount(targets[supervision_mask], minlength=num_classes).astype(np.float64)
    if args.weight_scheme == "inverse_frequency":
        class_weights = np.power(train_target_counts.sum() / np.maximum(train_target_counts, 1), args.class_weight_power)
    else:
        frequencies = train_target_counts / max(train_target_counts.sum(), 1.0)
        class_weights = np.power(1.0 - frequencies, args.class_balance_rho)
    class_weights /= class_weights.mean()
    class_weights_tensor = torch.from_numpy(class_weights.astype(np.float32)).to(args.device)

    count_variance = metadata.loc[metadata.split == "train", COUNT_COLUMNS].var(ddof=0).to_numpy(dtype=np.float32)
    count_variance = torch.from_numpy(np.maximum(count_variance, 1.0)).to(args.device)
    ranges = patch_index_ranges(patch_ids)
    learning_rates = [float(value) for value in args.learning_rates.split(",") if value.strip()]
    if not learning_rates:
        raise ValueError("At least one learning rate is required")

    def validation_metrics(model):
        val_indices = np.flatnonzero(val_record_mask)
        assignments, _ = infer_assignments(model, features[val_indices], args.device, num_classes, args.inference_batch_size)
        all_assignments = np.zeros(len(features), dtype=np.int8)
        all_assignments[val_indices] = assignments
        return evaluate_fixed_masks(
            all_assignments,
            labels,
            ious,
            patch_ids,
            central,
            metadata,
            gt_full_counts,
            "val",
            match_iou=args.match_iou,
        )

    curve_rows = []
    best_overall = None
    for learning_rate in learning_rates:
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        model = LinearClassifier(
            embed_dim=features.shape[1],
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            drop_rate=args.dropout,
        ).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
        def classification_loss(logits, batch_targets):
            if args.loss == "cross_entropy":
                return nn.functional.cross_entropy(
                    logits,
                    batch_targets,
                    weight=class_weights_tensor,
                    label_smoothing=args.label_smoothing,
                )
            per_example = nn.functional.cross_entropy(
                logits,
                batch_targets,
                weight=class_weights_tensor,
                label_smoothing=args.label_smoothing,
                reduction="none",
            )
            true_probability = torch.softmax(logits, dim=1).gather(1, batch_targets[:, None]).squeeze(1)
            return (((1.0 - true_probability) ** args.focal_gamma) * per_example).mean()
        candidate_best = None
        stale_epochs = 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            shuffled_patches = rng.permutation(train_patch_ids)
            sums = {"total": 0.0, "ce": 0.0, "count": 0.0, "pq": 0.0, "batches": 0}
            for offset in range(0, len(shuffled_patches), args.patch_batch_size):
                batch_patches = shuffled_patches[offset : offset + args.patch_batch_size]
                record_chunks = [ranges[int(patch)] for patch in batch_patches if int(patch) in ranges]
                if not record_chunks:
                    continue
                record_indices = np.concatenate(record_chunks)
                local_patch = np.concatenate([
                    np.full(len(ranges[int(patch)]), local, dtype=np.int64)
                    for local, patch in enumerate(batch_patches)
                    if int(patch) in ranges
                ])
                x = torch.from_numpy(features[record_indices]).float().to(args.device)
                logits = model(x)
                batch_targets = torch.from_numpy(targets[record_indices]).long().to(args.device)
                if args.dustbin:
                    ce_loss = classification_loss(logits, batch_targets)
                    cell_probs = torch.softmax(logits, dim=1)[:, 1:]
                else:
                    valid = torch.from_numpy(matched[record_indices]).bool().to(args.device)
                    ce_loss = classification_loss(logits[valid], batch_targets[valid]) if valid.any() else logits.sum() * 0
                    cell_probs = torch.softmax(logits, dim=1)

                local_patch_tensor = torch.from_numpy(local_patch).long().to(args.device)
                central_tensor = torch.from_numpy(central[record_indices]).bool().to(args.device)
                soft_counts = torch.zeros((len(batch_patches), 6), device=args.device)
                soft_counts.index_add_(0, local_patch_tensor[central_tensor], cell_probs[central_tensor])
                true_counts = torch.from_numpy(metadata_counts[batch_patches]).float().to(args.device)
                count_loss = (((soft_counts - true_counts) ** 2) / count_variance).mean()

                pred_totals = cell_probs.sum(dim=0)
                batch_gt_totals = torch.from_numpy(gt_full_counts[batch_patches].sum(axis=0)).float().to(args.device)
                pq_numerator = torch.zeros(6, device=args.device)
                matched_batch = matched[record_indices]
                if matched_batch.any():
                    matched_indices = np.flatnonzero(matched_batch)
                    matched_classes = torch.from_numpy(labels[record_indices][matched_indices].astype(np.int64) - 1).long().to(args.device)
                    matched_ious = torch.from_numpy(ious[record_indices][matched_indices]).float().to(args.device)
                    contributions = cell_probs[torch.from_numpy(matched_indices).long().to(args.device), matched_classes] * matched_ious
                    pq_numerator.index_add_(0, matched_classes, contributions)
                pq_denominator = 0.5 * (pred_totals + batch_gt_totals)
                represented = batch_gt_totals > 0
                soft_pq = (pq_numerator[represented] / (pq_denominator[represented] + 1e-6)).mean()
                pq_loss = 1.0 - soft_pq

                loss = ce_loss + args.count_loss_weight * count_loss + args.pq_loss_weight * pq_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                sums["total"] += float(loss.item())
                sums["ce"] += float(ce_loss.item())
                sums["count"] += float(count_loss.item())
                sums["pq"] += float(pq_loss.item())
                sums["batches"] += 1

            metrics = validation_metrics(model)
            if args.selection_metric == "R2":
                selection_score = metrics["R2"]
            elif args.selection_metric == "mPQ+":
                selection_score = metrics["mPQ+"]
            else:
                selection_score = 0.5 * (metrics["R2"] + metrics["mPQ+"])
            row = {
                "learning_rate": learning_rate,
                "epoch": epoch,
                "train_loss": sums["total"] / sums["batches"],
                "train_ce_loss": sums["ce"] / sums["batches"],
                "train_count_loss": sums["count"] / sums["batches"],
                "train_pq_loss": sums["pq"] / sums["batches"],
                "val_R2": metrics["R2"],
                "val_mPQ+": metrics["mPQ+"],
                "val_rejected_fraction": metrics["rejected_fraction"],
                "selection_score": selection_score,
            }
            curve_rows.append(row)
            if candidate_best is None or selection_score > candidate_best["score"]:
                candidate_best = {
                    "score": selection_score,
                    "epoch": epoch,
                    "metrics": {key: value for key, value in metrics.items() if key != "predicted_counts"},
                    "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
            print(
                f"lr={learning_rate:g} epoch={epoch}/{args.epochs} loss={row['train_loss']:.4f} "
                f"val_R2={metrics['R2']:.4f} val_mPQ+={metrics['mPQ+']:.4f} "
                f"reject={metrics['rejected_fraction']:.3f}",
                flush=True,
            )
            if stale_epochs >= args.patience:
                break
        if best_overall is None or candidate_best["score"] > best_overall["score"]:
            best_overall = {**candidate_best, "learning_rate": learning_rate}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": best_overall["state_dict"],
        "embed_dim": int(features.shape[1]),
        "hidden_dim": args.hidden_dim,
        "num_classes": num_classes,
        "class_names": (["dustbin"] + CLASS_NAMES) if args.dustbin else CLASS_NAMES,
        "selected_learning_rate": best_overall["learning_rate"],
        "selected_epoch": best_overall["epoch"],
        "selection_metric": args.selection_metric,
        "validation_metrics": best_overall["metrics"],
        "count_loss_weight": args.count_loss_weight,
        "pq_loss_weight": args.pq_loss_weight,
        "class_weight_power": args.class_weight_power,
        "loss": args.loss,
        "weight_scheme": args.weight_scheme,
        "class_balance_rho": args.class_balance_rho,
        "focal_gamma": args.focal_gamma,
        "label_smoothing": args.label_smoothing,
        "seed": args.seed,
    }
    torch.save(checkpoint, args.out)
    curve_path = args.out.with_name(f"{args.out.stem}_curve.csv")
    with curve_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)
    curve_path.with_suffix(".json").write_text(json.dumps({
        "selected_learning_rate": best_overall["learning_rate"],
        "selected_epoch": best_overall["epoch"],
        "selection_metric": args.selection_metric,
        "validation_metrics": best_overall["metrics"],
        "learning_rates": learning_rates,
        "configuration": {
            "dustbin": args.dustbin,
            "count_loss_weight": args.count_loss_weight,
            "pq_loss_weight": args.pq_loss_weight,
            "class_weight_power": args.class_weight_power,
            "loss": args.loss,
            "weight_scheme": args.weight_scheme,
            "class_balance_rho": args.class_balance_rho,
            "focal_gamma": args.focal_gamma,
            "label_smoothing": args.label_smoothing,
        },
    }, indent=2))
    print(f"selected lr={best_overall['learning_rate']:g} epoch={best_overall['epoch']} score={best_overall['score']:.5f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--instance-maps", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dustbin", action="store_true")
    parser.add_argument("--count-loss-weight", type=float, default=0.0)
    parser.add_argument("--pq-loss-weight", type=float, default=0.0)
    parser.add_argument("--class-weight-power", type=float, default=0.0)
    parser.add_argument("--loss", choices=["cross_entropy", "focal"], default="cross_entropy")
    parser.add_argument("--weight-scheme", choices=["inverse_frequency", "complement_frequency"], default="inverse_frequency")
    parser.add_argument("--class-balance-rho", type=float, default=3.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--learning-rates", default="1e-4,3e-4,1e-3")
    parser.add_argument("--selection-metric", choices=["R2", "mPQ+", "mean"], default="mean")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--patch-batch-size", type=int, default=32)
    parser.add_argument("--inference-batch-size", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
