#!/usr/bin/env python
"""Run CellViT++ SAM-H on CoNIC patches and adapt its token classifier.

The CellViT++ segmentation model remains frozen. We extract its per-cell token
features, pair predicted instances with CoNIC annotations only on the training
split, train the lightweight six-class head, and apply that head to held-out
predicted instances. The model's binary/HV outputs are converted to instance
maps with a deterministic watershed postprocessor.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

ROOT = Path(__file__).resolve().parents[1]
CELLVIT_ROOT = ROOT / "third_party" / "CellViT-plus-plus"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CELLVIT_ROOT))

from cpath_conic.constants import CLASS_NAMES
from cpath_conic.data import central_crop_counts, load_metadata
from cpath_conic.hv import decode_hv, load_decoder_config
from cpath_conic.lora import load_lora_adapter
from cpath_conic.tta import (
    invert_hv_horizontal_flip,
    invert_hv_rotation,
    invert_hv_vertical_flip,
    invert_spatial_rotation,
)


def postprocess_instances(binary_prob: np.ndarray, threshold: float = 0.5, min_size: int = 10) -> np.ndarray:
    foreground = binary_prob[..., 1] >= threshold
    distance = ndi.distance_transform_edt(foreground)
    peaks = peak_local_max(distance, min_distance=5, threshold_abs=1.5, labels=foreground)
    markers = np.zeros_like(foreground, dtype=np.int32)
    for index, (y, x) in enumerate(peaks, start=1):
        markers[y, x] = index
    if not len(peaks):
        markers, _ = ndi.label(foreground)
    instances = watershed(-distance, markers, mask=foreground).astype(np.int32)
    for instance_id in np.unique(instances):
        if instance_id == 0:
            continue
        if int((instances == instance_id).sum()) < min_size:
            instances[instances == instance_id] = 0
    remapped = np.zeros_like(instances)
    for new_id, old_id in enumerate(np.unique(instances)[1:], start=1):
        remapped[instances == old_id] = new_id
    return remapped


def instance_bbox(instance_map: np.ndarray, instance_id: int) -> tuple[int, int, int, int]:
    ys, xs = np.where(instance_map == instance_id)
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def pool_instance_token(tokens: np.ndarray, instance_map: np.ndarray, instance_id: int) -> np.ndarray:
    """Pool a CHW token grid over the token cells touched by an instance bbox."""
    y0, y1, x0, x1 = instance_bbox(instance_map, instance_id)
    stride_y = instance_map.shape[0] / tokens.shape[1]
    stride_x = instance_map.shape[1] / tokens.shape[2]
    ty0 = max(0, int(np.floor(y0 / stride_y)))
    ty1 = min(tokens.shape[1], max(ty0 + 1, int(np.ceil(y1 / stride_y))))
    tx0 = max(0, int(np.floor(x0 / stride_x)))
    tx1 = min(tokens.shape[2], max(tx0 + 1, int(np.ceil(x1 / stride_x))))
    return tokens[:, ty0:ty1, tx0:tx1].mean(axis=(1, 2))


def match_gt_class(pred_map: np.ndarray, instance_id: int, gt_inst: np.ndarray, gt_cls: np.ndarray) -> tuple[int, float]:
    pred_mask = pred_map == instance_id
    gt_ids, overlap = np.unique(gt_inst[pred_mask], return_counts=True)
    candidates = [(int(count), int(gt_id)) for gt_id, count in zip(gt_ids, overlap) if gt_id != 0]
    if not candidates:
        return 0, 0.0
    overlap_count, gt_id = max(candidates)
    gt_mask = gt_inst == gt_id
    union = int(pred_mask.sum() + gt_mask.sum() - overlap_count)
    iou = overlap_count / union if union else 0.0
    cls_values = gt_cls[gt_mask]
    cls = int(np.bincount(cls_values.astype(np.int64), minlength=7).argmax())
    return cls, float(iou)


def load_model(checkpoint_path: Path, device: str):
    import torch
    from cellvit.models.cell_segmentation.cellvit_sam import CellViTSAM

    def unflatten_dict(flat, separator):
        nested = {}
        for key, value in flat.items():
            cursor = nested
            parts = key.split(separator) if isinstance(key, str) else [key]
            for part in parts[:-1]:
                cursor = cursor.setdefault(part, {})
            cursor[parts[-1]] = value
        return nested

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    run_conf = unflatten_dict(checkpoint["config"], ".")
    model_conf = run_conf["model"]
    data_conf = run_conf["data"]
    model = CellViTSAM(
        model_path=None,
        num_nuclei_classes=int(data_conf.get("num_nuclei_classes", 6)),
        num_tissue_classes=int(data_conf.get("num_tissue_classes", 0)),
        vit_structure=model_conf.get("backbone", "SAM-H"),
        regression_loss=bool(model_conf.get("regression_loss", False)),
    )
    message = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    print(f"loaded CellViT++ checkpoint: {message}")
    model.to(device).eval()
    normalize = run_conf.get("transformations", {}).get("normalize", {})
    mean = np.asarray(normalize.get("mean", [0.5, 0.5, 0.5]), dtype=np.float32)
    std = np.asarray(normalize.get("std", [0.5, 0.5, 0.5]), dtype=np.float32)
    return model, mean, std


def extract(args: argparse.Namespace) -> None:
    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    metadata = load_metadata(args.prepared).sort_values("patch_id")
    if args.max_patches:
        metadata = metadata.head(args.max_patches)
    model, mean, std = load_model(args.checkpoint, args.device)
    decoder_config = load_decoder_config(args.decoder_config) if args.postprocessor == "cellvit-hv" else None
    if args.lora_adapter is not None:
        configuration = load_lora_adapter(model, args.lora_adapter)
        model.to(args.device).eval()
        print(f"loaded LoRA adapter {args.lora_adapter}: {configuration}", flush=True)
    n_total = int(load_metadata(args.prepared).patch_id.max()) + 1
    fixed_instance_maps = np.load(args.fixed_instance_maps, mmap_mode="r") if args.fixed_instance_maps else None
    if fixed_instance_maps is not None and len(fixed_instance_maps) != n_total:
        raise ValueError(f"Fixed instance maps contain {len(fixed_instance_maps)} patches, expected {n_total}")
    instance_maps = None if fixed_instance_maps is not None else np.zeros((n_total, 256, 256), dtype=np.int32)
    features, feature_patch_ids, feature_instance_ids, feature_labels, feature_ious = [], [], [], [], []
    processed = []
    with torch.no_grad():
        for position, row in enumerate(metadata.itertuples(index=False), start=1):
            patch_id = int(row.patch_id)
            image = np.asarray(Image.open(args.prepared / "images" / f"{patch_id:05d}.png").convert("RGB"), dtype=np.float32) / 255.0
            tensor = torch.from_numpy(((image - mean) / std).transpose(2, 0, 1)).unsqueeze(0).float().to(args.device)
            output = model(tensor, retrieve_tokens=True)
            if fixed_instance_maps is not None:
                token_terms = [output["tokens"][0].detach().float().cpu()]
                if args.flip_tta:
                    output_h = model(tensor.flip(-1), retrieve_tokens=True)
                    output_v = model(tensor.flip(-2), retrieve_tokens=True)
                    token_terms.extend([
                        output_h["tokens"][0].detach().float().flip(-1).cpu(),
                        output_v["tokens"][0].detach().float().flip(-2).cpu(),
                    ])
                if args.rotation_tta:
                    for k in (1, 2, 3):
                        output_r = model(torch.rot90(tensor, k, dims=(-2, -1)), retrieve_tokens=True)
                        token_terms.append(invert_spatial_rotation(output_r["tokens"].detach().float(), k)[0].cpu())
                tokens = torch.stack(token_terms).mean(dim=0)
                instances = np.asarray(fixed_instance_maps[patch_id])
            elif args.postprocessor == "cellvit-hv":
                foreground = torch.softmax(output["nuclei_binary_map"].float(), dim=1)[0, 1].cpu().numpy()
                hv_map = output["hv_map"][0].float().permute(1, 2, 0).cpu().numpy()
                tokens = output["tokens"][0].detach().float().cpu()
                foreground_terms = [foreground]
                hv_terms = [hv_map]
                token_terms = [tokens]
                if args.flip_tta:
                    output_h = model(tensor.flip(-1), retrieve_tokens=True)
                    foreground_h = torch.softmax(output_h["nuclei_binary_map"].float(), dim=1)[0, 1].flip(-1).cpu().numpy()
                    hv_h = invert_hv_horizontal_flip(output_h["hv_map"].float())[0]
                    tokens_h = output_h["tokens"][0].detach().float().flip(-1).cpu()
                    output_v = model(tensor.flip(-2), retrieve_tokens=True)
                    foreground_v = torch.softmax(output_v["nuclei_binary_map"].float(), dim=1)[0, 1].flip(-2).cpu().numpy()
                    hv_v = invert_hv_vertical_flip(output_v["hv_map"].float())[0]
                    tokens_v = output_v["tokens"][0].detach().float().flip(-2).cpu()
                    foreground_terms.extend([foreground_h, foreground_v])
                    hv_terms.extend([hv_h.permute(1, 2, 0).cpu().numpy(), hv_v.permute(1, 2, 0).cpu().numpy()])
                    token_terms.extend([tokens_h, tokens_v])
                if args.rotation_tta:
                    for k in (1, 2, 3):
                        output_r = model(torch.rot90(tensor, k, dims=(-2, -1)), retrieve_tokens=True)
                        foreground_r = invert_spatial_rotation(
                            torch.softmax(output_r["nuclei_binary_map"].float(), dim=1)[:, 1], k
                        )[0].cpu().numpy()
                        hv_r = invert_hv_rotation(output_r["hv_map"].float(), k)[0].permute(1, 2, 0).cpu().numpy()
                        tokens_r = invert_spatial_rotation(output_r["tokens"].detach().float(), k)[0].cpu()
                        foreground_terms.append(foreground_r)
                        hv_terms.append(hv_r)
                        token_terms.append(tokens_r)
                foreground = np.mean(foreground_terms, axis=0)
                hv_map = np.mean(hv_terms, axis=0)
                tokens = torch.stack(token_terms).mean(dim=0)
                instances = decode_hv(foreground, hv_map, **decoder_config)
            else:
                binary = torch.softmax(output["nuclei_binary_map"], dim=1)[0].permute(1, 2, 0).float().cpu().numpy()
                instances = postprocess_instances(binary, threshold=args.binary_threshold)
                tokens = output["tokens"][0].detach().float().cpu()
            if instance_maps is not None:
                instance_maps[patch_id] = instances
            label = np.load(args.prepared / "labels" / f"{patch_id:05d}.npy", allow_pickle=True).item()
            tokens = tokens.numpy()
            gt_inst, gt_cls = label["inst_map"], label["class_map"]
            for instance_id in np.unique(instances):
                if instance_id == 0:
                    continue
                feature = pool_instance_token(tokens, instances, int(instance_id))
                cls, iou = match_gt_class(instances, int(instance_id), gt_inst, gt_cls)
                features.append(feature)
                feature_patch_ids.append(patch_id)
                feature_instance_ids.append(int(instance_id))
                feature_labels.append(cls)
                feature_ious.append(iou)
            processed.append(patch_id)
            if position % 10 == 0 or position == len(metadata):
                print(f"extracted {position}/{len(metadata)} patches", flush=True)
    args.outdir.mkdir(parents=True, exist_ok=True)
    if instance_maps is not None:
        np.save(args.outdir / "instance_maps.npy", instance_maps)
    np.savez_compressed(args.outdir / "features.npz", features=np.asarray(features, dtype=np.float32), patch_ids=np.asarray(feature_patch_ids, dtype=np.int32), instance_ids=np.asarray(feature_instance_ids, dtype=np.int32), labels=np.asarray(feature_labels, dtype=np.int8), ious=np.asarray(feature_ious, dtype=np.float32))
    (args.outdir / "processed_patch_ids.json").write_text(json.dumps(processed, indent=2))
    (args.outdir / "extraction.json").write_text(json.dumps({"checkpoint": str(args.checkpoint), "lora_adapter": str(args.lora_adapter) if args.lora_adapter else None, "fixed_instance_maps": str(args.fixed_instance_maps) if args.fixed_instance_maps else None, "device": args.device, "n_total": n_total, "n_processed": len(processed), "n_features": len(features), "class_names": CLASS_NAMES, "postprocessing": {"name": "fixed-instance-maps" if fixed_instance_maps is not None else args.postprocessor, "binary_threshold": args.binary_threshold if args.postprocessor == "simple" and fixed_instance_maps is None else None, "magnification": args.magnification if args.postprocessor == "cellvit-hv" and fixed_instance_maps is None else None, "decoder_config": decoder_config if fixed_instance_maps is None else None, "flip_tta": args.flip_tta, "rotation_tta": args.rotation_tta, "min_size": 10}}, indent=2))


def train_classifier(args: argparse.Namespace) -> None:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from cellvit.models.classifier.linear_classifier import LinearClassifier
    from sklearn.metrics import f1_score

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = np.load(args.features)
    metadata = load_metadata(args.prepared)
    split_by_patch = metadata.set_index("patch_id")["split"].to_dict()
    valid = (data["labels"] > 0) & (data["ious"] > args.min_train_iou)
    train_mask = np.asarray([valid[i] and split_by_patch.get(int(patch), "") == "train" for i, patch in enumerate(data["patch_ids"])])
    val_mask = np.asarray([valid[i] and split_by_patch.get(int(patch), "") == "val" for i, patch in enumerate(data["patch_ids"])])
    if not train_mask.any():
        raise RuntimeError("No matched training features; run extraction over the train split")
    print(f"classifier supervision: {int(train_mask.sum())} train instances, {int(val_mask.sum())} validation instances, min_iou={args.min_train_iou:g}", flush=True)
    x_train = torch.from_numpy(data["features"][train_mask]).float()
    y_train = torch.from_numpy(data["labels"][train_mask].astype(np.int64)) - 1
    counts = np.bincount(y_train.numpy(), minlength=6).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1)
    weights = torch.from_numpy(weights / weights.mean()).float().to(args.device)

    x_val = torch.from_numpy(data["features"][val_mask]).float() if val_mask.any() else None
    y_val = torch.from_numpy(data["labels"][val_mask].astype(np.int64) - 1).long() if val_mask.any() else None

    def score(model, features, labels, loss_fn):
        if features is None or labels is None:
            return None
        model.eval()
        with torch.no_grad():
            logits = model(features.to(args.device))
            loss = float(loss_fn(logits, labels.to(args.device)).item())
            predicted = logits.argmax(1).cpu().numpy()
        target = labels.numpy()
        return {
            "loss": loss,
            "accuracy": float((predicted == target).mean()),
            "macro_f1": float(f1_score(target, predicted, labels=np.arange(6), average="macro", zero_division=0)),
            "per_class_f1": f1_score(target, predicted, labels=np.arange(6), average=None, zero_division=0).tolist(),
        }

    learning_rates = [args.learning_rate] if args.learning_rate is not None else [float(value) for value in args.learning_rates.split(",") if value.strip()]
    if not learning_rates:
        raise ValueError("At least one learning rate is required")

    curve_rows = []
    best_overall = None
    selection_metric = "val_macro_f1" if val_mask.any() else "train_macro_f1"
    for learning_rate in learning_rates:
        torch.manual_seed(args.seed)
        model = LinearClassifier(embed_dim=x_train.shape[1], hidden_dim=args.hidden_dim, num_classes=6, drop_rate=0.1).to(args.device)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        loader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=args.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
        candidate_best = None
        stale_epochs = 0
        for epoch in range(args.epochs):
            model.train()
            running = 0.0
            for features, labels in loader:
                logits = model(features.to(args.device))
                loss = loss_fn(logits, labels.to(args.device))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                running += float(loss.item()) * len(labels)

            train_metrics = score(model, x_train, y_train, loss_fn)
            val_metrics = score(model, x_val, y_val, loss_fn)
            row = {
                "learning_rate": learning_rate,
                "epoch": epoch + 1,
                "train_loss": running / len(x_train),
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "val_loss": val_metrics["loss"] if val_metrics else None,
                "val_accuracy": val_metrics["accuracy"] if val_metrics else None,
                "val_macro_f1": val_metrics["macro_f1"] if val_metrics else None,
            }
            if val_metrics:
                row.update({f"val_f1_{name}": value for name, value in zip(CLASS_NAMES, val_metrics["per_class_f1"])})
            curve_rows.append(row)
            score_value = row[selection_metric]
            if candidate_best is None or score_value > candidate_best["score"]:
                candidate_best = {"score": score_value, "epoch": epoch + 1, "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, "row": row}
                stale_epochs = 0
            else:
                stale_epochs += 1
            print(
                f"lr={learning_rate:g} epoch={epoch + 1}/{args.epochs} "
                f"train_loss={row['train_loss']:.4f} train_macro_f1={row['train_macro_f1']:.4f} "
                f"val_macro_f1={row['val_macro_f1'] if row['val_macro_f1'] is not None else 'n/a'}",
                flush=True,
            )
            if stale_epochs >= args.patience:
                break
        if best_overall is None or candidate_best["score"] > best_overall["score"]:
            best_overall = {**candidate_best, "learning_rate": learning_rate}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model = LinearClassifier(embed_dim=x_train.shape[1], hidden_dim=args.hidden_dim, num_classes=6, drop_rate=0.1)
    model.load_state_dict(best_overall["state_dict"])
    torch.save({
        "model_state_dict": model.state_dict(),
        "embed_dim": int(x_train.shape[1]),
        "hidden_dim": args.hidden_dim,
        "num_classes": 6,
        "class_names": CLASS_NAMES,
        "selected_learning_rate": best_overall["learning_rate"],
        "selected_epoch": best_overall["epoch"],
        "selection_metric": selection_metric,
        "seed": args.seed,
        "min_train_iou": args.min_train_iou,
    }, args.out)
    curve_out = args.curve_out or args.out.with_name(f"{args.out.stem}_training_curve.csv")
    curve_out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in curve_rows for key in row})
    with curve_out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(curve_rows)
    summary_out = curve_out.with_suffix(".json")
    summary_out.write_text(json.dumps({
        "learning_rates_tested": learning_rates,
        "selected_learning_rate": best_overall["learning_rate"],
        "selected_epoch": best_overall["epoch"],
        "selection_metric": selection_metric,
        "best_metrics": best_overall["row"],
        "epochs_requested": args.epochs,
        "patience": args.patience,
        "seed": args.seed,
        "min_train_iou": args.min_train_iou,
    }, indent=2))
    print(f"selected lr={best_overall['learning_rate']:g} epoch={best_overall['epoch']} {selection_metric}={best_overall['score']:.4f}", flush=True)


def predict(args: argparse.Namespace) -> None:
    import torch
    from cellvit.models.classifier.linear_classifier import LinearClassifier

    feature_data = np.load(args.features)
    feature_matrix = feature_data["features"]
    feature_patch_ids = feature_data["patch_ids"].astype(np.int32)
    feature_instance_ids = feature_data["instance_ids"].astype(np.int32)
    instances = np.load(args.instance_maps)
    checkpoint = torch.load(args.classifier, map_location="cpu", weights_only=False)
    model = LinearClassifier(embed_dim=int(checkpoint["embed_dim"]), hidden_dim=int(checkpoint["hidden_dim"]), num_classes=6).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    output = np.zeros((*instances.shape, 2), dtype=np.int32)
    with torch.no_grad():
        logits = model(torch.from_numpy(feature_matrix).float().to(args.device))
        class_probs = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
    classes = class_probs.argmax(1).astype(np.int32) + 1
    # Build each patch's instance-class map with one lookup-table pass. A
    # per-instance full-image mask would be unnecessarily quadratic here.
    for patch_id in np.unique(feature_patch_ids):
        rows = feature_patch_ids == patch_id
        instance_map = instances[int(patch_id)]
        class_lut = np.zeros(int(instance_map.max()) + 1, dtype=np.int32)
        class_lut[feature_instance_ids[rows]] = classes[rows]
        output[int(patch_id), ..., 0] = instance_map
        output[int(patch_id), ..., 1] = class_lut[instance_map]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, output)
    np.savez_compressed(args.out.with_name("cell_probabilities.npz"), patch_ids=feature_patch_ids, instance_ids=feature_instance_ids, class_probs=class_probs)
    np.save(args.out.with_name("counts.npy"), np.asarray([central_crop_counts(patch[..., 0], patch[..., 1]) for patch in output], dtype=np.int32))


def download_checkpoint(args: argparse.Namespace) -> None:
    from urllib.request import urlretrieve

    args.out.parent.mkdir(parents=True, exist_ok=True)
    url = "https://figshare.com/ndownloader/files/45351940"
    print(f"downloading {url} -> {args.out}")
    urlretrieve(url, args.out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["extract", "train", "predict", "download-checkpoint"], required=True)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/cellvit_conic"))
    parser.add_argument("--features", type=Path)
    parser.add_argument("--instance-maps", type=Path)
    parser.add_argument("--fixed-instance-maps", type=Path, default=None, help="Pool tokens over these existing masks instead of decoding new instances")
    parser.add_argument("--classifier", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument("--binary-threshold", type=float, default=0.5)
    parser.add_argument("--postprocessor", choices=["simple", "cellvit-hv"], default="simple")
    parser.add_argument("--magnification", type=int, choices=[20, 40], default=20)
    parser.add_argument("--decoder-config", type=Path, default=None, help="Validation-selected HV decoder report/config JSON")
    parser.add_argument("--flip-tta", action="store_true", help="Average original, horizontal-flip, and vertical-flip maps/tokens before decoding")
    parser.add_argument("--rotation-tta", action="store_true", help="Also average 90/180/270-degree maps/tokens after exact spatial and HV-vector inversion")
    parser.add_argument("--lora-adapter", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=None, help="Use one explicit learning rate instead of the default sweep")
    parser.add_argument("--learning-rates", default="1e-4,3e-4,1e-3", help="Comma-separated learning-rate sweep")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--curve-out", type=Path, default=None, help="CSV path for the per-epoch learning curves")
    parser.add_argument("--min-train-iou", type=float, default=0.5, help="Minimum predicted/GT instance IoU for classifier supervision")
    args = parser.parse_args()
    if args.mode == "download-checkpoint":
        if not args.out:
            parser.error("--out is required for download-checkpoint")
        download_checkpoint(args)
    elif args.mode == "extract":
        for name in ["prepared", "checkpoint"]:
            if getattr(args, name) is None:
                parser.error(f"--{name} is required for extract")
        extract(args)
    elif args.mode == "train":
        for name in ["prepared", "features", "out"]:
            if getattr(args, name) is None:
                parser.error(f"--{name} is required for train")
        train_classifier(args)
    elif args.mode == "predict":
        for attr, flag in [("features", "--features"), ("instance_maps", "--instance-maps"), ("classifier", "--classifier"), ("out", "--out")]:
            if getattr(args, attr) is None:
                parser.error(f"{flag} is required for predict")
        predict(args)


if __name__ == "__main__":
    main()
