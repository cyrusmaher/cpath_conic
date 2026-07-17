#!/usr/bin/env python
"""Validation-selected LoRA adaptation of CellViT-SAM segmentation."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from skimage.color import hed2rgb, rgb2hed

ROOT = Path(__file__).resolve().parents[1]
CELLVIT_ROOT = ROOT / "third_party" / "CellViT-plus-plus"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CELLVIT_ROOT))

from cpath_conic.lora import inject_sam_lora, lora_state_dict, save_lora_adapter, set_lora_train_mode
from cpath_conic.hv import decode_hv, fast_binary_pq_stats, load_decoder_config
from cpath_conic.constants import CLASS_NAMES
from cpath_conic.directional import expand_hv_head, instance_directional_map, set_directional_header_trainable
from cpath_conic.segmentation import instance_hv_map
from cpath_conic.sampling import minority_patch_weights, source_class_patch_weights, source_patch_weights
from scripts.run_cellvit_conic import load_model


def color_blur_augmentation(
    image: Image.Image,
    jitter: float,
    blur_probability: float,
    blur_radius_min: float,
    blur_radius_max: float,
) -> Image.Image:
    """Apply label-preserving pathology color jitter and occasional light blur."""
    import torch

    factors = 1.0 + (2.0 * torch.rand(3).numpy() - 1.0) * jitter
    enhancers = [ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color]
    for index in torch.randperm(3).tolist():
        image = enhancers[index](image).enhance(float(factors[index]))
    if float(torch.rand(())) < blur_probability:
        radius = blur_radius_min + float(torch.rand(())) * (blur_radius_max - blur_radius_min)
        image = image.filter(ImageFilter.GaussianBlur(radius=radius))
    return image


def hed_stain_augmentation(
    image: Image.Image,
    probability: float = 0.8,
    scale_jitter: float = 0.1,
    offset_jitter: float = 0.02,
) -> Image.Image:
    """Perturb H/E optical-density channels without spatial distortion."""
    import torch

    if float(torch.rand(())) >= probability:
        return image
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    hed = rgb2hed(rgb)
    scales = 1.0 + (2.0 * torch.rand(2).numpy() - 1.0) * scale_jitter
    offsets = (2.0 * torch.rand(2).numpy() - 1.0) * offset_jitter
    hed[..., :2] = np.maximum(hed[..., :2] * scales + offsets, 0.0)
    hed[..., 2] = np.maximum(hed[..., 2], 0.0)
    augmented = np.clip(hed2rgb(hed), 0.0, 1.0)
    green_artifact = (augmented[..., 1] > augmented[..., 0] + 0.08) & (
        augmented[..., 1] > augmented[..., 2] + 0.08
    )
    if float(green_artifact.mean()) > 0.01:
        return image
    return Image.fromarray(np.rint(augmented * 255.0).astype(np.uint8), mode="RGB")


def select_train_validation_ids(
    metadata: pd.DataFrame,
    excluded_train_sources: set[str],
    validation_split: str,
    validation_sources: set[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply source-stress filters and enforce disjoint train/validation IDs."""
    train_mask = metadata.split.eq("train") & ~metadata.source.isin(excluded_train_sources)
    val_mask = metadata.split.eq(validation_split)
    if validation_sources:
        val_mask &= metadata.source.isin(validation_sources)
    train_ids = metadata.loc[train_mask, "patch_id"].to_numpy(dtype=np.int32)
    val_ids = metadata.loc[val_mask, "patch_id"].to_numpy(dtype=np.int32)
    if not len(train_ids) or not len(val_ids):
        raise ValueError("Source/split filters produced an empty training or validation set")
    overlap = np.intersect1d(train_ids, val_ids)
    if len(overlap):
        raise ValueError(f"Training and validation filters overlap on {len(overlap)} patches")
    return train_ids, val_ids


class CoNICSegmentationDataset:
    def __init__(
        self,
        prepared: Path,
        patch_ids: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        augmentation: str = "none",
        color_jitter: float = 0.1,
        blur_probability: float = 0.3,
        blur_radius_min: float = 0.3,
        blur_radius_max: float = 1.0,
        hed_probability: float = 0.8,
        hed_scale_jitter: float = 0.1,
        hed_offset_jitter: float = 0.02,
        directional_maps: bool = False,
    ):
        self.prepared = prepared
        self.patch_ids = patch_ids.astype(np.int32)
        self.mean = mean[:, None, None]
        self.std = std[:, None, None]
        self.augmentation = augmentation
        self.color_jitter = color_jitter
        self.blur_probability = blur_probability
        self.blur_radius_min = blur_radius_min
        self.blur_radius_max = blur_radius_max
        self.hed_probability = hed_probability
        self.hed_scale_jitter = hed_scale_jitter
        self.hed_offset_jitter = hed_offset_jitter
        self.directional_maps = directional_maps

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, index):
        import torch

        patch_id = int(self.patch_ids[index])
        image = Image.open(self.prepared / "images" / f"{patch_id:05d}.png").convert("RGB")
        if self.augmentation in {"color", "blur", "color_blur"}:
            image = color_blur_augmentation(
                image,
                self.color_jitter if self.augmentation in {"color", "color_blur"} else 0.0,
                self.blur_probability if self.augmentation in {"blur", "color_blur"} else 0.0,
                self.blur_radius_min,
                self.blur_radius_max,
            )
        elif self.augmentation == "hed":
            image = hed_stain_augmentation(
                image,
                self.hed_probability,
                self.hed_scale_jitter,
                self.hed_offset_jitter,
            )
        image = np.asarray(image, dtype=np.float32) / 255.0
        image = ((image.transpose(2, 0, 1) - self.mean) / self.std).astype(np.float32)
        label = np.load(self.prepared / "labels" / f"{patch_id:05d}.npy", allow_pickle=True).item()
        instance_map = label["inst_map"].astype(np.int32)
        binary = (instance_map > 0).astype(np.int64)
        hv = instance_directional_map(instance_map) if self.directional_maps else instance_hv_map(instance_map)
        return {
            "patch_id": patch_id,
            "image": torch.from_numpy(image),
            "binary": torch.from_numpy(binary),
            "hv": torch.from_numpy(hv),
            "instance_map": torch.from_numpy(instance_map),
        }


def segmentation_loss(output, binary, hv_target, hv_weight: float, hv_gradient_weight: float):
    import torch
    import torch.nn.functional as functional

    binary_logits = output["nuclei_binary_map"]
    ce = functional.cross_entropy(binary_logits, binary)
    foreground = torch.softmax(binary_logits, dim=1)[:, 1]
    target = binary.float()
    dice = 1.0 - (2 * (foreground * target).sum() + 1e-3) / (foreground.sum() + target.sum() + 1e-3)
    focus = target[:, None]
    hv_error = (output["hv_map"] - hv_target) ** 2
    channels = output["hv_map"].shape[1]
    hv_mse = (hv_error * focus).sum() / (channels * focus.sum() + 1e-6)

    pred_dx = output["hv_map"][:, :, :, 1:] - output["hv_map"][:, :, :, :-1]
    true_dx = hv_target[:, :, :, 1:] - hv_target[:, :, :, :-1]
    pred_dy = output["hv_map"][:, :, 1:, :] - output["hv_map"][:, :, :-1, :]
    true_dy = hv_target[:, :, 1:, :] - hv_target[:, :, :-1, :]
    focus_x = focus[:, :, :, 1:]
    focus_y = focus[:, :, 1:, :]
    gradient = (((pred_dx - true_dx) ** 2 * focus_x).sum() + ((pred_dy - true_dy) ** 2 * focus_y).sum())
    gradient /= channels * (focus_x.sum() + focus_y.sum()) + 1e-6
    total = ce + dice + hv_weight * hv_mse + hv_gradient_weight * gradient
    return total, {"ce": ce, "dice": dice, "hv_mse": hv_mse, "hv_gradient": gradient}


def validation_binary_pq(model, loader, device: str, decoder_config: dict) -> dict:
    import torch

    model.eval()
    totals = np.zeros(4, dtype=np.float64)
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].float().to(device)
            # Match run_cellvit_conic.py extraction exactly.  HV watershed is
            # threshold-sensitive; selecting adapters with bfloat16 maps and
            # deploying float32 maps can change the detected instance count.
            output = model(images)
            foreground = torch.softmax(output["nuclei_binary_map"].float(), dim=1)[:, 1].cpu().numpy()
            hv_maps = output["hv_map"].float().permute(0, 2, 3, 1).cpu().numpy()
            predicted = [decode_hv(binary, hv, **decoder_config) for binary, hv in zip(foreground, hv_maps)]
            for true_map, predicted_map in zip(batch["instance_map"].numpy(), predicted):
                totals += fast_binary_pq_stats(true_map, predicted_map)
    tp, fp, fn, sum_iou = totals
    denominator = tp + 0.5 * fp + 0.5 * fn
    return {
        "bPQ": float(sum_iou / denominator if denominator else 0.0),
        "DQ": float(tp / denominator if denominator else 0.0),
        "SQ": float(sum_iou / tp if tp else 0.0),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def main() -> None:
    import torch
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--learning-rates", default="1e-5,3e-5,1e-4")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--last-n-blocks", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--validation-batch-size",
        type=int,
        default=1,
        help="Validation inference batch size; keep at 1 to match run_cellvit_conic.py extraction",
    )
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0, help="Parallel workers for image/label/HV target preparation")
    parser.add_argument("--sampler", choices=["uniform", "minority", "source", "source_class"], default="uniform")
    parser.add_argument("--sampling-blend", type=float, default=0.5, help="Blend between uniform and equal-class-mass patch sampling")
    parser.add_argument("--augmentation", choices=["none", "color", "blur", "color_blur", "hed"], default="none")
    parser.add_argument("--color-jitter", type=float, default=0.1, help="Brightness/contrast/saturation factor range around 1")
    parser.add_argument("--blur-probability", type=float, default=0.3)
    parser.add_argument("--blur-radius-min", type=float, default=0.3)
    parser.add_argument("--blur-radius-max", type=float, default=1.0)
    parser.add_argument("--hed-probability", type=float, default=0.8)
    parser.add_argument("--hed-scale-jitter", type=float, default=0.1)
    parser.add_argument("--hed-offset-jitter", type=float, default=0.02)
    parser.add_argument("--directional-maps", action="store_true", help="Train H/V plus +45/-45 degree centroid maps")
    parser.add_argument("--map-head-lr-multiplier", type=float, default=10.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hv-weight", type=float, default=1.0)
    parser.add_argument("--hv-gradient-weight", type=float, default=1.0)
    parser.add_argument("--magnification", type=int, choices=[20, 40], default=20)
    parser.add_argument("--decoder-config", type=Path, default=None, help="Validation-selected HV decoder report/config JSON")
    parser.add_argument("--max-train-patches", type=int, default=None)
    parser.add_argument("--max-val-patches", type=int, default=None)
    parser.add_argument("--exclude-train-sources", default="", help="Comma-separated sources excluded from training")
    parser.add_argument("--validation-split", choices=["train", "val"], default="val")
    parser.add_argument("--validation-sources", default="", help="Optional comma-separated validation source filter")
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; LoRA segmentation is not practical on CPU")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    metadata = pd.read_csv(args.prepared / "metadata.csv")
    excluded_sources = {value.strip() for value in args.exclude_train_sources.split(",") if value.strip()}
    validation_sources = {value.strip() for value in args.validation_sources.split(",") if value.strip()}
    train_ids, val_ids = select_train_validation_ids(
        metadata,
        excluded_sources,
        args.validation_split,
        validation_sources,
    )
    if args.max_train_patches:
        train_ids = train_ids[: args.max_train_patches]
    if args.max_val_patches:
        val_ids = val_ids[: args.max_val_patches]

    model, mean, std = load_model(args.checkpoint, args.device)
    injected = inject_sam_lora(model, args.rank, args.alpha, args.dropout, args.last_n_blocks, True)
    map_head_parameters = []
    if args.directional_maps:
        expand_hv_head(model)
        map_head_parameters = set_directional_header_trainable(model)
    model.to(args.device)
    initial_adapter = {name: value.clone() for name, value in lora_state_dict(model).items()}
    initial_buffers = {name: value.detach().cpu().clone() for name, value in model.named_buffers()}
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    print(f"LoRA targets={len(injected)} trainable_parameters={sum(parameter.numel() for parameter in trainable):,}", flush=True)

    if not 0.0 <= args.color_jitter <= 1.0:
        raise ValueError("--color-jitter must be in [0, 1]")
    if not 0.0 <= args.blur_probability <= 1.0:
        raise ValueError("--blur-probability must be in [0, 1]")
    if not 0.0 <= args.blur_radius_min <= args.blur_radius_max:
        raise ValueError("blur radii must satisfy 0 <= min <= max")
    if not 0.0 <= args.hed_probability <= 1.0:
        raise ValueError("--hed-probability must be in [0, 1]")
    if not 0.0 <= args.hed_scale_jitter <= 1.0 or args.hed_offset_jitter < 0.0:
        raise ValueError("HED jitter magnitudes must be non-negative and scale jitter no larger than one")
    train_dataset = CoNICSegmentationDataset(
        args.prepared,
        train_ids,
        mean,
        std,
        augmentation=args.augmentation,
        color_jitter=args.color_jitter,
        blur_probability=args.blur_probability,
        blur_radius_min=args.blur_radius_min,
        blur_radius_max=args.blur_radius_max,
        hed_probability=args.hed_probability,
        hed_scale_jitter=args.hed_scale_jitter,
        hed_offset_jitter=args.hed_offset_jitter,
        directional_maps=args.directional_maps,
    )
    val_dataset = CoNICSegmentationDataset(args.prepared, val_ids, mean, std, directional_maps=args.directional_maps)
    loader_options = {
        "num_workers": args.num_workers,
        "persistent_workers": args.num_workers > 0,
        "pin_memory": args.device.startswith("cuda"),
    }
    # CellViT's thresholded HV decoder can magnify small kernel-level numeric
    # differences.  Validate with the same batch shape used by deployment so
    # LR/epoch selection is reproducible from persisted instance maps.
    val_loader = DataLoader(val_dataset, batch_size=args.validation_batch_size, shuffle=False, **loader_options)
    decoder_config = load_decoder_config(args.decoder_config)
    sampling_weights = None
    sampling_summary = {"sampler": args.sampler}
    train_rows = metadata.set_index("patch_id").loc[train_ids]
    counts_by_patch = train_rows[[f"count_{name}" for name in CLASS_NAMES]].to_numpy()
    if args.sampler == "minority":
        sampling_weights = minority_patch_weights(counts_by_patch, args.sampling_blend)
    elif args.sampler == "source":
        sampling_weights = source_patch_weights(train_rows.source.to_numpy(), args.sampling_blend)
    elif args.sampler == "source_class":
        sampling_weights = source_class_patch_weights(train_rows.source.to_numpy(), counts_by_patch)
    if sampling_weights is not None:
        probabilities = sampling_weights / sampling_weights.sum()
        sampling_summary.update(
            {
                "blend": args.sampling_blend,
                "min_weight": float(sampling_weights.min()),
                "median_weight": float(np.median(sampling_weights)),
                "p95_weight": float(np.quantile(sampling_weights, 0.95)),
                "max_weight": float(sampling_weights.max()),
                "effective_sample_size": float(1.0 / np.square(probabilities).sum()),
            }
        )
        print(f"patch sampling: {sampling_summary}", flush=True)
    baseline_validation = validation_binary_pq(model, val_loader, args.device, decoder_config)
    print(f"zero-adapter validation bPQ={baseline_validation['bPQ']:.4f}", flush=True)
    learning_rates = [float(value) for value in args.learning_rates.split(",") if value.strip()]
    curve_rows = []
    best_overall = None
    for learning_rate in learning_rates:
        model.load_state_dict(initial_adapter, strict=False)
        torch.manual_seed(args.seed)
        generator = torch.Generator().manual_seed(args.seed)
        if sampling_weights is None:
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator, **loader_options)
        else:
            sampler = torch.utils.data.WeightedRandomSampler(
                torch.as_tensor(sampling_weights, dtype=torch.double),
                num_samples=len(train_dataset),
                replacement=True,
                generator=generator,
            )
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, **loader_options)
        if args.directional_maps:
            map_head_ids = {id(parameter) for parameter in map_head_parameters}
            lora_parameters = [parameter for parameter in trainable if id(parameter) not in map_head_ids]
            optimizer = torch.optim.AdamW(
                [
                    {"params": lora_parameters, "lr": learning_rate},
                    {"params": map_head_parameters, "lr": learning_rate * args.map_head_lr_multiplier},
                ],
                weight_decay=args.weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=args.weight_decay)
        candidate_best = None
        stale_epochs = 0
        for epoch in range(1, args.epochs + 1):
            # This is adapter-only fine-tuning.  Leaving the whole model in
            # train mode silently updates frozen BatchNorm running statistics,
            # which are not part of a LoRA checkpoint and cannot be reproduced
            # after reloading the adapter for extraction.
            set_lora_train_mode(model)
            optimizer.zero_grad(set_to_none=True)
            sums = {"total": 0.0, "ce": 0.0, "dice": 0.0, "hv_mse": 0.0, "hv_gradient": 0.0, "batches": 0}
            for batch_index, batch in enumerate(train_loader, start=1):
                images = batch["image"].float().to(args.device)
                binary = batch["binary"].long().to(args.device)
                hv_target = batch["hv"].float().to(args.device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
                    output = model(images)
                    loss, components = segmentation_loss(output, binary, hv_target, args.hv_weight, args.hv_gradient_weight)
                    scaled_loss = loss / args.accumulation_steps
                scaled_loss.backward()
                if batch_index % args.accumulation_steps == 0 or batch_index == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                sums["total"] += float(loss.item())
                for name, value in components.items():
                    sums[name] += float(value.item())
                sums["batches"] += 1
            validation = validation_binary_pq(model, val_loader, args.device, decoder_config)
            row = {
                "learning_rate": learning_rate,
                "map_head_learning_rate": learning_rate * args.map_head_lr_multiplier if args.directional_maps else None,
                "epoch": epoch,
                **{f"train_{name}": value / sums["batches"] for name, value in sums.items() if name != "batches"},
                **{f"val_{name}": value for name, value in validation.items()},
            }
            curve_rows.append(row)
            if candidate_best is None or validation["bPQ"] > candidate_best["score"]:
                candidate_best = {
                    "score": validation["bPQ"],
                    "epoch": epoch,
                    "validation": validation,
                    "state": {name: value.clone() for name, value in lora_state_dict(model).items()},
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
            print(f"lr={learning_rate:g} epoch={epoch}/{args.epochs} loss={row['train_total']:.4f} val_bPQ={validation['bPQ']:.4f}", flush=True)
            if stale_epochs >= args.patience:
                break
        if best_overall is None or candidate_best["score"] > best_overall["score"]:
            best_overall = {**candidate_best, "learning_rate": learning_rate}

    model.load_state_dict(best_overall["state"], strict=False)
    changed_buffers = [
        name
        for name, value in model.named_buffers()
        if name in initial_buffers and not torch.equal(value.detach().cpu(), initial_buffers[name])
    ]
    if changed_buffers:
        raise RuntimeError(f"Frozen model buffers changed during LoRA training: {changed_buffers[:8]}")
    restored_validation = validation_binary_pq(model, val_loader, args.device, decoder_config)
    if abs(restored_validation["bPQ"] - best_overall["score"]) > 1e-12:
        raise RuntimeError(
            "Restored LoRA state does not reproduce its selected validation score: "
            f"{restored_validation['bPQ']:.8f} != {best_overall['score']:.8f}"
        )
    configuration = {
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": args.dropout,
        "last_n_blocks": args.last_n_blocks,
        "target_projection": True,
        "base_checkpoint": str(args.checkpoint),
        "selected_learning_rate": best_overall["learning_rate"],
        "selected_epoch": best_overall["epoch"],
        "selection_metric": "validation pooled binary PQ with CellViT HV postprocessing",
        "validation_precision": "float32 (matched to extraction)",
        "validation_batch_size": args.validation_batch_size,
        "frozen_base_buffers": True,
        "sampling": sampling_summary,
        "augmentation": {
            "name": args.augmentation,
            "color_jitter": args.color_jitter if args.augmentation in {"color", "color_blur"} else None,
            "blur_probability": args.blur_probability if args.augmentation in {"blur", "color_blur"} else None,
            "blur_radius": [args.blur_radius_min, args.blur_radius_max] if args.augmentation in {"blur", "color_blur"} else None,
            "hed_probability": args.hed_probability if args.augmentation == "hed" else None,
            "hed_scale_jitter": args.hed_scale_jitter if args.augmentation == "hed" else None,
            "hed_offset_jitter": args.hed_offset_jitter if args.augmentation == "hed" else None,
        },
        "data_filters": {
            "excluded_train_sources": sorted(excluded_sources),
            "validation_split": args.validation_split,
            "validation_sources": sorted(validation_sources),
            "n_train": len(train_ids),
            "n_validation": len(val_ids),
        },
        "directional_maps": {
            "enabled": args.directional_maps,
            "channels": ["horizontal", "vertical", "diagonal", "anti_diagonal"] if args.directional_maps else ["horizontal", "vertical"],
            "map_head_lr_multiplier": args.map_head_lr_multiplier if args.directional_maps else None,
            "trainable_scope": "LoRA plus final map header" if args.directional_maps else "LoRA only",
        },
        "validation": best_overall["validation"],
        "zero_adapter_validation": baseline_validation,
        "magnification": args.magnification,
        "decoder_config": decoder_config,
        "seed": args.seed,
    }
    save_lora_adapter(model, args.out, configuration)
    curve_path = args.out.with_name(f"{args.out.stem}_curve.csv")
    with curve_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)
    curve_path.with_suffix(".json").write_text(json.dumps(configuration, indent=2))
    print(f"selected lr={best_overall['learning_rate']:g} epoch={best_overall['epoch']} val_bPQ={best_overall['score']:.5f}")


if __name__ == "__main__":
    main()
