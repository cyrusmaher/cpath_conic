#!/usr/bin/env python3
"""Train the official CoNIC HoVer-Net architecture on our group-disjoint split.

The public CoNIC checkpoint is deliberately never accepted as initialization.
Only the ImageNet ResNet-50 backbone specified by the authors may be loaded.
Validation R2 and mPQ+ are computed with the same evaluator used elsewhere in
this project, allowing learning-rate and checkpoint selection without test use.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
import warnings
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

# imgaug 0.4.0 is the version pinned by the official CoNIC branch and still
# refers to NumPy aliases removed in NumPy 1.24.
if "bool" not in np.__dict__:
    np.bool = np.bool_  # type: ignore[attr-defined]
if "int" not in np.__dict__:
    np.int = int  # type: ignore[attr-defined]
if "float" not in np.__dict__:
    np.float = float  # type: ignore[attr-defined]

import imgaug as ia
from imgaug import augmenters as iaa
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
HOVERNET_ROOT = ROOT / "third_party" / "hover_net"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HOVERNET_ROOT))

from cpath_conic.constants import CLASS_NAMES, COUNT_COLUMNS
from cpath_conic.data import central_crop_counts
from cpath_conic.metrics import (
    binary_instance_segmentation_metrics,
    instance_type_confusion,
    multiclass_pq_plus,
    multiclass_r2,
)
from cpath_conic.sampling import source_class_patch_weights
from cpath_conic.stain import EmpiricalHEDTargetBank, hed_concentration, hed_stain_augmentation_array
from dataloader.augs import (
    add_to_brightness,
    add_to_contrast,
    add_to_hue,
    add_to_saturation,
    gaussian_blur,
    median_blur,
)
from misc.utils import cropping_center
from models.hovernet.net_desc import HoVerNetExt
from models.hovernet.targets import gen_targets
from models.hovernet.utils import dice_loss, mse_loss, xentropy_loss
from scripts.run_hovernet_control import instance_class_probabilities, process_prediction


OFFICIAL_COMMIT = "90441c092192e83b6ac6c2098f7927eb36be347c"
BACKBONE_ARCHITECTURES = ("resnet50", "seresnext101_32x4d")
warnings.filterwarnings("ignore", message="Only one label was provided to `remove_small_objects`.*")


def initialization_declaration(backbone_architecture: str) -> str:
    if backbone_architecture == "resnet50":
        # Preserve the declaration embedded in all existing E32 checkpoints.
        return "ImageNet ResNet-50 only; no CoNIC checkpoint"
    if backbone_architecture == "seresnext101_32x4d":
        return "ImageNet SE-ResNeXt-101 32x4d only; no CoNIC checkpoint"
    raise ValueError(f"unsupported backbone architecture: {backbone_architecture}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def capture_rng_state(generators: dict[str, torch.Generator]) -> dict:
    """Capture every RNG stream that can affect training or sampled patch order."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "generators": {name: generator.get_state() for name, generator in generators.items()},
    }


def restore_rng_state(state: dict, generators: dict[str, torch.Generator]) -> None:
    """Restore exact continuation state; refuse legacy checkpoints that would replay draws."""
    stored_generators = state.get("generators", {}) if state else {}
    if set(stored_generators) != set(generators):
        raise RuntimeError(
            "resume checkpoint lacks exact sampler/worker RNG state; restart this candidate from epoch 0 "
            "instead of replaying or changing its sampling/augmentation sequence"
        )
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    for name, generator in generators.items():
        generator.set_state(stored_generators[name])


def require_resumable_worker_policy(policy: str) -> None:
    """Reject a resume whose persistent worker-local augmentation state is unknowable."""
    if policy == "persistent":
        raise RuntimeError(
            "persistent-worker augmentation RNG cannot be resumed exactly; archive the partial run and "
            "restart this matched candidate from epoch 0, or use --worker-rng-policy epoch_reseed for a "
            "new experiment family"
        )
    if policy != "epoch_reseed":
        raise ValueError(f"unknown worker RNG policy: {policy}")


def sampling_mode(
    source_fraction: float,
    class_fraction: float,
    uniform_replacement: bool = False,
) -> str:
    """Name the patch-draw mechanism so replacement is an explicit intervention."""
    weighted = source_fraction > 0 or class_fraction > 0
    if uniform_replacement and weighted:
        raise ValueError("uniform replacement control cannot also use source/class weighting")
    if uniform_replacement:
        return "uniform_with_replacement"
    if weighted:
        return "weighted_with_replacement"
    return "without_replacement"


class PreparedHoverDataset(Dataset):
    """Prepared PNG/NPY adapter preserving the official CoNIC augmentations."""

    def __init__(
        self,
        prepared: Path,
        rows: pd.DataFrame,
        train: bool,
        seed: int,
        hed_probability: float = 0.0,
        hed_target_concentrations: np.ndarray | None = None,
        hed_target_sources: np.ndarray | None = None,
        hed_target_jitter: float = 0.05,
        hed_tail_expansion: float = 0.1,
        hed_strength_min: float = 0.25,
        hed_strength_max: float = 1.0,
        hed_scale_min: float = 0.5,
        hed_scale_max: float = 2.0,
    ):
        self.prepared = prepared
        self.rows = rows.reset_index(drop=True)
        self.train = train
        self.seed = seed
        self.hed_probability = hed_probability
        self.hed_target_bank = None
        self.hed_strength_min = hed_strength_min
        self.hed_strength_max = hed_strength_max
        self.hed_scale_min = hed_scale_min
        self.hed_scale_max = hed_scale_max
        if hed_probability > 0:
            if hed_target_concentrations is None or hed_target_sources is None:
                raise ValueError("empirical HED augmentation requires target concentrations and sources")
            self.hed_target_bank = EmpiricalHEDTargetBank(
                hed_target_concentrations,
                hed_target_sources,
                jitter=hed_target_jitter,
                tail_expansion=hed_tail_expansion,
            )
        self.rng = np.random.default_rng(seed)
        self.shape_augs = None
        self.input_augs = None
        self.setup_augmentor(0, seed)

    def __len__(self) -> int:
        return len(self.rows)

    def setup_augmentor(self, worker_id: int, seed: int) -> None:
        rng = int((seed + worker_id) % (2**32 - 1))
        self.rng = np.random.default_rng(rng)
        if self.train:
            shape_augs = [
                iaa.CropToFixedSize(256, 256, position="center"),
                iaa.Fliplr(0.5, seed=rng),
                iaa.Flipud(0.5, seed=rng),
            ]
            input_augs = [
                iaa.OneOf(
                    [
                        iaa.Lambda(seed=rng, func_images=lambda *args: gaussian_blur(*args, max_ksize=3)),
                        iaa.Lambda(seed=rng, func_images=lambda *args: median_blur(*args, max_ksize=3)),
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                    ]
                ),
                iaa.Sequential(
                    [
                        iaa.Lambda(seed=rng, func_images=lambda *args: add_to_hue(*args, range=(-8, 8))),
                        iaa.Lambda(seed=rng, func_images=lambda *args: add_to_saturation(*args, range=(-0.2, 0.2))),
                        iaa.Lambda(seed=rng, func_images=lambda *args: add_to_brightness(*args, range=(-26, 26))),
                        iaa.Lambda(seed=rng, func_images=lambda *args: add_to_contrast(*args, range=(0.75, 1.25))),
                    ],
                    random_order=True,
                ),
            ]
        else:
            shape_augs = [iaa.CropToFixedSize(256, 256, position="center")]
            input_augs = []
        self.shape_augs = iaa.Sequential(shape_augs)
        self.input_augs = iaa.Sequential(input_augs)

    def __getitem__(self, index: int) -> dict[str, np.ndarray | int]:
        patch_id = int(self.rows.iloc[index].patch_id)
        image = np.asarray(Image.open(self.prepared / "images" / f"{patch_id:05d}.png").convert("RGB"), dtype=np.uint8)
        if self.train and self.hed_probability > 0:
            target_concentration, _ = self.hed_target_bank.sample(self.rng)
            image = hed_stain_augmentation_array(
                image,
                self.rng,
                self.hed_probability,
                target_concentration,
                self.hed_strength_min,
                self.hed_strength_max,
                self.hed_scale_min,
                self.hed_scale_max,
            )
        label = np.load(self.prepared / "labels" / f"{patch_id:05d}.npy", allow_pickle=True).item()
        instance = np.asarray(label["inst_map"], dtype=np.int32)
        type_map = np.asarray(label["class_map"], dtype=np.int32)
        annotation = np.stack([instance, type_map], axis=-1)

        shape_augs = self.shape_augs.to_deterministic()
        image = shape_augs.augment_image(image)
        annotation = shape_augs.augment_image(annotation)
        if self.train:
            image = self.input_augs.to_deterministic().augment_image(image)

        image = cropping_center(image, [256, 256])
        instance = annotation[..., 0]
        targets = gen_targets(instance, [256, 256])
        return {
            "patch_id": patch_id,
            "img": image,
            "inst_map": cropping_center(instance, [256, 256]).astype(np.int64),
            "np_map": targets["np_map"].astype(np.int64),
            "hv_map": targets["hv_map"].astype(np.float32),
            "tp_map": cropping_center(annotation[..., 1], [256, 256]).astype(np.int64),
        }


def worker_init(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    worker_seed = int(info.seed % (2**32 - 1))
    info.dataset.setup_augmentor(worker_id, worker_seed)


def msge_loss(true: torch.Tensor, pred: torch.Tensor, focus: torch.Tensor) -> torch.Tensor:
    """Official HV gradient loss with device-independent Sobel construction."""
    device = pred.device
    axis = torch.arange(-2, 3, dtype=torch.float32, device=device)
    h, v = torch.meshgrid(axis, axis, indexing="ij")
    denominator = h * h + v * v + 1.0e-15
    kernel_h = (h / denominator).view(1, 1, 5, 5)
    kernel_v = (v / denominator).view(1, 1, 5, 5)

    def gradient(hv: torch.Tensor) -> torch.Tensor:
        h_ch = F.conv2d(hv[..., 0].unsqueeze(1), kernel_h, padding=2)
        v_ch = F.conv2d(hv[..., 1].unsqueeze(1), kernel_v, padding=2)
        return torch.cat([h_ch, v_ch], dim=1).permute(0, 2, 3, 1).contiguous()

    mask = focus[..., None].float().expand(-1, -1, -1, 2)
    difference = gradient(pred) - gradient(true)
    return (mask * difference.square()).sum() / (mask.sum() + 1.0e-8)


def instance_equalized_pixel_weights(
    instance_map: torch.Tensor,
    blend: float,
    max_weight: float | None = None,
) -> torch.Tensor:
    """Preserve foreground mass while distributing it across instances.

    ``max_weight`` optionally caps the final blended per-pixel weight. The
    remaining foreground weights are rescaled so every patch still has mean
    foreground weight one; background pixels always remain exactly one.
    """
    if not 0.0 <= blend <= 1.0:
        raise ValueError(f"instance loss blend must be in [0, 1], got {blend}")
    if max_weight is not None and max_weight < 1.0:
        raise ValueError(f"instance loss max weight must be at least 1, got {max_weight}")
    weights = torch.ones_like(instance_map, dtype=torch.float32)
    if blend == 0.0:
        return weights
    for batch_index in range(instance_map.shape[0]):
        ids, inverse, counts = torch.unique(
            instance_map[batch_index], sorted=True, return_inverse=True, return_counts=True
        )
        foreground_ids = ids > 0
        instance_count = int(foreground_ids.sum())
        if instance_count == 0:
            continue
        foreground_pixels = counts[foreground_ids].sum().float()
        equal_mass = foreground_pixels / float(instance_count)
        id_weights = torch.ones_like(counts, dtype=torch.float32)
        foreground_counts = counts[foreground_ids].float()
        foreground_weights = equal_mass / foreground_counts
        foreground_weights = 1.0 + blend * (foreground_weights - 1.0)
        if max_weight is not None:
            # Solve sum_i count_i * min(cap, scale * weight_i) = foreground_pixels.
            # The vectorized water-filling candidates avoid a per-step host/device
            # synchronization loop during training.
            sorted_weights, order = torch.sort(foreground_weights)
            sorted_counts = foreground_counts[order]
            weighted_prefix = torch.cumsum(sorted_weights * sorted_counts, dim=0)
            count_prefix = torch.cumsum(sorted_counts, dim=0)
            total_count = count_prefix[-1]
            saturated_count = total_count - count_prefix
            scales = (total_count - float(max_weight) * saturated_count) / weighted_prefix.clamp_min(1.0e-12)
            upper_ok = scales * sorted_weights <= float(max_weight) + 1.0e-6
            next_weights = torch.cat([sorted_weights[1:], sorted_weights.new_tensor([float("inf")])])
            lower_ok = scales * next_weights >= float(max_weight) - 1.0e-6
            valid = upper_ok & lower_ok & (scales >= 0)
            candidate_index = torch.nonzero(valid, as_tuple=False)
            if candidate_index.numel() == 0:
                raise RuntimeError("could not construct mass-preserving capped instance weights")
            scale = scales[candidate_index[0, 0]]
            foreground_weights = torch.clamp(foreground_weights * scale, max=float(max_weight))
        id_weights[foreground_ids] = foreground_weights
        equalized = id_weights[inverse].reshape_as(instance_map[batch_index])
        weights[batch_index] = equalized
    return weights


def weighted_xentropy_loss(true: torch.Tensor, pred: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    epsilon = 1.0e-7
    normalized = pred / torch.sum(pred, -1, keepdim=True)
    normalized = torch.clamp(normalized, epsilon, 1.0 - epsilon)
    per_pixel = -torch.sum(true * torch.log(normalized), dim=-1)
    return (per_pixel * weights).sum() / weights.sum().clamp_min(1.0e-8)


def complement_frequency_type_weights(
    foreground_counts: np.ndarray,
    rho: float,
    background_weight: float = 1.0,
) -> np.ndarray:
    """Return transparent, fixed HoVer type weights from training nuclei only.

    Foreground weights follow ``(1 - frequency) ** rho`` and are normalized to
    mean one across the six scored classes.  Background is deliberately kept
    separate because it has no nucleus-count analogue in the metadata.
    """
    counts = np.asarray(foreground_counts, dtype=np.float64)
    if counts.shape != (6,) or np.any(counts < 0):
        raise ValueError("foreground type counts must contain six non-negative values")
    if rho < 0 or background_weight <= 0:
        raise ValueError("rho must be non-negative and background weight positive")
    frequencies = counts / max(float(counts.sum()), 1.0)
    foreground = np.power(1.0 - frequencies, rho)
    foreground /= max(float(foreground.mean()), 1.0e-12)
    return np.concatenate([[background_weight], foreground]).astype(np.float32)


def weighted_focal_xentropy_loss(
    true: torch.Tensor,
    pred: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float,
    label_smoothing: float = 0.0,
    pixel_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Multiclass focal CE on probabilities with auditable class/pixel weights."""
    if gamma < 0 or not 0 <= label_smoothing < 1:
        raise ValueError("focal gamma must be non-negative and label smoothing in [0, 1)")
    if class_weights.ndim != 1 or class_weights.shape[0] != pred.shape[-1]:
        raise ValueError("class weights must match the probability channel count")
    epsilon = 1.0e-7
    normalized = pred / torch.sum(pred, -1, keepdim=True)
    normalized = torch.clamp(normalized, epsilon, 1.0 - epsilon)
    hard_target = true.argmax(dim=-1)
    target_probability = normalized.gather(-1, hard_target[..., None]).squeeze(-1)
    smoothed = true * (1.0 - label_smoothing) + label_smoothing / float(true.shape[-1])
    weighted_log_probability = smoothed * class_weights * torch.log(normalized)
    per_pixel = -weighted_log_probability.sum(dim=-1) * (1.0 - target_probability).pow(gamma)
    target_weight = class_weights[hard_target]
    if pixel_weights is not None:
        per_pixel = per_pixel * pixel_weights
        target_weight = target_weight * pixel_weights
    return per_pixel.sum() / target_weight.sum().clamp_min(1.0e-8)


def weighted_mse_loss(true: torch.Tensor, pred: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    per_pixel = (pred - true).square().mean(dim=-1)
    return (per_pixel * weights).sum() / weights.sum().clamp_min(1.0e-8)


def instance_pooled_type_loss(
    true_instances: torch.Tensor,
    true_types: torch.Tensor,
    predicted_type_probabilities: torch.Tensor,
) -> torch.Tensor:
    """One type cross-entropy term per GT nucleus after pooling its pixels."""
    losses = []
    num_types = predicted_type_probabilities.shape[-1]
    for batch_index in range(true_instances.shape[0]):
        instance_flat = true_instances[batch_index].reshape(-1)
        type_flat = true_types[batch_index].reshape(-1)
        probabilities = predicted_type_probabilities[batch_index].reshape(-1, num_types)
        instance_ids, inverse = torch.unique(instance_flat, sorted=True, return_inverse=True)
        count = torch.bincount(inverse, minlength=len(instance_ids)).to(probabilities.dtype)
        probability_sum = torch.zeros(
            (len(instance_ids), num_types), dtype=probabilities.dtype, device=probabilities.device
        )
        probability_sum.index_add_(0, inverse, probabilities)
        pooled = probability_sum / count[:, None].clamp_min(1.0)

        type_votes = torch.zeros_like(probability_sum)
        type_votes.index_add_(0, inverse, F.one_hot(type_flat, num_classes=num_types).to(probabilities.dtype))
        target = type_votes.argmax(dim=1)
        valid = (instance_ids > 0) & (target > 0)
        if valid.any():
            selected = pooled[valid].gather(1, target[valid, None]).squeeze(1)
            losses.append(-torch.log(selected.clamp_min(1.0e-7)))
    if not losses:
        return predicted_type_probabilities.sum() * 0.0
    return torch.cat(losses).mean()


@torch.inference_mode()
def instance_type_diagnostic_sums(
    true_instances: torch.Tensor,
    true_types: torch.Tensor,
    predicted_type_probabilities: torch.Tensor,
) -> dict[str, float]:
    """Return equal-nucleus type consistency sums for validation diagnostics."""
    totals = {
        "nuclei": 0.0,
        "nll": 0.0,
        "target_probability": 0.0,
        "entropy": 0.0,
        "mean_pixel_entropy": 0.0,
        "spatial_js_disagreement": 0.0,
        "pixel_accuracy": 0.0,
    }
    num_types = predicted_type_probabilities.shape[-1]
    entropy_scale = float(np.log(max(num_types, 2)))
    for batch_index in range(true_instances.shape[0]):
        instance_flat = true_instances[batch_index].reshape(-1)
        type_flat = true_types[batch_index].reshape(-1)
        probabilities = predicted_type_probabilities[batch_index].reshape(-1, num_types)
        instance_ids, inverse = torch.unique(instance_flat, sorted=True, return_inverse=True)
        count = torch.bincount(inverse, minlength=len(instance_ids)).to(probabilities.dtype)
        probability_sum = torch.zeros(
            (len(instance_ids), num_types), dtype=probabilities.dtype, device=probabilities.device
        )
        probability_sum.index_add_(0, inverse, probabilities)
        pooled = probability_sum / count[:, None].clamp_min(1.0)
        type_votes = torch.zeros_like(probability_sum)
        type_votes.index_add_(0, inverse, F.one_hot(type_flat, num_classes=num_types).to(probabilities.dtype))
        target = type_votes.argmax(dim=1)
        valid = (instance_ids > 0) & (target > 0)
        if not valid.any():
            continue
        selected = pooled[valid]
        target_probability = selected.gather(1, target[valid, None]).squeeze(1).clamp_min(1.0e-7)
        pixel_correct = (probabilities.argmax(dim=1) == type_flat).to(probabilities.dtype)
        correct_sum = torch.zeros(len(instance_ids), dtype=probabilities.dtype, device=probabilities.device)
        correct_sum.index_add_(0, inverse, pixel_correct)
        per_instance_accuracy = correct_sum / count.clamp_min(1.0)
        entropy = -(selected.clamp_min(1.0e-7) * torch.log(selected.clamp_min(1.0e-7))).sum(dim=1) / entropy_scale
        pixel_entropy = -(
            probabilities.clamp_min(1.0e-7) * torch.log(probabilities.clamp_min(1.0e-7))
        ).sum(dim=1) / entropy_scale
        pixel_entropy_sum = torch.zeros(
            len(instance_ids), dtype=probabilities.dtype, device=probabilities.device
        )
        pixel_entropy_sum.index_add_(0, inverse, pixel_entropy)
        mean_pixel_entropy = pixel_entropy_sum / count.clamp_min(1.0)
        # Generalized Jensen-Shannon divergence across the pixels of a
        # nucleus: H(mean pixel probability) - mean H(pixel probability).
        # Unlike pooled entropy alone, this is zero when all pixels agree and
        # positive when contradictory pixel predictions average to the same
        # pooled class vector. Normalize by log(K) to keep it in [0, 1].
        spatial_js = (entropy - mean_pixel_entropy[valid]).clamp_min(0.0)
        totals["nuclei"] += float(valid.sum().cpu())
        totals["nll"] += float((-torch.log(target_probability)).sum().cpu())
        totals["target_probability"] += float(target_probability.sum().cpu())
        totals["entropy"] += float(entropy.sum().cpu())
        totals["mean_pixel_entropy"] += float(mean_pixel_entropy[valid].sum().cpu())
        totals["spatial_js_disagreement"] += float(spatial_js.sum().cpu())
        totals["pixel_accuracy"] += float(per_instance_accuracy[valid].sum().cpu())
    return totals


def batch_loss(
    model: HoVerNetExt,
    batch: dict,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    instance_loss_blend: float = 0.0,
    instance_loss_max_weight: float | None = None,
    instance_type_loss_weight: float = 0.0,
    type_class_weights: torch.Tensor | None = None,
    type_focal_gamma: float = 0.0,
    type_label_smoothing: float = 0.0,
    return_foreground: bool = False,
) -> tuple[torch.Tensor, dict, torch.Tensor | None]:
    images = batch["img"].permute(0, 3, 1, 2).contiguous().float().to(device, non_blocking=True)
    true_np = batch["np_map"].long().to(device, non_blocking=True)
    true_hv = batch["hv_map"].float().to(device, non_blocking=True)
    true_tp = batch["tp_map"].long().to(device, non_blocking=True)
    true_instances = batch["inst_map"].long().to(device, non_blocking=True)
    true_np_onehot = F.one_hot(true_np, num_classes=2).float()
    true_tp_onehot = F.one_hot(true_tp, num_classes=7).float()

    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
        output = model(images)
        pred = OrderedDict((key, value.permute(0, 2, 3, 1).contiguous()) for key, value in output.items())
        pred["np"] = F.softmax(pred["np"].float(), dim=-1)
        pred["tp"] = F.softmax(pred["tp"].float(), dim=-1)
        pixel_weights = instance_equalized_pixel_weights(
            true_instances,
            instance_loss_blend,
            max_weight=instance_loss_max_weight,
        )
        if instance_loss_blend > 0:
            np_ce = weighted_xentropy_loss(true_np_onehot, pred["np"], pixel_weights)
            hv_mse = weighted_mse_loss(true_hv, pred["hv"].float(), pixel_weights)
            hv_msge = msge_loss(true_hv, pred["hv"].float(), true_np_onehot[..., 1] * pixel_weights)
        else:
            np_ce = xentropy_loss(true_np_onehot, pred["np"])
            hv_mse = mse_loss(true_hv, pred["hv"].float())
            hv_msge = msge_loss(true_hv, pred["hv"].float(), true_np_onehot[..., 1])
        use_type_focal = type_class_weights is not None and (
            type_focal_gamma > 0 or type_label_smoothing > 0 or not torch.all(type_class_weights == 1)
        )
        if use_type_focal:
            tp_ce = weighted_focal_xentropy_loss(
                true_tp_onehot,
                pred["tp"],
                type_class_weights,
                gamma=type_focal_gamma,
                label_smoothing=type_label_smoothing,
                pixel_weights=pixel_weights if instance_loss_blend > 0 else None,
            )
        elif instance_loss_blend > 0:
            tp_ce = weighted_xentropy_loss(true_tp_onehot, pred["tp"], pixel_weights)
        else:
            tp_ce = xentropy_loss(true_tp_onehot, pred["tp"])
        terms = {
            "np_ce": np_ce,
            "np_dice": dice_loss(true_np_onehot, pred["np"]),
            "hv_mse": hv_mse,
            "hv_msge": hv_msge,
            "tp_ce": tp_ce,
            "tp_dice": dice_loss(true_tp_onehot, pred["tp"]),
        }
        if instance_type_loss_weight > 0:
            terms["tp_instance_ce"] = instance_type_loss_weight * instance_pooled_type_loss(
                true_instances, true_tp, pred["tp"]
            )
        total = sum(terms.values())
    foreground = pred["np"].argmax(-1).detach().cpu() if return_foreground else None
    return total, {key: float(value.detach().cpu()) for key, value in terms.items()}, foreground


@torch.inference_mode()
def validate_loss(
    model: HoVerNetExt,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    instance_loss_blend: float = 0.0,
    instance_loss_max_weight: float | None = None,
    instance_type_loss_weight: float = 0.0,
    type_class_weights: torch.Tensor | None = None,
    type_focal_gamma: float = 0.0,
    type_label_smoothing: float = 0.0,
) -> dict:
    model.eval()
    totals: dict[str, float] = {}
    samples = 0
    intersection = 0
    denominator = 0
    for batch in loader:
        loss, terms, foreground = batch_loss(
            model,
            batch,
            device,
            amp_dtype,
            instance_loss_blend=instance_loss_blend,
            instance_loss_max_weight=instance_loss_max_weight,
            instance_type_loss_weight=instance_type_loss_weight,
            type_class_weights=type_class_weights,
            type_focal_gamma=type_focal_gamma,
            type_label_smoothing=type_label_smoothing,
            return_foreground=True,
        )
        size = len(batch["patch_id"])
        samples += size
        totals["loss"] = totals.get("loss", 0.0) + float(loss.detach().cpu()) * size
        for key, value in terms.items():
            totals[key] = totals.get(key, 0.0) + value * size
        truth = batch["np_map"]
        intersection += int(((foreground == 1) & (truth == 1)).sum())
        denominator += int((foreground == 1).sum() + (truth == 1).sum())
    result = {f"val_{key}": value / max(samples, 1) for key, value in totals.items()}
    result["val_np_dice_metric"] = 2.0 * intersection / max(denominator, 1)
    return result


def count_error_stats(true_counts: np.ndarray, predicted_counts: np.ndarray) -> dict:
    """Summarize directional count error and tail prevalence over patch×class points."""
    truth = np.asarray(true_counts, dtype=np.float64)
    predicted = np.asarray(predicted_counts, dtype=np.float64)
    if truth.shape != predicted.shape or truth.ndim != 2:
        raise ValueError("count arrays must have matching patches-by-classes shapes")
    residual = predicted - truth
    values = residual.ravel()
    absolute = np.abs(values)
    if not len(values):
        raise ValueError("count arrays cannot be empty")
    return {
        "points": int(len(values)),
        "mean_signed_error": float(values.mean()),
        "MAE": float(absolute.mean()),
        "under_fraction": float(np.mean(values < 0)),
        "exact_fraction": float(np.mean(values == 0)),
        "over_fraction": float(np.mean(values > 0)),
        **{f"absolute_error_gt_{threshold}_fraction": float(np.mean(absolute > threshold)) for threshold in (2, 5, 10, 20)},
        **{f"under_error_lt_minus_{threshold}_fraction": float(np.mean(values < -threshold)) for threshold in (2, 5, 10, 20)},
        **{f"over_error_gt_{threshold}_fraction": float(np.mean(values > threshold)) for threshold in (2, 5, 10, 20)},
    }


def zero_truth_count_stats(true_counts: np.ndarray, predicted_counts: np.ndarray) -> dict:
    """Summarize false-count prevalence where a patch's class truth is exactly zero."""
    truth = np.asarray(true_counts, dtype=np.float64)
    predicted = np.asarray(predicted_counts, dtype=np.float64)
    if truth.shape != predicted.shape or truth.ndim != 2:
        raise ValueError("count arrays must have matching patches-by-classes shapes")

    def summarize(mask: np.ndarray, values: np.ndarray) -> dict:
        selected = values[mask]
        if not len(selected):
            return {
                "support": 0,
                "nonzero_fraction": None,
                "over_5_fraction": None,
                "over_10_fraction": None,
                "over_20_fraction": None,
                "mean_prediction": None,
                "max_prediction": None,
            }
        return {
            "support": int(len(selected)),
            "nonzero_fraction": float(np.mean(selected > 0)),
            "over_5_fraction": float(np.mean(selected > 5)),
            "over_10_fraction": float(np.mean(selected > 10)),
            "over_20_fraction": float(np.mean(selected > 20)),
            "mean_prediction": float(selected.mean()),
            "max_prediction": float(selected.max()),
        }

    zero = truth == 0
    return {
        "all_zero_truth_points": summarize(zero, predicted),
        "per_class": {
            name: summarize(zero[:, index], predicted[:, index])
            for index, name in enumerate(CLASS_NAMES)
        },
    }


def sampling_exposure_summary(sampled_patch_ids: list[int], train_rows: pd.DataFrame) -> dict:
    """Describe the realized replacement-sampling budget for one optimizer epoch."""
    ids = np.asarray(sampled_patch_ids, dtype=np.int64)
    if not len(ids):
        raise ValueError("sampled patch IDs cannot be empty")
    indexed = train_rows.set_index("patch_id")
    if not set(map(int, ids)).issubset(set(map(int, indexed.index))):
        raise ValueError("sampled IDs include patches outside the training manifest")
    sampled = indexed.loc[ids]
    counts = sampled[COUNT_COLUMNS].to_numpy(dtype=np.float64)
    nucleus_totals = counts.sum(axis=0)
    all_nuclei = float(nucleus_totals.sum())
    source_counts = sampled.source.astype(str).value_counts()
    return {
        "draws": int(len(ids)),
        "draw_sequence_sha256": hashlib.sha256(ids.astype("<i8", copy=False).tobytes()).hexdigest(),
        "unique_patches": int(len(np.unique(ids))),
        "unique_patch_fraction": float(len(np.unique(ids)) / len(ids)),
        "source_draw_fraction": {
            str(source): float(source_counts.get(str(source), 0) / len(ids))
            for source in sorted(train_rows.source.astype(str).unique())
        },
        "class_positive_patch_fraction": {
            name: float(np.mean(counts[:, index] > 0)) for index, name in enumerate(CLASS_NAMES)
        },
        "nucleus_class_fraction": {
            name: float(nucleus_totals[index] / all_nuclei) if all_nuclei else 0.0
            for index, name in enumerate(CLASS_NAMES)
        },
        "mean_nuclei_per_draw": {
            name: float(counts[:, index].mean()) for index, name in enumerate(CLASS_NAMES)
        },
    }


@torch.inference_mode()
def validation_leaderboard_metrics(
    model: HoVerNetExt,
    loader: DataLoader,
    prepared: Path,
    metadata: pd.DataFrame,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    artifact_path: Path | None = None,
) -> dict:
    model.eval()
    patch_ids: list[int] = []
    predictions: list[np.ndarray] = []
    pred_counts: list[np.ndarray] = []
    probability_patch_ids: list[np.ndarray] = []
    probability_instance_ids: list[np.ndarray] = []
    probability_values: list[np.ndarray] = []
    type_diagnostic_totals = {
        "nuclei": 0.0,
        "nll": 0.0,
        "target_probability": 0.0,
        "entropy": 0.0,
        "mean_pixel_entropy": 0.0,
        "spatial_js_disagreement": 0.0,
        "pixel_accuracy": 0.0,
    }
    for batch in loader:
        images = batch["img"].permute(0, 3, 1, 2).contiguous().float().to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            output = model(images)
        np_prob = torch.softmax(output["np"].permute(0, 2, 3, 1).float(), dim=-1)[..., 1].cpu().numpy()
        hv_maps = output["hv"].permute(0, 2, 3, 1).float().cpu().numpy()
        type_probabilities = torch.softmax(output["tp"].float(), dim=1).permute(0, 2, 3, 1).contiguous()
        diagnostics = instance_type_diagnostic_sums(
            batch["inst_map"].to(device, non_blocking=True),
            batch["tp_map"].to(device, non_blocking=True),
            type_probabilities,
        )
        for key, value in diagnostics.items():
            type_diagnostic_totals[key] += value
        type_maps = type_probabilities.argmax(-1).cpu().numpy()
        type_probability_maps = type_probabilities.cpu().numpy() if artifact_path is not None else None
        for offset, patch_id in enumerate(batch["patch_id"].tolist()):
            instances, classes = process_prediction(np_prob[offset], hv_maps[offset], type_maps[offset])
            predictions.append(np.stack([instances, classes], axis=-1))
            pred_counts.append(central_crop_counts(instances, classes))
            patch_ids.append(int(patch_id))
            if type_probability_maps is not None:
                instance_ids, class_probs = instance_class_probabilities(
                    instances,
                    type_probability_maps[offset],
                )
                probability_patch_ids.append(np.full(len(instance_ids), int(patch_id), dtype=np.int32))
                probability_instance_ids.append(instance_ids)
                probability_values.append(class_probs)

    order = np.argsort(patch_ids)
    patch_ids_array = np.asarray(patch_ids, dtype=np.int32)[order]
    predicted = np.asarray(predictions, dtype=np.int32)[order]
    count_array = np.asarray(pred_counts, dtype=np.int32)[order]
    truth = np.zeros_like(predicted)
    for index, patch_id in enumerate(patch_ids_array):
        label = np.load(prepared / "labels" / f"{int(patch_id):05d}.npy", allow_pickle=True).item()
        truth[index, ..., 0] = label["inst_map"]
        truth[index, ..., 1] = label["class_map"]
    pq = multiclass_pq_plus(truth, predicted)
    segmentation = binary_instance_segmentation_metrics(truth, predicted)
    type_confusion = instance_type_confusion(truth, predicted)
    rows = metadata.set_index("patch_id").loc[patch_ids_array]
    true_counts = rows[COUNT_COLUMNS].copy()
    true_counts.columns = CLASS_NAMES
    predicted_counts = pd.DataFrame(count_array, columns=CLASS_NAMES)
    r2 = multiclass_r2(true_counts.reset_index(drop=True), predicted_counts)
    true_array = true_counts.to_numpy(dtype=np.float64)
    predicted_array = predicted_counts.to_numpy(dtype=np.float64)
    residual = predicted_array - true_array
    true_total = true_array.sum(axis=0)
    predicted_total = predicted_array.sum(axis=0)
    true_mean = true_array.mean(axis=0)
    per_class_sst = np.square(true_array - true_mean).sum(axis=0)
    per_class_sse = np.square(residual).sum(axis=0)
    diagnostic_nuclei = max(type_diagnostic_totals["nuclei"], 1.0)
    source_values = rows.source.astype(str).to_numpy()
    per_source = {}
    for source in sorted(np.unique(source_values)):
        mask = source_values == source
        source_true = pd.DataFrame(true_array[mask], columns=CLASS_NAMES)
        source_predicted = pd.DataFrame(predicted_array[mask], columns=CLASS_NAMES)
        source_r2 = multiclass_r2(source_true, source_predicted)
        source_pq = multiclass_pq_plus(truth[mask], predicted[mask])
        source_type_confusion = instance_type_confusion(truth[mask], predicted[mask])
        source_residual = predicted_array[mask] - true_array[mask]
        per_source[source] = {
            "patches": int(mask.sum()),
            "R2": float(source_r2["R2"]),
            "mPQ+": float(source_pq["mPQ+"]),
            "mDQ+": float(source_pq["mDQ+"]),
            "mSQ+": float(source_pq["mSQ+"]),
            "count_error": count_error_stats(true_array[mask], predicted_array[mask]),
            "zero_truth_overcount": zero_truth_count_stats(true_array[mask], predicted_array[mask]),
            "per_class_R2": source_r2["per_class"],
            "per_class_SSE": {
                name: float(np.square(source_residual[:, index]).sum()) for index, name in enumerate(CLASS_NAMES)
            },
            "per_class_signed_error": {
                name: float(source_residual[:, index].mean()) for index, name in enumerate(CLASS_NAMES)
            },
            "per_class_MAE": {
                name: float(np.abs(source_residual[:, index]).mean()) for index, name in enumerate(CLASS_NAMES)
            },
            "per_class_PQ": {name: values["pq"] for name, values in source_pq["per_class"].items()},
            "per_class_DQ": {name: values["dq"] for name, values in source_pq["per_class"].items()},
            "per_class_SQ": {name: values["sq"] for name, values in source_pq["per_class"].items()},
            "per_class_PQ_stats": {
                name: {key: values[key] for key in ("tp", "fp", "fn", "sum_iou")}
                for name, values in source_pq["per_class"].items()
            },
            "instance_type_confusion": source_type_confusion,
        }
    if artifact_path is not None:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        flat_probability_patch_ids = (
            np.concatenate(probability_patch_ids) if probability_patch_ids else np.empty(0, dtype=np.int32)
        )
        flat_probability_instance_ids = (
            np.concatenate(probability_instance_ids) if probability_instance_ids else np.empty(0, dtype=np.int32)
        )
        flat_probability_values = (
            np.concatenate(probability_values)
            if probability_values
            else np.empty((0, len(CLASS_NAMES)), dtype=np.float32)
        )
        probability_order = np.lexsort((flat_probability_instance_ids, flat_probability_patch_ids))
        np.savez_compressed(
            artifact_path,
            patch_ids=patch_ids_array,
            predictions=predicted,
            predicted_counts=count_array,
            probability_patch_ids=flat_probability_patch_ids[probability_order],
            probability_instance_ids=flat_probability_instance_ids[probability_order],
            class_probs=flat_probability_values[probability_order],
        )
    return {
        "val_R2": float(r2["R2"]),
        "val_mPQ+": float(pq["mPQ+"]),
        "val_mDQ+": float(pq["mDQ+"]),
        "val_mSQ+": float(pq["mSQ+"]),
        "val_gt_instance_type_nuclei": int(type_diagnostic_totals["nuclei"]),
        "val_gt_instance_type_nll": float(type_diagnostic_totals["nll"] / diagnostic_nuclei),
        "val_gt_instance_type_target_probability": float(type_diagnostic_totals["target_probability"] / diagnostic_nuclei),
        "val_gt_instance_type_entropy": float(type_diagnostic_totals["entropy"] / diagnostic_nuclei),
        "val_gt_instance_type_mean_pixel_entropy": float(type_diagnostic_totals["mean_pixel_entropy"] / diagnostic_nuclei),
        "val_gt_instance_type_spatial_js_disagreement": float(type_diagnostic_totals["spatial_js_disagreement"] / diagnostic_nuclei),
        "val_gt_instance_pixel_type_accuracy": float(type_diagnostic_totals["pixel_accuracy"] / diagnostic_nuclei),
        "val_instance_type_confusion": type_confusion,
        **{f"val_{key}": value for key, value in segmentation.items()},
        "val_per_class_R2": r2["per_class"],
        "val_per_class_PQ": {key: value["pq"] for key, value in pq["per_class"].items()},
        "val_per_class_DQ": {key: value["dq"] for key, value in pq["per_class"].items()},
        "val_per_class_SQ": {key: value["sq"] for key, value in pq["per_class"].items()},
        "val_per_class_TP": {key: value["tp"] for key, value in pq["per_class"].items()},
        "val_per_class_FP": {key: value["fp"] for key, value in pq["per_class"].items()},
        "val_per_class_FN": {key: value["fn"] for key, value in pq["per_class"].items()},
        "val_per_class_SSE": {name: float(per_class_sse[index]) for index, name in enumerate(CLASS_NAMES)},
        "val_per_class_SST": {name: float(per_class_sst[index]) for index, name in enumerate(CLASS_NAMES)},
        "val_per_class_signed_error": {name: float(residual[:, index].mean()) for index, name in enumerate(CLASS_NAMES)},
        "val_per_class_MAE": {name: float(np.abs(residual[:, index]).mean()) for index, name in enumerate(CLASS_NAMES)},
        "val_per_class_count_ratio": {
            name: float(predicted_total[index] / true_total[index]) if true_total[index] else None
            for index, name in enumerate(CLASS_NAMES)
        },
        "val_count_error": count_error_stats(true_array, predicted_array),
        "val_zero_truth_overcount": zero_truth_count_stats(true_array, predicted_array),
        "val_per_source": per_source,
    }


def save_checkpoint(
    path: Path,
    model: HoVerNetExt,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    phase: int,
    phase_epoch: int,
    global_epoch: int,
    args: argparse.Namespace,
    row: dict,
    rng_generators: dict[str, torch.Generator],
) -> None:
    torch.save(
        {
            "desc": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "phase": phase,
            "phase_epoch": phase_epoch,
            "epoch": global_epoch,
            "args": vars(args),
            "metrics": row,
            "rng_state": capture_rng_state(rng_generators),
            "official_hovernet_commit": OFFICIAL_COMMIT,
            "backbone_architecture": args.backbone_architecture,
            "initialization": initialization_declaration(args.backbone_architecture),
        },
        path,
    )


def write_curve(outdir: Path, rows: list[dict]) -> None:
    (outdir / "training_curve.json").write_text(json.dumps(rows, indent=2))
    scalar_keys = sorted({key for row in rows for key, value in row.items() if not isinstance(value, dict)})
    with (outdir / "training_curve.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in scalar_keys})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--backbone-architecture", choices=BACKBONE_ARCHITECTURES, default="resnet50")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--train-ids", type=Path, help="Optional .npy patch IDs for a development-fold training set")
    parser.add_argument("--val-ids", type=Path, help="Optional .npy patch IDs for the matching held-out development fold")
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--learning-rate-phase2", type=float)
    parser.add_argument(
        "--backbone-lr-multiplier",
        type=float,
        default=1.0,
        help="Phase-2 backbone LR relative to the decoder LR; 1.0 reproduces the official recipe.",
    )
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="Resume an interrupted phase in-place, preserving optimizer, scheduler, and curve history.",
    )
    parser.add_argument("--epochs-phase1", type=int, default=50)
    parser.add_argument("--epochs-phase2", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--val-batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--worker-rng-policy",
        choices=("persistent", "epoch_reseed"),
        default="persistent",
        help=(
            "persistent preserves the original matched-experiment protocol but forbids resume; "
            "epoch_reseed restarts workers from checkpointed RNG each epoch and supports exact resume"
        ),
    )
    parser.add_argument("--hed-probability", type=float, default=0.0)
    parser.add_argument("--hed-profile", type=Path, help="NPZ of development-patch H/E concentrations")
    parser.add_argument("--hed-target-jitter", type=float, default=0.05)
    parser.add_argument("--hed-tail-expansion", type=float, default=0.1)
    parser.add_argument("--hed-strength-min", type=float, default=0.25)
    parser.add_argument("--hed-strength-max", type=float, default=1.0)
    parser.add_argument("--hed-scale-min", type=float, default=0.5)
    parser.add_argument("--hed-scale-max", type=float, default=2.0)
    parser.add_argument(
        "--source-sampling-fraction",
        type=float,
        default=0.0,
        help="Fraction of sampling mass devoted to equal-source weighting.",
    )
    parser.add_argument(
        "--class-sampling-fraction",
        type=float,
        default=0.0,
        help="Fraction of sampling mass devoted to equal-class-mass weighting.",
    )
    parser.add_argument(
        "--uniform-replacement-sampling",
        action="store_true",
        help=(
            "Draw uniform patches with replacement as a mechanism control for weighted samplers; "
            "cannot be combined with nonzero source/class fractions."
        ),
    )
    parser.add_argument(
        "--instance-loss-blend",
        type=float,
        default=0.0,
        help=(
            "Blend from ordinary pixel weighting (0) to equal total loss mass per foreground instance (1); "
            "applied to NP/TP cross-entropy and HV MSE/gradient terms."
        ),
    )
    parser.add_argument(
        "--instance-loss-max-weight",
        type=float,
        default=0.0,
        help=(
            "Optional cap on the final instance-equalized foreground pixel weight; zero disables the cap. "
            "Foreground weights are mass-renormalized after capping."
        ),
    )
    parser.add_argument(
        "--instance-type-loss-weight",
        type=float,
        default=0.0,
        help="Weight for an auxiliary one-cross-entropy-per-GT-nucleus pooled type loss.",
    )
    parser.add_argument(
        "--type-class-balance-rho",
        type=float,
        default=0.0,
        help="Exponent for fixed foreground type weights (1 - training nucleus frequency)^rho.",
    )
    parser.add_argument(
        "--type-focal-gamma",
        type=float,
        default=0.0,
        help="Focal exponent for the HoVer type-branch cross-entropy; zero preserves ordinary CE.",
    )
    parser.add_argument(
        "--type-label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing used only by the optional weighted/focal type loss.",
    )
    parser.add_argument("--metric-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--max-train-patches", type=int)
    parser.add_argument("--max-val-patches", type=int)
    parser.add_argument("--amp", choices=["none", "bf16"], default="none")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.backbone_lr_multiplier <= 0:
        raise ValueError("--backbone-lr-multiplier must be positive")
    if not 0 <= args.hed_probability <= 1:
        raise ValueError("--hed-probability must lie in [0, 1]")
    if not 0 <= args.hed_strength_min <= args.hed_strength_max <= 1:
        raise ValueError("HED strength bounds must satisfy 0 <= min <= max <= 1")
    if not 0 < args.hed_scale_min <= 1 <= args.hed_scale_max:
        raise ValueError("HED scale bounds must satisfy 0 < min <= 1 <= max")
    if not 0 <= args.hed_target_jitter < 1 or args.hed_tail_expansion < 0:
        raise ValueError("HED target jitter/tail expansion are out of range")
    if args.hed_probability > 0 and args.hed_profile is None:
        raise ValueError("--hed-profile is required when HED augmentation is enabled")
    if (
        args.source_sampling_fraction < 0
        or args.class_sampling_fraction < 0
        or args.source_sampling_fraction + args.class_sampling_fraction > 1
    ):
        raise ValueError("source/class sampling fractions must be non-negative and sum to at most one")
    draw_mode = sampling_mode(
        args.source_sampling_fraction,
        args.class_sampling_fraction,
        args.uniform_replacement_sampling,
    )
    if not 0 <= args.instance_loss_blend <= 1:
        raise ValueError("--instance-loss-blend must lie in [0, 1]")
    if args.instance_loss_max_weight < 0 or 0 < args.instance_loss_max_weight < 1:
        raise ValueError("--instance-loss-max-weight must be zero or at least one")
    if args.instance_type_loss_weight < 0:
        raise ValueError("--instance-type-loss-weight must be non-negative")
    if args.type_class_balance_rho < 0 or args.type_focal_gamma < 0:
        raise ValueError("type class-balance rho and focal gamma must be non-negative")
    if not 0 <= args.type_label_smoothing < 1:
        raise ValueError("--type-label-smoothing must lie in [0, 1)")
    if not args.backbone.exists():
        raise FileNotFoundError(args.backbone)
    seed_everything(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(args.prepared / "metadata.csv").sort_values("patch_id")
    if (args.train_ids is None) != (args.val_ids is None):
        raise ValueError("--train-ids and --val-ids must be supplied together")
    if args.train_ids is not None:
        train_ids = np.load(args.train_ids).astype(np.int32)
        val_ids = np.load(args.val_ids).astype(np.int32)
        if len(np.unique(train_ids)) != len(train_ids) or len(np.unique(val_ids)) != len(val_ids):
            raise ValueError("fold ID arrays contain duplicates")
        if len(np.intersect1d(train_ids, val_ids)):
            raise RuntimeError("fold train/validation patch IDs overlap")
        known_ids = set(metadata.patch_id.astype(int))
        if not set(train_ids).issubset(known_ids) or not set(val_ids).issubset(known_ids):
            raise ValueError("fold manifests contain unknown patch IDs")
        forbidden = set(metadata.loc[metadata.split.eq("test"), "patch_id"].astype(int))
        if set(train_ids) & forbidden or set(val_ids) & forbidden:
            raise RuntimeError("locked test patches may not enter development folds")
        train_rows = metadata.loc[metadata.patch_id.isin(train_ids)]
        val_rows = metadata.loc[metadata.patch_id.isin(val_ids)]
    else:
        train_rows = metadata.loc[metadata.split.eq("train")]
        val_rows = metadata.loc[metadata.split.eq("val")]
    if args.max_train_patches:
        train_rows = train_rows.head(args.max_train_patches)
    if args.max_val_patches:
        val_rows = val_rows.head(args.max_val_patches)
    if set(train_rows.source_group) & set(val_rows.source_group):
        raise RuntimeError("train/validation source groups overlap")

    hed_target_concentrations = None
    hed_target_sources = None
    if args.hed_probability > 0:
        with np.load(args.hed_profile) as profile:
            profile_patch_ids = profile["patch_ids"].astype(np.int32)
            profile_concentrations = profile["concentrations"].astype(np.float32)
            profile_sources = profile["sources"].astype(str)
        target_mask = np.isin(profile_patch_ids, train_rows.patch_id.to_numpy(dtype=np.int32))
        if int(target_mask.sum()) != len(train_rows):
            raise RuntimeError("HED profile does not contain every training patch exactly once")
        hed_target_concentrations = profile_concentrations[target_mask]
        hed_target_sources = profile_sources[target_mask]

    train_dataset = PreparedHoverDataset(
        args.prepared,
        train_rows,
        train=True,
        seed=args.seed,
        hed_probability=args.hed_probability,
        hed_target_concentrations=hed_target_concentrations,
        hed_target_sources=hed_target_sources,
        hed_target_jitter=args.hed_target_jitter,
        hed_tail_expansion=args.hed_tail_expansion,
        hed_strength_min=args.hed_strength_min,
        hed_strength_max=args.hed_strength_max,
        hed_scale_min=args.hed_scale_min,
        hed_scale_max=args.hed_scale_max,
    )
    val_dataset = PreparedHoverDataset(args.prepared, val_rows, train=False, seed=args.seed)
    training_type_counts = train_rows[COUNT_COLUMNS].sum(axis=0).to_numpy(dtype=np.float64)
    type_class_weights_array = complement_frequency_type_weights(
        training_type_counts,
        rho=args.type_class_balance_rho,
    )
    loader_options = {
        "num_workers": args.workers,
        "pin_memory": True,
        "worker_init_fn": worker_init,
    }
    if args.worker_rng_policy == "persistent":
        # Preserve the exact loader protocol used by completed E36/E37 pilots.
        # Worker-local augmentation state cannot be checkpointed, so resume is
        # forbidden below rather than silently changing a matched run.
        rng_generators = {"data_loader": torch.Generator().manual_seed(args.seed)}
        sampler_generator = rng_generators["data_loader"]
        train_worker_generator = rng_generators["data_loader"]
        loader_options["persistent_workers"] = args.workers > 0
    else:
        # Exactly resumable at epoch boundaries: workers restart from their own
        # checkpointed generator every epoch.
        rng_generators = {
            "sampler": torch.Generator().manual_seed(args.seed),
            "train_workers": torch.Generator().manual_seed(args.seed + 1_000_003),
            "val_workers": torch.Generator().manual_seed(args.seed + 2_000_003),
        }
        sampler_generator = rng_generators["sampler"]
        train_worker_generator = rng_generators["train_workers"]
        loader_options["persistent_workers"] = False
    sampling_weights = np.ones(len(train_rows), dtype=np.float64)
    sampler = None
    if draw_mode != "without_replacement":
        if draw_mode == "weighted_with_replacement":
            sampling_weights = source_class_patch_weights(
                train_rows.source.astype(str).to_numpy(),
                train_rows[COUNT_COLUMNS].to_numpy(dtype=np.int64),
                source_fraction=args.source_sampling_fraction,
                class_fraction=args.class_sampling_fraction,
            )
        sampler = WeightedRandomSampler(
            torch.as_tensor(sampling_weights, dtype=torch.double),
            num_samples=len(train_dataset),
            replacement=True,
            generator=sampler_generator,
        )
    elif args.worker_rng_policy == "epoch_reseed":
        sampler = RandomSampler(train_dataset, generator=sampler_generator)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=True,
        generator=train_worker_generator,
        **loader_options,
    )
    val_loader_options = dict(loader_options)
    if args.worker_rng_policy == "epoch_reseed":
        val_loader_options["generator"] = rng_generators["val_workers"]
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        drop_last=False,
        **val_loader_options,
    )

    device = torch.device(args.device)
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else None
    type_class_weights = torch.as_tensor(type_class_weights_array, dtype=torch.float32, device=device)
    model = HoVerNetExt(
        num_types=7,
        freeze=True,
        pretrained_backbone=str(args.backbone),
        backbone_name=args.backbone_architecture,
    ).to(device)
    if args.initial_checkpoint is not None and args.resume_checkpoint is not None:
        raise ValueError("use either --initial-checkpoint or --resume-checkpoint, not both")
    initial_global_epoch = 0
    resume_checkpoint = None
    if args.initial_checkpoint is not None:
        checkpoint = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=False)
        if checkpoint.get("initialization") != initialization_declaration(args.backbone_architecture):
            raise RuntimeError("initial checkpoint does not carry the leakage-free initialization declaration")
        if checkpoint.get("backbone_architecture", "resnet50") != args.backbone_architecture:
            raise RuntimeError("initial checkpoint backbone architecture does not match the requested model")
        model.load_state_dict(checkpoint["desc"], strict=True)
        initial_global_epoch = int(checkpoint.get("epoch", 0)) if int(checkpoint.get("phase", 0)) == 1 else 0
    if args.resume_checkpoint is not None:
        require_resumable_worker_policy(args.worker_rng_policy)
        resume_checkpoint = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        if resume_checkpoint.get("initialization") != initialization_declaration(args.backbone_architecture):
            raise RuntimeError("resume checkpoint does not carry the leakage-free initialization declaration")
        if resume_checkpoint.get("backbone_architecture", "resnet50") != args.backbone_architecture:
            raise RuntimeError("resume checkpoint backbone architecture does not match the requested model")
        model.load_state_dict(resume_checkpoint["desc"], strict=True)
        initial_global_epoch = int(
            resume_checkpoint.get("epoch", resume_checkpoint.get("metrics", {}).get("epoch", 0))
        )
        restore_rng_state(resume_checkpoint.get("rng_state"), rng_generators)
    curve_path = args.outdir / "training_curve.json"
    rows: list[dict] = json.loads(curve_path.read_text()) if resume_checkpoint is not None and curve_path.exists() else []
    global_epoch = initial_global_epoch
    best_r2 = max((float(row["val_R2"]) for row in rows if row.get("val_R2") is not None), default=-float("inf"))
    best_mpq = max((float(row["val_mPQ+"]) for row in rows if row.get("val_mPQ+") is not None), default=-float("inf"))
    phase_epochs = [args.epochs_phase1, args.epochs_phase2]
    for phase, epochs in enumerate(phase_epochs, start=1):
        resume_phase = int(resume_checkpoint.get("phase", 0)) if resume_checkpoint is not None else 0
        if resume_checkpoint is not None and phase < resume_phase:
            continue
        model.freeze = phase == 1
        phase_learning_rate = args.learning_rate_phase2 if phase == 2 and args.learning_rate_phase2 is not None else args.learning_rate
        if phase == 2 and args.backbone_lr_multiplier != 1.0:
            backbone_parameters = list(model.backbone.parameters())
            backbone_ids = {id(parameter) for parameter in backbone_parameters}
            decoder_parameters = [parameter for parameter in model.parameters() if id(parameter) not in backbone_ids]
            optimizer = torch.optim.Adam(
                [
                    {"params": backbone_parameters, "lr": phase_learning_rate * args.backbone_lr_multiplier},
                    {"params": decoder_parameters, "lr": phase_learning_rate},
                ],
                betas=(0.9, 0.999),
            )
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=phase_learning_rate, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
        first_phase_epoch = 1
        if resume_checkpoint is not None and phase == resume_phase:
            completed_phase_epochs = int(
                resume_checkpoint.get(
                    "phase_epoch", resume_checkpoint.get("metrics", {}).get("phase_epoch", resume_checkpoint.get("epoch", 0))
                )
            )
            optimizer.load_state_dict(resume_checkpoint["optimizer"])
            if "scheduler" in resume_checkpoint:
                scheduler.load_state_dict(resume_checkpoint["scheduler"])
            else:
                # Legacy checkpoints were written immediately after scheduler.step().
                scheduler.last_epoch = completed_phase_epochs
                scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]
            first_phase_epoch = completed_phase_epochs + 1
        for phase_epoch in range(first_phase_epoch, epochs + 1):
            global_epoch += 1
            started = time.time()
            model.train()
            totals: dict[str, float] = {}
            samples = 0
            sampled_patch_ids: list[int] = []
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss, terms, _ = batch_loss(
                    model,
                    batch,
                    device,
                    amp_dtype,
                    instance_loss_blend=args.instance_loss_blend,
                    instance_loss_max_weight=(args.instance_loss_max_weight or None),
                    instance_type_loss_weight=args.instance_type_loss_weight,
                    type_class_weights=type_class_weights,
                    type_focal_gamma=args.type_focal_gamma,
                    type_label_smoothing=args.type_label_smoothing,
                )
                loss.backward()
                optimizer.step()
                size = len(batch["patch_id"])
                samples += size
                sampled_patch_ids.extend(int(value) for value in batch["patch_id"].tolist())
                totals["train_loss"] = totals.get("train_loss", 0.0) + float(loss.detach().cpu()) * size
                for key, value in terms.items():
                    totals[f"train_{key}"] = totals.get(f"train_{key}", 0.0) + value * size
            row = {
                "phase": phase,
                "phase_epoch": phase_epoch,
                "epoch": global_epoch,
                "learning_rate": optimizer.param_groups[-1]["lr"],
                "decoder_learning_rate": optimizer.param_groups[-1]["lr"],
                "backbone_learning_rate": optimizer.param_groups[0]["lr"],
                "train_sampling_actual": sampling_exposure_summary(sampled_patch_ids, train_rows),
                **{key: value / max(samples, 1) for key, value in totals.items()},
                **validate_loss(
                    model,
                    val_loader,
                    device,
                    amp_dtype,
                    instance_loss_blend=args.instance_loss_blend,
                    instance_loss_max_weight=(args.instance_loss_max_weight or None),
                    instance_type_loss_weight=args.instance_type_loss_weight,
                    type_class_weights=type_class_weights,
                    type_focal_gamma=args.type_focal_gamma,
                    type_label_smoothing=args.type_label_smoothing,
                ),
            }
            should_score = phase_epoch == epochs or phase_epoch == 1 or phase_epoch % args.metric_every == 0
            if should_score:
                row.update(validation_leaderboard_metrics(model, val_loader, args.prepared, metadata, device, amp_dtype))
            row["seconds"] = time.time() - started
            rows.append(row)
            scheduler.step()
            write_curve(args.outdir, rows)
            save_checkpoint(
                args.outdir / "latest.pth", model, optimizer, scheduler, phase, phase_epoch,
                global_epoch, args, row, rng_generators,
            )
            if row.get("val_R2", -float("inf")) > best_r2:
                best_r2 = row["val_R2"]
                save_checkpoint(
                    args.outdir / "best_r2.pth", model, optimizer, scheduler, phase, phase_epoch,
                    global_epoch, args, row, rng_generators,
                )
            if row.get("val_mPQ+", -float("inf")) > best_mpq:
                best_mpq = row["val_mPQ+"]
                save_checkpoint(
                    args.outdir / "best_mpq.pth", model, optimizer, scheduler, phase, phase_epoch,
                    global_epoch, args, row, rng_generators,
                )
            print(json.dumps(row), flush=True)

    summary = {
        "official_hovernet_commit": OFFICIAL_COMMIT,
        "initialization": str(args.backbone),
        "initialization_declaration": initialization_declaration(args.backbone_architecture),
        "backbone_architecture": args.backbone_architecture,
        "conic_checkpoint_used": False,
        "train_patches": len(train_rows),
        "validation_patches": len(val_rows),
        "sampling": {
            "draw_mode": draw_mode,
            "replacement": draw_mode != "without_replacement",
            "source_fraction": args.source_sampling_fraction,
            "class_fraction": args.class_sampling_fraction,
            "expected_source_mass": {
                str(source): float(sampling_weights[train_rows.source.astype(str).to_numpy() == str(source)].sum() / sampling_weights.sum())
                for source in sorted(train_rows.source.astype(str).unique())
            },
        },
        "instance_loss": {
            "blend": args.instance_loss_blend,
            "max_weight": args.instance_loss_max_weight or None,
            "cap_mass_renormalized": bool(args.instance_loss_max_weight),
        },
        "type_loss": {
            "class_balance_rho": args.type_class_balance_rho,
            "focal_gamma": args.type_focal_gamma,
            "label_smoothing": args.type_label_smoothing,
            "training_foreground_nucleus_counts": {
                name: int(value) for name, value in zip(CLASS_NAMES, training_type_counts)
            },
            "class_weights_background_then_scored_classes": type_class_weights_array.tolist(),
            "frequency_source": "development-training nucleus counts only",
        },
        "hed_target_bank": {
            "profile": str(args.hed_profile) if args.hed_profile is not None else None,
            "training_only": True,
            "patches": int(len(hed_target_concentrations)) if hed_target_concentrations is not None else 0,
            "sources": sorted(np.unique(hed_target_sources).tolist()) if hed_target_sources is not None else [],
            "source_balanced_target_sampling": bool(hed_target_sources is not None),
            "target_jitter": args.hed_target_jitter,
            "tail_expansion": args.hed_tail_expansion,
        },
        "best_validation_R2": best_r2,
        "best_validation_mPQ+": best_mpq,
        "args": vars(args),
    }
    summary["args"] = {key: str(value) if isinstance(value, Path) else value for key, value in summary["args"].items()}
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
