from __future__ import annotations

import numpy as np
from skimage.color import hed2rgb, rgb2hed


def hed_concentration(image: np.ndarray, quantile: float = 0.95) -> np.ndarray:
    """Return a robust joint H/E concentration descriptor for an RGB patch."""
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("image must be HxWx3 RGB")
    if not 0 < quantile < 1:
        raise ValueError("quantile must lie in (0, 1)")
    hed = rgb2hed(image.astype(np.float32) / 255.0)
    values = np.quantile(np.maximum(hed[..., :2], 0.0), quantile, axis=(0, 1))
    return np.maximum(values, 1.0e-6).astype(np.float32)


class EmpiricalHEDTargetBank:
    """Source-balanced sampler over observed joint H/E concentration pairs."""

    def __init__(
        self,
        concentrations: np.ndarray,
        sources: np.ndarray,
        jitter: float = 0.05,
        tail_expansion: float = 0.1,
    ) -> None:
        values = np.asarray(concentrations, dtype=np.float32)
        source_values = np.asarray(sources).astype(str)
        if values.ndim != 2 or values.shape[1] != 2 or len(values) != len(source_values):
            raise ValueError("HED target concentrations/sources have incompatible shapes")
        if not len(values) or np.any(values <= 0) or not np.all(np.isfinite(values)):
            raise ValueError("HED target concentrations must be finite and positive")
        if not 0 <= jitter < 1 or tail_expansion < 0:
            raise ValueError("HED target jitter/tail expansion are out of range")
        self.concentrations = values
        self.sources = source_values
        self.jitter = float(jitter)
        self.tail_expansion = float(tail_expansion)
        self.indices = {
            source: np.flatnonzero(source_values == source) for source in sorted(np.unique(source_values))
        }
        robust_low = np.quantile(values, 0.01, axis=0)
        robust_high = np.quantile(values, 0.99, axis=0)
        self.lower = np.maximum(robust_low * (1.0 - tail_expansion), 1.0e-6)
        self.upper = robust_high * (1.0 + tail_expansion)

    def sample(self, rng: np.random.Generator) -> tuple[np.ndarray, str]:
        source_names = tuple(self.indices)
        source = source_names[int(rng.integers(len(source_names)))]
        candidates = self.indices[source]
        index = int(candidates[int(rng.integers(len(candidates)))])
        target = self.concentrations[index].astype(np.float64)
        if self.jitter > 0:
            target *= rng.uniform(1.0 - self.jitter, 1.0 + self.jitter, size=2)
        target = np.clip(target, self.lower, self.upper)
        return target.astype(np.float32), source


def hed_stain_augmentation_array(
    image: np.ndarray,
    rng: np.random.Generator,
    probability: float,
    target_concentration: np.ndarray,
    strength_min: float = 0.25,
    strength_max: float = 1.0,
    scale_min: float = 0.5,
    scale_max: float = 2.0,
) -> np.ndarray:
    """Move a patch toward an empirical joint H/E concentration target."""
    if probability <= 0 or float(rng.random()) >= probability:
        return image
    target = np.asarray(target_concentration, dtype=np.float64)
    if target.shape != (2,) or np.any(target <= 0) or not np.all(np.isfinite(target)):
        raise ValueError("target_concentration must contain two finite positive values")
    if not 0 <= strength_min <= strength_max <= 1:
        raise ValueError("strength bounds must satisfy 0 <= min <= max <= 1")
    if not 0 < scale_min <= 1 <= scale_max:
        raise ValueError("scale bounds must satisfy 0 < min <= 1 <= max")
    rgb = image.astype(np.float32) / 255.0
    hed = rgb2hed(rgb)
    current = np.quantile(np.maximum(hed[..., :2], 0.0), 0.95, axis=(0, 1))
    safe_current = np.maximum(current, 1.0e-6)
    full_scale = np.clip(target / safe_current, scale_min, scale_max)
    strength = float(rng.uniform(strength_min, strength_max))
    scale = np.exp(strength * np.log(full_scale))
    hed[..., :2] = np.maximum(hed[..., :2] * scale, 0.0)
    hed[..., 2] = np.maximum(hed[..., 2], 0.0)
    augmented = np.clip(hed2rgb(hed), 0.0, 1.0)
    green_artifact = (augmented[..., 1] > augmented[..., 0] + 0.08) & (
        augmented[..., 1] > augmented[..., 2] + 0.08
    )
    original_luminance = rgb.mean(axis=-1)
    augmented_luminance = augmented.mean(axis=-1)
    contrast_collapsed = float(augmented_luminance.std()) < 0.5 * float(original_luminance.std())
    newly_white = float((augmented_luminance > 0.95).mean()) - float((original_luminance > 0.95).mean())
    brightness_shift = abs(float(augmented_luminance.mean()) - float(original_luminance.mean()))
    if (
        float(green_artifact.mean()) > 0.01
        or contrast_collapsed
        or newly_white > 0.25
        or brightness_shift > 0.2
    ):
        return image
    return np.rint(augmented * 255.0).astype(np.uint8)


def deterministic_hed_stain_transfer(
    image: np.ndarray,
    target_concentration: np.ndarray,
    strength: float = 1.0,
    scale_min: float = 0.5,
    scale_max: float = 2.0,
) -> np.ndarray:
    """Deterministically translate an RGB patch toward one H/E target."""
    if not 0 <= strength <= 1:
        raise ValueError("strength must lie in [0, 1]")
    return hed_stain_augmentation_array(
        image,
        np.random.default_rng(0),
        probability=1.0,
        target_concentration=target_concentration,
        strength_min=strength,
        strength_max=strength,
        scale_min=scale_min,
        scale_max=scale_max,
    )
