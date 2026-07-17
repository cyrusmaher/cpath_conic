from __future__ import annotations

import numpy as np


def minority_patch_weights(class_counts: np.ndarray, blend: float = 0.5) -> np.ndarray:
    """Image-level importance weights with equal total mass per cell class.

    Each class contributes ``count_in_patch / total_class_count``.  The resulting
    importance distribution is mean-normalized and blended with uniform weights
    to limit variance from extremely rare classes.
    """
    counts = np.asarray(class_counts, dtype=np.float64)
    if counts.ndim != 2 or not len(counts):
        raise ValueError("class_counts must be a non-empty patches-by-classes matrix")
    if np.any(counts < 0):
        raise ValueError("class_counts cannot be negative")
    if not 0 <= blend <= 1:
        raise ValueError("blend must lie in [0, 1]")
    totals = counts.sum(axis=0)
    represented = totals > 0
    if not represented.any():
        return np.ones(len(counts), dtype=np.float64)
    importance = (counts[:, represented] / totals[represented]).sum(axis=1)
    if not np.any(importance > 0):
        return np.ones(len(counts), dtype=np.float64)
    normalized = importance / importance.mean()
    return ((1.0 - blend) + blend * normalized).astype(np.float64)


def source_patch_weights(sources: np.ndarray, blend: float = 0.5) -> np.ndarray:
    """Patch weights that allocate equal aggregate mass to every source."""
    values = np.asarray(sources)
    if values.ndim != 1 or not len(values):
        raise ValueError("sources must be a non-empty vector")
    if not 0 <= blend <= 1:
        raise ValueError("blend must lie in [0, 1]")
    _, inverse, counts = np.unique(values.astype(str), return_inverse=True, return_counts=True)
    balanced = 1.0 / counts[inverse].astype(np.float64)
    balanced /= balanced.mean()
    return ((1.0 - blend) + blend * balanced).astype(np.float64)


def source_class_patch_weights(
    sources: np.ndarray,
    class_counts: np.ndarray,
    source_fraction: float = 0.25,
    class_fraction: float = 0.25,
) -> np.ndarray:
    """Blend uniform, equal-source, and equal-class-mass patch sampling."""
    if source_fraction < 0 or class_fraction < 0 or source_fraction + class_fraction > 1:
        raise ValueError("sampling fractions must be non-negative and sum to at most one")
    source = source_patch_weights(sources, blend=1.0)
    minority = minority_patch_weights(class_counts, blend=1.0)
    uniform_fraction = 1.0 - source_fraction - class_fraction
    weights = uniform_fraction + source_fraction * source + class_fraction * minority
    return (weights / weights.mean()).astype(np.float64)


def expected_unique_draws(weights: np.ndarray, draws: int) -> float:
    """Expected number of unique items under weighted sampling with replacement."""
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("weights must be a non-empty vector")
    if np.any(values < 0) or not np.any(values > 0):
        raise ValueError("weights must be non-negative with positive total mass")
    if draws < 0:
        raise ValueError("draws must be non-negative")
    if draws == 0:
        return 0.0
    probabilities = values / values.sum()
    return float(np.sum(-np.expm1(draws * np.log1p(-probabilities))))


def effective_sample_size(weights: np.ndarray) -> float:
    """Importance-sampling effective sample size for normalized patch weights."""
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("weights must be a non-empty vector")
    if np.any(values < 0) or not np.any(values > 0):
        raise ValueError("weights must be non-negative with positive total mass")
    probabilities = values / values.sum()
    return float(1.0 / np.square(probabilities).sum())
