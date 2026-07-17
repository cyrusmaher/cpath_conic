#!/usr/bin/env python3
"""Audit whether two validation-only HoVer endpoints have blendable count errors."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cpath_conic.constants import CLASS_NAMES, COUNT_COLUMNS


def r2_summary(truth: np.ndarray, prediction: np.ndarray) -> dict:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    mean = truth.mean(axis=0)
    sst = np.square(truth - mean).sum(axis=0)
    sse = np.square(prediction - truth).sum(axis=0)
    values = np.full(truth.shape[1], np.nan, dtype=np.float64)
    valid = sst > 0
    values[valid] = 1.0 - sse[valid] / sst[valid]
    return {
        "R2": float(np.nanmean(values)),
        "per_class_R2": {name: float(values[index]) for index, name in enumerate(CLASS_NAMES)},
        "per_class_SSE": {name: float(sse[index]) for index, name in enumerate(CLASS_NAMES)},
    }


def directional_error_summary(truth: np.ndarray, prediction: np.ndarray) -> dict:
    residual = np.asarray(prediction, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    absolute = np.abs(residual)
    return {
        "mean_signed_error": float(residual.mean()),
        "MAE": float(absolute.mean()),
        "under_fraction": float((residual < 0).mean()),
        "exact_fraction": float((residual == 0).mean()),
        "over_fraction": float((residual > 0).mean()),
        **{f"absolute_error_gt_{threshold}_fraction": float((absolute > threshold).mean()) for threshold in (5, 10, 20)},
        **{f"under_error_lt_minus_{threshold}_fraction": float((residual < -threshold).mean()) for threshold in (5, 10, 20)},
        **{f"over_error_gt_{threshold}_fraction": float((residual > threshold).mean()) for threshold in (5, 10, 20)},
        "per_class_signed_error": {
            name: float(residual[:, index].mean()) for index, name in enumerate(CLASS_NAMES)
        },
        "per_class_MAE": {
            name: float(absolute[:, index].mean()) for index, name in enumerate(CLASS_NAMES)
        },
    }


def zero_truth_overcount_summary(truth: np.ndarray, prediction: np.ndarray) -> dict:
    """Describe false count tails only where the corresponding truth count is zero."""
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)

    def summarize_mask(mask: np.ndarray, values: np.ndarray) -> dict:
        selected = values[mask]
        support = int(mask.sum())
        if support == 0:
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
            "support": support,
            "nonzero_fraction": float((selected > 0).mean()),
            "over_5_fraction": float((selected > 5).mean()),
            "over_10_fraction": float((selected > 10).mean()),
            "over_20_fraction": float((selected > 20).mean()),
            "mean_prediction": float(selected.mean()),
            "max_prediction": float(selected.max()),
        }

    zero = truth == 0
    return {
        "all_zero_truth_points": summarize_mask(zero, prediction),
        "per_class": {
            name: summarize_mask(zero[:, index], prediction[:, index])
            for index, name in enumerate(CLASS_NAMES)
        },
    }


def summarize(truth: np.ndarray, prediction: np.ndarray) -> dict:
    return {
        **r2_summary(truth, prediction),
        "count_error": directional_error_summary(truth, prediction),
        "zero_truth_overcount": zero_truth_overcount_summary(truth, prediction),
    }


def blend_counts(first: np.ndarray, second: np.ndarray, second_weights: np.ndarray | float) -> np.ndarray:
    weights = np.asarray(second_weights, dtype=np.float64)
    return np.asarray(first, dtype=np.float64) * (1.0 - weights) + np.asarray(second, dtype=np.float64) * weights


def r2_score_1d(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    sst = float(np.square(truth - truth.mean()).sum())
    return float(1.0 - np.square(prediction - truth).sum() / sst) if sst > 0 else float("nan")


def select_per_class_weights(
    truth: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    selected = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    for class_index in range(len(CLASS_NAMES)):
        candidates = []
        for weight in weights:
            prediction = blend_counts(first[:, class_index], second[:, class_index], weight)
            score = r2_score_1d(truth[:, class_index], prediction)
            if not np.isfinite(score):
                score = -np.inf
            # Prefer the simpler endpoint on an exact tie, then the first model.
            endpoint_distance = min(float(weight), 1.0 - float(weight))
            candidates.append((score, -endpoint_distance, -float(weight), float(weight)))
        selected[class_index] = max(candidates)[-1]
    return selected


def residual_correlations(truth: np.ndarray, first: np.ndarray, second: np.ndarray) -> dict:
    first_residual = np.asarray(first, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    second_residual = np.asarray(second, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    result = {}
    for index, name in enumerate(CLASS_NAMES):
        left = first_residual[:, index]
        right = second_residual[:, index]
        value = float(np.corrcoef(left, right)[0, 1]) if left.std() > 0 and right.std() > 0 else None
        result[name] = value
    flat_left = first_residual.ravel()
    flat_right = second_residual.ravel()
    result["all_class_patch_points"] = (
        float(np.corrcoef(flat_left, flat_right)[0, 1])
        if flat_left.std() > 0 and flat_right.std() > 0 else None
    )
    return result


def leave_one_source_out(
    truth: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    sources: np.ndarray,
    weights: np.ndarray,
) -> dict:
    prediction = np.zeros_like(first, dtype=np.float64)
    selections = {}
    held_out = {}
    for source in sorted(np.unique(sources)):
        test_mask = sources == source
        train_mask = ~test_mask
        selected = select_per_class_weights(
            truth[train_mask], first[train_mask], second[train_mask], weights
        )
        prediction[test_mask] = blend_counts(first[test_mask], second[test_mask], selected)
        selections[str(source)] = dict(zip(CLASS_NAMES, selected.tolist()))
        held_out[str(source)] = {
            "patches": int(test_mask.sum()),
            **summarize(truth[test_mask], prediction[test_mask]),
        }
    return {
        "interpretation": "Each source's class weights were selected using every other source, then applied only to the held-out source.",
        "selected_second_model_weights_by_held_source": selections,
        "pooled_out_of_source": summarize(truth, prediction),
        "held_out_sources": held_out,
    }


def summarize_by_source(
    truth: np.ndarray,
    prediction: np.ndarray,
    sources: np.ndarray,
) -> dict:
    return {
        str(source): {
            "patches": int((sources == source).sum()),
            **summarize(truth[sources == source], prediction[sources == source]),
        }
        for source in sorted(np.unique(sources))
    }


def load_artifact(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as payload:
        required = {"patch_ids", "predicted_counts"}
        if not required.issubset(payload.files):
            raise ValueError(f"{path} must contain {sorted(required)}")
        return payload["patch_ids"].astype(np.int32), payload["predicted_counts"].astype(np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, nargs=2, required=True)
    parser.add_argument("--names", nargs=2, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--weights", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    weights = np.asarray(sorted(set(args.weights)), dtype=np.float64)
    if len(weights) < 2 or np.any(~np.isfinite(weights)) or weights[0] < 0 or weights[-1] > 1:
        parser.error("--weights must contain at least two distinct finite values in [0, 1]")
    first_ids, first = load_artifact(args.artifacts[0])
    second_ids, second = load_artifact(args.artifacts[1])
    if not np.array_equal(first_ids, second_ids) or first.shape != second.shape:
        raise ValueError("paired artifacts must contain identical patch IDs and count shapes")

    metadata = pd.read_csv(args.prepared / "metadata.csv").set_index("patch_id").loc[first_ids]
    split_values = sorted(metadata.split.astype(str).unique())
    if any(value.lower() == "test" for value in split_values):
        raise ValueError("count-complementarity weights may not be selected on the locked test set")
    truth = metadata[COUNT_COLUMNS].to_numpy(dtype=np.float64)
    sources = metadata.source.astype(str).to_numpy()

    global_candidates = []
    for weight in weights:
        prediction = blend_counts(first, second, weight)
        global_candidates.append({"second_model_weight": float(weight), **summarize(truth, prediction)})
    selected_global = max(global_candidates, key=lambda item: item["R2"])

    selected_per_class = select_per_class_weights(truth, first, second, weights)
    per_class_prediction = blend_counts(first, second, selected_per_class)
    cross_source = leave_one_source_out(truth, first, second, sources, weights)
    best_endpoint = max(summarize(truth, first)["R2"], summarize(truth, second)["R2"])
    full_blend_r2 = summarize(truth, per_class_prediction)["R2"]
    cross_source_r2 = cross_source["pooled_out_of_source"]["R2"]

    report = {
        "protocol": "validation-only paired count-residual and bounded blend audit; no test labels accessed",
        "evaluation_set": {
            "patches": int(len(first_ids)),
            "metadata_splits": split_values,
            "sources": {str(key): int(value) for key, value in metadata.source.value_counts().sort_index().items()},
        },
        "first_model": {
            "name": args.names[0], "artifact": str(args.artifacts[0]),
            **summarize(truth, first), "by_source": summarize_by_source(truth, first, sources),
        },
        "second_model": {
            "name": args.names[1], "artifact": str(args.artifacts[1]),
            **summarize(truth, second), "by_source": summarize_by_source(truth, second, sources),
        },
        "signed_residual_correlation": residual_correlations(truth, first, second),
        "global_weight_candidates": global_candidates,
        "selected_global_weight": selected_global,
        "selected_per_class": {
            "second_model_weights": dict(zip(CLASS_NAMES, selected_per_class.tolist())),
            **summarize(truth, per_class_prediction),
        },
        "leave_one_source_out": cross_source,
        "stability": {
            "best_single_endpoint_R2": float(best_endpoint),
            "full_validation_per_class_blend_delta_R2": float(full_blend_r2 - best_endpoint),
            "leave_one_source_out_blend_delta_R2": float(cross_source_r2 - best_endpoint),
            "full_minus_cross_source_R2": float(full_blend_r2 - cross_source_r2),
        },
        "guardrail": (
            "Promote a count blend only if class-normalized SSE gains survive source-held-out selection, "
            "directional bias/outlier tails remain acceptable, and supported true-zero source/class strata "
            "do not acquire a materially larger false-count tail."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=True))
    print(json.dumps(report["stability"], indent=2))


if __name__ == "__main__":
    main()
