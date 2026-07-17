#!/usr/bin/env python3
"""Compare validation-only HoVer-Net LR pilots with a matched control."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cpath_conic.constants import CLASS_NAMES


def load_family(root: Path) -> dict[float, dict]:
    runs = {}
    for curve_path in sorted(root.glob("lr_*/training_curve.json")):
        if re.fullmatch(r"lr_\d+(?:\.\d+)?e[+-]?\d+", curve_path.parent.name) is None:
            continue
        rows = json.loads(curve_path.read_text())
        if not rows:
            continue
        learning_rate = float(rows[0]["decoder_learning_rate"])
        runs[learning_rate] = {"path": str(curve_path.parent), "rows": rows}
    if not runs:
        raise FileNotFoundError(f"no LR curves found under {root}")
    return runs


def scored(rows: list[dict]) -> list[dict]:
    return [
        row for row in rows
        if row.get("val_R2") is not None
        and row.get("val_mPQ+") is not None
        and np.isfinite(float(row["val_R2"]))
        and np.isfinite(float(row["val_mPQ+"]))
    ]


def compact(row: dict, learning_rate: float, path: str) -> dict:
    keys = [
        "val_R2", "val_mPQ+", "val_mDQ+", "val_mSQ+", "val_loss",
        "val_foreground_jaccard", "val_foreground_dice", "val_bPQ",
        "val_binary_DQ", "val_binary_SQ", "val_AJI+", "val_boundary_F1",
        "val_per_class_R2", "val_per_class_PQ",
        "val_per_class_DQ", "val_per_class_SQ", "val_per_class_signed_error",
        "val_per_class_MAE", "val_per_class_count_ratio", "val_per_class_TP",
        "val_per_class_FP", "val_per_class_FN", "val_per_class_SSE", "val_per_class_SST",
        "val_count_error", "val_per_source", "val_instance_type_confusion",
    ]
    result = {
        "learning_rate": learning_rate,
        "epoch": int(row["epoch"]),
        "run_dir": path,
        **{key: row.get(key) for key in keys},
    }
    # Curves produced by workers launched before mDQ+/mSQ+ became explicit
    # still contain the six per-class components, so retain full comparability.
    for output_key, per_class_key in (("val_mDQ+", "val_per_class_DQ"), ("val_mSQ+", "val_per_class_SQ")):
        if result.get(output_key) is None and result.get(per_class_key):
            result[output_key] = float(np.mean([result[per_class_key][name] for name in CLASS_NAMES]))
    return result


def select(runs: dict[float, dict], metric: str) -> dict:
    candidates = []
    for learning_rate, run in runs.items():
        candidates.extend(
            compact(row, learning_rate, run["path"])
            for row in scored(run["rows"])
        )
    return max(candidates, key=lambda item: (float(item[metric]), int(item["epoch"])))


def at_checkpoint(runs: dict[float, dict], selected: dict) -> dict:
    learning_rate = float(selected["learning_rate"])
    epoch = int(selected["epoch"])
    if learning_rate not in runs:
        raise KeyError(f"LR {learning_rate:g} is absent from matched family")
    for row in scored(runs[learning_rate]["rows"]):
        if int(row["epoch"]) == epoch:
            return compact(row, learning_rate, runs[learning_rate]["path"])
    raise KeyError(f"checkpoint LR={learning_rate:g}, epoch={epoch} is absent from matched family")


def maybe_at_checkpoint(runs: dict[float, dict], selected: dict) -> tuple[dict | None, str | None]:
    """Return an unavailable reason for intentionally partial staged grids."""
    try:
        return at_checkpoint(runs, selected), None
    except KeyError as error:
        return None, str(error)


def selection_boundary_flags(runs: dict[float, dict], selected: dict) -> dict:
    """Make LR/horizon edge selections explicit before full-run promotion."""
    learning_rates = sorted(runs)
    learning_rate = float(selected["learning_rate"])
    scored_epochs = [int(row["epoch"]) for row in scored(runs[learning_rate]["rows"])]
    at_lower_lr = learning_rate == learning_rates[0]
    at_upper_lr = learning_rate == learning_rates[-1]
    at_horizon = int(selected["epoch"]) == max(scored_epochs)
    return {
        "selected_learning_rate": learning_rate,
        "selected_epoch": int(selected["epoch"]),
        "searched_learning_rates": learning_rates,
        "at_lower_learning_rate_boundary": at_lower_lr,
        "at_upper_learning_rate_boundary": at_upper_lr,
        "at_scored_horizon_boundary": at_horizon,
        "requires_lr_expansion_before_promotion": bool(at_lower_lr or at_upper_lr),
        "requires_horizon_extension_before_promotion": bool(at_horizon),
    }


def matched_grid(candidate: dict[float, dict], control: dict[float, dict]) -> list[dict]:
    results = []
    for learning_rate in sorted(set(candidate) & set(control)):
        candidate_rows = {int(row["epoch"]): row for row in scored(candidate[learning_rate]["rows"])}
        control_rows = {int(row["epoch"]): row for row in scored(control[learning_rate]["rows"])}
        for epoch in sorted(set(candidate_rows) & set(control_rows)):
            left = compact(candidate_rows[epoch], learning_rate, candidate[learning_rate]["path"])
            right = compact(control_rows[epoch], learning_rate, control[learning_rate]["path"])
            item = {
                "learning_rate": learning_rate,
                "epoch": epoch,
                "delta_R2": float(left["val_R2"] - right["val_R2"]),
                "delta_mPQ+": float(left["val_mPQ+"] - right["val_mPQ+"]),
                "delta_mDQ+": float(left["val_mDQ+"] - right["val_mDQ+"]),
                "delta_mSQ+": float(left["val_mSQ+"] - right["val_mSQ+"]),
                "delta_val_loss": float(left["val_loss"] - right["val_loss"]),
            }
            for key in ("val_per_class_R2", "val_per_class_PQ", "val_per_class_signed_error"):
                if left.get(key) and right.get(key):
                    item[f"delta_{key}"] = {
                        name: float(left[key][name] - right[key][name]) for name in CLASS_NAMES
                    }
            results.append(item)
    return results


def r2_sse_drivers(candidate: dict, control: dict) -> dict:
    """Exactly decompose macro-R² change into source×class SSE contributions."""
    candidate_sources = candidate.get("val_per_source") or {}
    control_sources = control.get("val_per_source") or {}
    sst = control.get("val_per_class_SST") or candidate.get("val_per_class_SST") or {}
    if not candidate_sources or not control_sources or not sst:
        return {"available": False, "reason": "source-level SSE was not recorded by this checkpoint"}
    by_source = {}
    by_source_class = []
    for source in sorted(set(candidate_sources) & set(control_sources)):
        contributions = {}
        for name in CLASS_NAMES:
            denominator = float(sst.get(name, 0.0))
            if denominator <= 0:
                continue
            candidate_sse = float(candidate_sources[source]["per_class_SSE"][name])
            control_sse = float(control_sources[source]["per_class_SSE"][name])
            contribution = -(candidate_sse - control_sse) / denominator / len(CLASS_NAMES)
            contributions[name] = contribution
            by_source_class.append(
                {
                    "source": source,
                    "class": name,
                    "delta_macro_R2_contribution": contribution,
                    "candidate_SSE": candidate_sse,
                    "control_SSE": control_sse,
                }
            )
        candidate_error = candidate_sources[source].get("count_error", {})
        control_error = control_sources[source].get("count_error", {})
        by_source[source] = {
            "delta_macro_R2_contribution": float(sum(contributions.values())),
            "per_class_contribution": contributions,
            "delta_mean_signed_error": (
                float(candidate_error["mean_signed_error"] - control_error["mean_signed_error"])
                if candidate_error and control_error else None
            ),
            "delta_MAE": (
                float(candidate_error["MAE"] - control_error["MAE"])
                if candidate_error and control_error else None
            ),
            "delta_absolute_error_gt_5_fraction": (
                float(candidate_error["absolute_error_gt_5_fraction"] - control_error["absolute_error_gt_5_fraction"])
                if candidate_error and control_error else None
            ),
            "delta_absolute_error_gt_10_fraction": (
                float(candidate_error["absolute_error_gt_10_fraction"] - control_error["absolute_error_gt_10_fraction"])
                if candidate_error and control_error else None
            ),
        }
    by_source_class.sort(key=lambda row: abs(row["delta_macro_R2_contribution"]), reverse=True)
    reconstructed = float(sum(item["delta_macro_R2_contribution"] for item in by_source.values()))
    observed = float(candidate["val_R2"] - control["val_R2"])
    return {
        "available": True,
        "interpretation": "Positive values reduce class-normalized SSE and therefore raise official macro-R².",
        "observed_delta_R2": observed,
        "reconstructed_delta_R2": reconstructed,
        "reconstruction_error": float(reconstructed - observed),
        "by_source": by_source,
        "largest_source_class_drivers": by_source_class[:20],
    }


def _mpq_from_source_stats(source_rows: dict[str, dict]) -> float:
    class_values = []
    for name in CLASS_NAMES:
        totals = {key: 0.0 for key in ("tp", "fp", "fn", "sum_iou")}
        for source in source_rows.values():
            stats = source["per_class_PQ_stats"][name]
            for key in totals:
                totals[key] += float(stats[key])
        denominator = totals["tp"] + 0.5 * totals["fp"] + 0.5 * totals["fn"]
        dq = totals["tp"] / (denominator + 1.0e-6)
        sq = totals["sum_iou"] / (totals["tp"] + 1.0e-6)
        class_values.append(dq * sq)
    return float(np.mean(class_values))


def mpq_source_counterfactuals(candidate: dict, control: dict) -> dict:
    """Measure each source's one-at-a-time effect on pooled mPQ sufficient statistics."""
    candidate_sources = candidate.get("val_per_source") or {}
    control_sources = control.get("val_per_source") or {}
    if not candidate_sources or not control_sources:
        return {"available": False, "reason": "source-level PQ statistics were not recorded by this checkpoint"}
    if any("per_class_PQ_stats" not in item for item in [*candidate_sources.values(), *control_sources.values()]):
        return {"available": False, "reason": "source-level PQ sufficient statistics are incomplete"}
    control_mpq = _mpq_from_source_stats(control_sources)
    candidate_mpq = _mpq_from_source_stats(candidate_sources)
    effects = []
    for source in sorted(set(candidate_sources) & set(control_sources)):
        counterfactual = dict(control_sources)
        counterfactual[source] = candidate_sources[source]
        value = _mpq_from_source_stats(counterfactual)
        effects.append({"source": source, "one_source_delta_mPQ+": float(value - control_mpq)})
    effects.sort(key=lambda row: abs(row["one_source_delta_mPQ+"]), reverse=True)
    return {
        "available": True,
        "interpretation": "Each effect swaps one source's candidate TP/FP/FN/IoU statistics into the control; effects are diagnostic and non-additive.",
        "candidate_mPQ+_from_stats": candidate_mpq,
        "control_mPQ+_from_stats": control_mpq,
        "observed_delta_mPQ+": float(candidate["val_mPQ+"] - control["val_mPQ+"]),
        "one_source_counterfactuals": effects,
    }


def type_confusion_drivers(candidate: dict, control: dict) -> dict:
    """Expose whether a typed-PQ change is typing, missed-GT, or spurious-prediction driven."""
    left = candidate.get("val_instance_type_confusion") or {}
    right = control.get("val_instance_type_confusion") or {}
    if not left or not right:
        return {"available": False, "reason": "decoded-instance type confusion was not recorded by this checkpoint"}
    if left.get("labels") != right.get("labels"):
        return {"available": False, "reason": "candidate/control confusion labels differ"}
    labels = left["labels"]
    candidate_matrix = np.asarray(left["matrix"], dtype=np.int64)
    control_matrix = np.asarray(right["matrix"], dtype=np.int64)
    delta = candidate_matrix - control_matrix
    swaps = []
    for true_index, true_name in enumerate(labels[:-1]):
        for pred_index, pred_name in enumerate(labels[:-1]):
            if true_index == pred_index:
                continue
            swaps.append({
                "ground_truth": true_name,
                "predicted": pred_name,
                "candidate": int(candidate_matrix[true_index, pred_index]),
                "control": int(control_matrix[true_index, pred_index]),
                "delta": int(delta[true_index, pred_index]),
            })
    swaps.sort(key=lambda row: abs(row["delta"]), reverse=True)
    return {
        "available": True,
        "interpretation": (
            "Instances are matched class-agnostically at IoU>0.5. Off-diagonal cells are typing swaps; "
            "the last column is missed ground truth and the last row is spurious prediction."
        ),
        "labels": labels,
        "candidate": left,
        "control": right,
        "delta_matrix": delta.tolist(),
        "delta_matched_type_accuracy": float(left["matched_type_accuracy"] - right["matched_type_accuracy"]),
        "missed_truth_delta_by_class": {
            name: int(delta[index, -1]) for index, name in enumerate(labels[:-1])
        },
        "spurious_prediction_delta_by_class": {
            name: int(delta[-1, index]) for index, name in enumerate(labels[:-1])
        },
        "largest_typing_swap_changes": swaps[:15],
    }


def count_error_deltas(candidate: dict, control: dict) -> dict:
    """Retain direction and prevalence for every logged count-error statistic."""
    left = candidate.get("val_count_error") or {}
    right = control.get("val_count_error") or {}
    if not left or not right:
        return {"available": False, "reason": "directional count-error statistics were not recorded"}
    keys = [key for key in left if key != "points" and key in right]
    return {
        "available": True,
        "interpretation": "Error is prediction minus ground truth; positive signed/tail values mean more over-counting.",
        "candidate": left,
        "control": right,
        "delta": {key: float(left[key] - right[key]) for key in keys},
    }


def plot_report(
    candidate: dict[float, dict],
    control: dict[float, dict],
    candidate_best_r2: dict,
    candidate_best_mpq: dict,
    control_best_r2: dict,
    control_best_mpq: dict,
    candidate_label: str,
    control_label: str,
    output: Path,
) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(14, 17), dpi=150, constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(candidate), 2)))
    color_by_lr = {lr: colors[index] for index, lr in enumerate(sorted(candidate))}
    for family, label, linestyle in (
        (candidate, candidate_label, "-"),
        (control, control_label, "--"),
    ):
        for learning_rate, run in sorted(family.items()):
            rows = scored(run["rows"])
            if not rows:
                continue
            legend = f"{label} LR={learning_rate:g}"
            axes[0, 0].plot(
                [row["epoch"] for row in rows], [row["val_R2"] for row in rows],
                marker="o", linestyle=linestyle, color=color_by_lr.get(learning_rate), label=legend,
            )
            axes[0, 1].plot(
                [row["epoch"] for row in rows], [row["val_mPQ+"] for row in rows],
                marker="o", linestyle=linestyle, color=color_by_lr.get(learning_rate), label=legend,
            )
            axes[1, 0].plot(
                [compact(row, learning_rate, run["path"])["epoch"] for row in rows],
                [compact(row, learning_rate, run["path"])["val_mDQ+"] for row in rows],
                linestyle=linestyle, color=color_by_lr.get(learning_rate), label=legend,
            )
            axes[1, 1].plot(
                [compact(row, learning_rate, run["path"])["epoch"] for row in rows],
                [compact(row, learning_rate, run["path"])["val_mSQ+"] for row in rows],
                linestyle=linestyle, color=color_by_lr.get(learning_rate), label=legend,
            )
            axes[2, 0].plot(
                [row["epoch"] for row in run["rows"]], [row["val_loss"] for row in run["rows"]],
                linestyle=linestyle, color=color_by_lr.get(learning_rate), label=legend,
            )

    positions = np.arange(len(CLASS_NAMES))
    width = 0.36
    comparisons = (
        (axes[2, 1], candidate_best_r2, control_best_r2, "val_per_class_R2", "Per-class R² at each R²-selected checkpoint"),
        (axes[3, 0], candidate_best_mpq, control_best_mpq, "val_per_class_PQ", "Per-class PQ at each mPQ+-selected checkpoint"),
        (axes[3, 1], candidate_best_r2, control_best_r2, "val_per_class_signed_error", "Signed count bias at each R²-selected checkpoint"),
    )
    for axis, candidate_best, control_best, key, title in comparisons:
        candidate_values = [candidate_best[key][name] for name in CLASS_NAMES]
        control_values = [control_best[key][name] for name in CLASS_NAMES]
        axis.bar(positions - width / 2, control_values, width, label=control_label, color="#777777")
        axis.bar(positions + width / 2, candidate_values, width, label=candidate_label, color="#2b8cbe")
        axis.set(title=title, xticks=positions, xticklabels=CLASS_NAMES)
        axis.tick_params(axis="x", labelrotation=28)

    axes[0, 0].set(title="Validation count trajectory", xlabel="epoch", ylabel="macro R²")
    axes[0, 1].set(title="Validation segmentation/type trajectory", xlabel="epoch", ylabel="mPQ+")
    axes[1, 0].set(title="Typed detection quality trajectory", xlabel="epoch", ylabel="mDQ+")
    axes[1, 1].set(title="Typed segmentation quality trajectory", xlabel="epoch", ylabel="mSQ+")
    axes[2, 0].set(title="Official six-term validation loss", xlabel="epoch", ylabel="loss")
    for axis in axes.flat:
        axis.axhline(0, color="#555555", linewidth=0.8, alpha=0.5)
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle("Matched validation-only HoVer-Net intervention audit")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--control-label", default="control")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-plot", type=Path, required=True)
    args = parser.parse_args()

    candidate = load_family(args.candidate_root)
    control = load_family(args.control_root)
    candidate_best_r2 = select(candidate, "val_R2")
    candidate_best_mpq = select(candidate, "val_mPQ+")
    control_best_r2 = select(control, "val_R2")
    control_best_mpq = select(control, "val_mPQ+")
    control_at_candidate_r2 = at_checkpoint(control, candidate_best_r2)
    control_at_candidate_mpq = at_checkpoint(control, candidate_best_mpq)
    candidate_at_control_r2, candidate_at_control_r2_reason = maybe_at_checkpoint(candidate, control_best_r2)
    candidate_at_control_mpq, candidate_at_control_mpq_reason = maybe_at_checkpoint(candidate, control_best_mpq)
    paired_at_control_selected_r2 = (
        {
            "available": True,
            "learning_rate": control_best_r2["learning_rate"],
            "epoch": control_best_r2["epoch"],
            "delta": float(candidate_at_control_r2["val_R2"] - control_best_r2["val_R2"]),
            "count_error": count_error_deltas(candidate_at_control_r2, control_best_r2),
            "matched_candidate": candidate_at_control_r2,
        }
        if candidate_at_control_r2 is not None
        else {"available": False, "reason": candidate_at_control_r2_reason}
    )
    paired_at_control_selected_mpq = (
        {
            "available": True,
            "learning_rate": control_best_mpq["learning_rate"],
            "epoch": control_best_mpq["epoch"],
            "delta": float(candidate_at_control_mpq["val_mPQ+"] - control_best_mpq["val_mPQ+"]),
            "delta_mDQ+": float(candidate_at_control_mpq["val_mDQ+"] - control_best_mpq["val_mDQ+"]),
            "delta_mSQ+": float(candidate_at_control_mpq["val_mSQ+"] - control_best_mpq["val_mSQ+"]),
            "count_error": count_error_deltas(candidate_at_control_mpq, control_best_mpq),
            "matched_candidate": candidate_at_control_mpq,
        }
        if candidate_at_control_mpq is not None
        else {"available": False, "reason": candidate_at_control_mpq_reason}
    )
    report = {
        "protocol": "validation-only; no internal-test predictions or labels read",
        "candidate": {
            "label": args.candidate_label,
            "root": str(args.candidate_root),
            "selected_R2": candidate_best_r2,
            "selected_mPQ+": candidate_best_mpq,
            "R2_selection_boundaries": selection_boundary_flags(candidate, candidate_best_r2),
            "mPQ_selection_boundaries": selection_boundary_flags(candidate, candidate_best_mpq),
        },
        "control": {
            "label": args.control_label,
            "root": str(args.control_root),
            "selected_R2": control_best_r2,
            "selected_mPQ+": control_best_mpq,
            "R2_selection_boundaries": selection_boundary_flags(control, control_best_r2),
            "mPQ_selection_boundaries": selection_boundary_flags(control, control_best_mpq),
        },
        "independently_selected_recipe_delta": {
            "R2": float(candidate_best_r2["val_R2"] - control_best_r2["val_R2"]),
            "mPQ+": float(candidate_best_mpq["val_mPQ+"] - control_best_mpq["val_mPQ+"]),
            "mDQ+_at_mPQ_selected": float(candidate_best_mpq["val_mDQ+"] - control_best_mpq["val_mDQ+"]),
            "mSQ+_at_mPQ_selected": float(candidate_best_mpq["val_mSQ+"] - control_best_mpq["val_mSQ+"]),
        },
        "paired_effect_at_candidate_selected_checkpoint": {
            "R2": {
                "learning_rate": candidate_best_r2["learning_rate"],
                "epoch": candidate_best_r2["epoch"],
                "delta": float(candidate_best_r2["val_R2"] - control_at_candidate_r2["val_R2"]),
                "count_error": count_error_deltas(candidate_best_r2, control_at_candidate_r2),
                "matched_control": control_at_candidate_r2,
            },
            "mPQ+": {
                "learning_rate": candidate_best_mpq["learning_rate"],
                "epoch": candidate_best_mpq["epoch"],
                "delta": float(candidate_best_mpq["val_mPQ+"] - control_at_candidate_mpq["val_mPQ+"]),
                "delta_mDQ+": float(candidate_best_mpq["val_mDQ+"] - control_at_candidate_mpq["val_mDQ+"]),
                "delta_mSQ+": float(candidate_best_mpq["val_mSQ+"] - control_at_candidate_mpq["val_mSQ+"]),
                "count_error": count_error_deltas(candidate_best_mpq, control_at_candidate_mpq),
                "matched_control": control_at_candidate_mpq,
            },
        },
        "paired_effect_at_control_selected_checkpoint": {
            "R2": paired_at_control_selected_r2,
            "mPQ+": paired_at_control_selected_mpq,
        },
        "metric_driver_analysis": {
            "at_candidate_selected_R2_checkpoint": r2_sse_drivers(candidate_best_r2, control_at_candidate_r2),
            "at_candidate_selected_mPQ_checkpoint": mpq_source_counterfactuals(candidate_best_mpq, control_at_candidate_mpq),
            "type_confusion_at_candidate_selected_mPQ_checkpoint": type_confusion_drivers(
                candidate_best_mpq, control_at_candidate_mpq
            ),
            "independently_selected_R2_recipes": r2_sse_drivers(candidate_best_r2, control_best_r2),
            "independently_selected_mPQ_recipes": mpq_source_counterfactuals(candidate_best_mpq, control_best_mpq),
            "type_confusion_for_independently_selected_mPQ_recipes": type_confusion_drivers(
                candidate_best_mpq, control_best_mpq
            ),
        },
        "matched_same_lr_epoch_deltas": matched_grid(candidate, control),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2))
    plot_report(
        candidate, control, candidate_best_r2, candidate_best_mpq, control_best_r2, control_best_mpq,
        args.candidate_label, args.control_label, args.out_plot,
    )
    paired = report["paired_effect_at_candidate_selected_checkpoint"]
    print(json.dumps({
        "independently_selected_recipe_delta": report["independently_selected_recipe_delta"],
        "paired_effect_at_candidate_selected_checkpoint": {
            "R2": {
                key: paired["R2"][key]
                for key in ("learning_rate", "epoch", "delta")
            },
            "mPQ+": {
                key: paired["mPQ+"][key]
                for key in ("learning_rate", "epoch", "delta", "delta_mDQ+", "delta_mSQ+")
            },
        },
        "full_report": str(args.out_json),
        "plot": str(args.out_plot),
    }, indent=2))


if __name__ == "__main__":
    main()
