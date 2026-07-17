#!/usr/bin/env python3
"""Refine the E37 mPQ-specific class-sampling intensity after serialized work."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cpath_conic.queue_integrity import archive_incomplete_persistent_run


def wait_for_json(path: Path, poll_seconds: int) -> dict:
    while True:
        if path.exists():
            try:
                payload = json.loads(path.read_text())
                time.sleep(max(10, poll_seconds))
                return payload
            except (OSError, json.JSONDecodeError):
                pass
        print(f"waiting for completed prerequisite: {path}", flush=True)
        time.sleep(max(1, poll_seconds))


def run_checked(command: list[str]) -> None:
    print("running:", " ".join(command), flush=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise SystemExit(f"command failed with exit code {completed.returncode}: {' '.join(command)}")


def component(row: dict, direct: str, per_class: str) -> float:
    if row.get(direct) is not None:
        return float(row[direct])
    return float(np.mean(list(row[per_class].values())))


def scored_rows(run_dir: Path, fraction: float, learning_rate: float) -> list[dict]:
    rows = json.loads((run_dir / "training_curve.json").read_text())
    scored = []
    for row in rows:
        if row.get("val_R2") is None or row.get("val_mPQ+") is None:
            continue
        count_error = row["val_count_error"]
        scored.append({
            "class_sampling_fraction": fraction,
            "learning_rate": learning_rate,
            "epoch": int(row["epoch"]),
            "val_R2": float(row["val_R2"]),
            "val_mPQ+": float(row["val_mPQ+"]),
            "val_mDQ+": component(row, "val_mDQ+", "val_per_class_DQ"),
            "val_mSQ+": component(row, "val_mSQ+", "val_per_class_SQ"),
            "val_MAE": float(count_error["MAE"]),
            "val_mean_signed_error": float(count_error["mean_signed_error"]),
            "val_absolute_error_gt_10_fraction": float(count_error["absolute_error_gt_10_fraction"]),
            "val_absolute_error_gt_20_fraction": float(count_error["absolute_error_gt_20_fraction"]),
            "val_under_error_lt_minus_10_fraction": float(count_error["under_error_lt_minus_10_fraction"]),
            "val_under_error_lt_minus_20_fraction": float(count_error["under_error_lt_minus_20_fraction"]),
            "val_over_error_gt_10_fraction": float(count_error["over_error_gt_10_fraction"]),
            "val_over_error_gt_20_fraction": float(count_error["over_error_gt_20_fraction"]),
            "run_dir": str(run_dir),
        })
    return scored


def select(rows: list[dict], metric: str) -> dict:
    finite = [row for row in rows if np.isfinite(row[metric])]
    if not finite:
        raise RuntimeError(f"no finite class-sampling candidates for {metric}")
    return max(finite, key=lambda row: (row[metric], row["epoch"]))


def scored_checkpoint(run_dir: Path, epoch: int) -> dict:
    rows = json.loads((run_dir / "training_curve.json").read_text())
    matches = [row for row in rows if int(row.get("epoch", -1)) == epoch]
    if len(matches) != 1 or matches[0].get("val_mPQ+") is None:
        raise RuntimeError(f"expected one scored checkpoint at epoch {epoch} in {run_dir}")
    return matches[0]


def selection_with_full_metrics(selection: dict) -> dict:
    """Return a compact selected row with the directional tail metrics needed by E45."""
    row = scored_checkpoint(Path(selection["run_dir"]), int(selection["epoch"]))
    count_error = row["val_count_error"]
    return {
        **selection,
        "val_R2": float(row["val_R2"]),
        "val_mPQ+": float(row["val_mPQ+"]),
        "val_mDQ+": component(row, "val_mDQ+", "val_per_class_DQ"),
        "val_mSQ+": component(row, "val_mSQ+", "val_per_class_SQ"),
        "val_MAE": float(count_error["MAE"]),
        "val_mean_signed_error": float(count_error["mean_signed_error"]),
        "val_absolute_error_gt_10_fraction": float(count_error["absolute_error_gt_10_fraction"]),
        "val_absolute_error_gt_20_fraction": float(count_error["absolute_error_gt_20_fraction"]),
        "val_under_error_lt_minus_10_fraction": float(count_error["under_error_lt_minus_10_fraction"]),
        "val_under_error_lt_minus_20_fraction": float(count_error["under_error_lt_minus_20_fraction"]),
        "val_over_error_gt_10_fraction": float(count_error["over_error_gt_10_fraction"]),
        "val_over_error_gt_20_fraction": float(count_error["over_error_gt_20_fraction"]),
    }


def selection_mechanism_audit(candidate: dict, reference: dict) -> dict:
    """Explain an independently selected E45 recipe against a selected control."""
    candidate_row = scored_checkpoint(Path(candidate["run_dir"]), int(candidate["epoch"]))
    reference_row = scored_checkpoint(Path(reference["run_dir"]), int(reference["epoch"]))
    per_class = {
        metric: {
            name: float(candidate_row[field][name] - reference_row[field][name])
            for name in candidate_row[field].keys() & reference_row[field].keys()
        }
        for metric, field in (
            ("PQ", "val_per_class_PQ"),
            ("DQ", "val_per_class_DQ"),
            ("SQ", "val_per_class_SQ"),
        )
    }
    per_source = {
        source: {
            metric: float(candidate_row["val_per_source"][source][metric] - reference_row["val_per_source"][source][metric])
            for metric in ("mPQ+", "mDQ+", "mSQ+")
        }
        for source in candidate_row["val_per_source"].keys() & reference_row["val_per_source"].keys()
    }
    candidate_confusion = candidate_row.get("val_instance_type_confusion") or {}
    reference_confusion = reference_row.get("val_instance_type_confusion") or {}
    confusion_delta = {
        key: int(candidate_confusion[key] - reference_confusion[key])
        for key in ("geometry_matched", "correctly_typed", "missed_truth", "spurious_prediction")
        if key in candidate_confusion and key in reference_confusion
    }
    if (
        candidate_confusion.get("matched_type_accuracy") is not None
        and reference_confusion.get("matched_type_accuracy") is not None
    ):
        confusion_delta["matched_type_accuracy"] = float(
            candidate_confusion["matched_type_accuracy"] - reference_confusion["matched_type_accuracy"]
        )
    major_sources_safe = all(
        per_source.get(source, {}).get(metric, -np.inf) > -0.01
        for source in ("crag", "dpath", "glas")
        for metric in ("mPQ+", "mDQ+", "mSQ+")
    )
    return {
        "candidate": {
            "learning_rate": float(candidate["learning_rate"]),
            "epoch": int(candidate["epoch"]),
            "run_dir": candidate["run_dir"],
        },
        "reference": {
            "learning_rate": float(reference["learning_rate"]),
            "epoch": int(reference["epoch"]),
            "run_dir": reference["run_dir"],
        },
        "per_class_delta": per_class,
        "per_source_delta": per_source,
        "confusion_delta": confusion_delta,
        "major_sources_safe": bool(major_sources_safe),
    }


def selection_boundary_audit(selection: dict, learning_rates: list[float], max_epoch: int) -> dict:
    rates = np.asarray(sorted(set(learning_rates)), dtype=np.float64)
    selected_rate = float(selection["learning_rate"])
    at_lower_lr = bool(np.isclose(selected_rate, rates[0]))
    at_upper_lr = bool(np.isclose(selected_rate, rates[-1]))
    at_horizon = int(selection["epoch"]) >= int(max_epoch)
    return {
        "at_lower_learning_rate_boundary": at_lower_lr,
        "at_upper_learning_rate_boundary": at_upper_lr,
        "at_scored_horizon_boundary": at_horizon,
        "requires_learning_rate_expansion": bool(at_lower_lr or at_upper_lr),
        "requires_horizon_extension": bool(at_horizon),
        "requires_boundary_confirmation": bool(at_lower_lr or at_upper_lr or at_horizon),
    }


def lower_fraction_promotion_audit(
    candidate: dict,
    uniform: dict,
    class_half: dict,
    boundary: dict | None = None,
    uniform_replacement: dict | None = None,
    uniform_replacement_boundary: dict | None = None,
    major_sources_safe: bool = True,
) -> dict:
    """Audit E45 separately for mPQ promotion and count-tail repair."""
    metric_delta = {
        "mPQ+": float(candidate["val_mPQ+"] - uniform["val_mPQ+"]),
        "mDQ+": float(candidate["val_mDQ+"] - uniform["val_mDQ+"]),
        "mSQ+": float(candidate["val_mSQ+"] - uniform["val_mSQ+"]),
    }
    practical_metric_gate = (
        metric_delta["mPQ+"] > 0.003
        and metric_delta["mDQ+"] > 0.003
        and metric_delta["mSQ+"] > -0.005
    )
    replacement_metric_delta = None
    replacement_metric_gate = True
    if uniform_replacement is not None:
        replacement_metric_delta = {
            "mPQ+": float(candidate["val_mPQ+"] - uniform_replacement["val_mPQ+"]),
            "mDQ+": float(candidate["val_mDQ+"] - uniform_replacement["val_mDQ+"]),
            "mSQ+": float(candidate["val_mSQ+"] - uniform_replacement["val_mSQ+"]),
        }
        replacement_metric_gate = (
            replacement_metric_delta["mPQ+"] > 0.003
            and replacement_metric_delta["mDQ+"] > 0.003
            and replacement_metric_delta["mSQ+"] > -0.005
        )
    metric_gate = practical_metric_gate and replacement_metric_gate and major_sources_safe
    tail_keys = (
        "val_MAE",
        "val_mean_signed_error",
        "val_absolute_error_gt_10_fraction",
        "val_absolute_error_gt_20_fraction",
        "val_under_error_lt_minus_10_fraction",
        "val_under_error_lt_minus_20_fraction",
        "val_over_error_gt_10_fraction",
        "val_over_error_gt_20_fraction",
    )
    tail_delta = {key: float(candidate[key] - class_half[key]) for key in tail_keys}
    absolute_bias_delta = float(
        abs(candidate["val_mean_signed_error"]) - abs(class_half["val_mean_signed_error"])
    )
    improves_large_tail = (
        tail_delta["val_absolute_error_gt_10_fraction"] < 0
        or tail_delta["val_absolute_error_gt_20_fraction"] < 0
    )
    tail_gate = (
        absolute_bias_delta < 0
        and improves_large_tail
        and tail_delta["val_MAE"] <= 0.25
        and tail_delta["val_absolute_error_gt_10_fraction"] <= 0.005
        and tail_delta["val_absolute_error_gt_20_fraction"] <= 0.005
    )
    frontier_delta = {
        "mPQ+": float(candidate["val_mPQ+"] - class_half["val_mPQ+"]),
        "mDQ+": float(candidate["val_mDQ+"] - class_half["val_mDQ+"]),
    }
    near_class_half_frontier = frontier_delta["mPQ+"] >= -0.005 and frontier_delta["mDQ+"] >= -0.005
    requires_boundary_confirmation = bool(
        (boundary and boundary["requires_boundary_confirmation"])
        or (
            uniform_replacement_boundary
            and uniform_replacement_boundary["requires_boundary_confirmation"]
        )
    )
    return {
        "metric_delta_vs_independently_selected_uniform": metric_delta,
        "passes_practical_uniform_metric_gate": bool(practical_metric_gate),
        "metric_delta_vs_independently_selected_uniform_replacement": replacement_metric_delta,
        "passes_uniform_replacement_mechanism_gate": bool(replacement_metric_gate),
        "major_sources_safe_vs_both_controls": bool(major_sources_safe),
        "passes_uniform_metric_gate": bool(metric_gate),
        "passes_mpq_candidate_gate": bool(metric_gate),
        "directional_tail_delta_vs_class_0.5": tail_delta,
        "absolute_mean_signed_error_delta_vs_class_0.5": absolute_bias_delta,
        "passes_tail_gate_vs_class_0.5": bool(tail_gate),
        "metric_delta_vs_class_0.5": frontier_delta,
        "within_0.005_of_class_0.5_mPQ_and_mDQ": bool(near_class_half_frontier),
        "selection_boundary": boundary,
        "uniform_replacement_selection_boundary": uniform_replacement_boundary,
        "passes_metric_and_tail_gates": bool(metric_gate and tail_gate),
        "advances_as_mpq_candidate": bool(metric_gate and not requires_boundary_confirmation),
        "promotable_over_class_0.5_for_mpq": bool(
            metric_gate and near_class_half_frontier and not requires_boundary_confirmation
        ),
        # These count-tail fields are a separate R2/reliability screen. A failed
        # tail gate must not veto an otherwise valid mPQ candidate.
        "advances_to_tail_repair_schedule_screen": bool(
            metric_gate and tail_gate and not requires_boundary_confirmation
        ),
        "safer_pareto_replacement_for_class_0.5": bool(
            metric_gate and tail_gate and near_class_half_frontier and not requires_boundary_confirmation
        ),
        # Backward-compatible aliases for older summaries and dashboard code.
        "advances_to_schedule_screen": bool(metric_gate and tail_gate and not requires_boundary_confirmation),
        "promotable_over_class_0.5": bool(
            metric_gate and tail_gate and near_class_half_frontier and not requires_boundary_confirmation
        ),
    }


def refinement_gate(intervention: dict, control: dict) -> dict:
    selection = intervention["families"]["e37_class_0.5"]["selected_mPQ+"]
    learning_rate = float(selection["learning_rate"])
    epoch = int(selection["epoch"])
    candidate = scored_checkpoint(Path(selection["run_dir"]), epoch)
    control_paths = [Path(path) for rate, path in control["runs"].items() if np.isclose(float(rate), learning_rate)]
    if len(control_paths) != 1:
        raise RuntimeError(f"expected one matched control at LR {learning_rate:g}")
    matched = scored_checkpoint(control_paths[0], epoch)
    delta = {
        "mPQ+": float(candidate["val_mPQ+"] - matched["val_mPQ+"]),
        "mDQ+": float(
            component(candidate, "val_mDQ+", "val_per_class_DQ")
            - component(matched, "val_mDQ+", "val_per_class_DQ")
        ),
        "mSQ+": float(
            component(candidate, "val_mSQ+", "val_per_class_SQ")
            - component(matched, "val_mSQ+", "val_per_class_SQ")
        ),
        "R2": float(candidate["val_R2"] - matched["val_R2"]),
    }
    passes = delta["mPQ+"] > 0.003 and delta["mDQ+"] > 0.003 and delta["mSQ+"] > -0.005
    return {
        "predeclared_rule": "same-LR/checkpoint mPQ+ and mDQ+ > +0.003, with mSQ+ > -0.005",
        "learning_rate": learning_rate,
        "epoch": epoch,
        "delta": delta,
        "passes": bool(passes),
    }


def sampler_prior_promotion_audit(report: dict) -> dict:
    """Gate sampler-exposure prior correction for mPQ with source-held-out evidence."""
    selected = report["selected"]["mPQ+"]
    selected_delta_raw = selected["delta_vs_raw"]
    selected_delta_pooled = selected["delta_vs_pooled_strength_0"]
    held = report["leave_one_source_out"]["mPQ+"]
    held_delta_raw = held["delta_vs_raw"]
    held_delta_pooled = held["delta_vs_pooled_strength_0"]
    pooled_zero = report["pooled_instance_strength_0"]
    held_metrics = held["pooled_out_of_source"]

    def typed_gate(delta: dict) -> bool:
        return delta["mPQ+"] > 0.003 and delta["mDQ+"] > 0.003 and delta["mSQ+"] > -0.005

    error_delta = {
        key: float(held_metrics["count_error"][key] - pooled_zero["count_error"][key])
        for key in ("MAE", "mean_signed_error", "absolute_error_gt_10_fraction", "absolute_error_gt_20_fraction")
    }
    tails_safe = (
        error_delta["MAE"] <= 0.25
        and error_delta["absolute_error_gt_10_fraction"] <= 0.005
        and error_delta["absolute_error_gt_20_fraction"] <= 0.005
    )
    source_deltas = {
        source: {
            metric: float(
                held_metrics["per_source"][source][metric]
                - pooled_zero["per_source"][source][metric]
            )
            for metric in ("mPQ+", "mDQ+", "mSQ+")
        }
        for source in held_metrics.get("per_source", {}).keys() & pooled_zero.get("per_source", {}).keys()
    }
    major_sources_safe = all(
        source_deltas.get(source, {}).get(metric, -np.inf) > -0.01
        for source in ("crag", "dpath", "glas")
        for metric in ("mPQ+", "mDQ+", "mSQ+")
    )
    full_gate = typed_gate(selected_delta_raw) and typed_gate(selected_delta_pooled)
    held_gate = typed_gate(held_delta_raw) and typed_gate(held_delta_pooled)
    stable = bool(held["stable_within_one_grid_step"])
    return {
        "interpretation": (
            "This corrects artificial class exposure from replacement sampling back toward the natural "
            "development-training nucleus prior. It is not supervised locked-test prior estimation. "
            "Count tails are reported for the independent R2/reliability track and do not veto mPQ promotion."
        ),
        "selected_strength": float(selected["strength"]),
        "full_validation_delta_vs_raw": selected_delta_raw,
        "full_validation_delta_vs_pooled_strength_0": selected_delta_pooled,
        "source_excluded_delta_vs_raw": held_delta_raw,
        "source_excluded_delta_vs_pooled_strength_0": held_delta_pooled,
        "selected_strength_excluding_each_source": held["selected_strength_excluding_each_source"],
        "strength_stable_within_one_grid_step": stable,
        "source_excluded_count_error_delta_vs_pooled_strength_0": error_delta,
        "source_excluded_typed_delta_by_source_vs_pooled_strength_0": source_deltas,
        "source_excluded_mPQ_delta_by_source_vs_pooled_strength_0": {
            source: values["mPQ+"] for source, values in source_deltas.items()
        },
        "full_typed_gate": bool(full_gate),
        "source_excluded_typed_gate": bool(held_gate),
        "count_tails_safe": bool(tails_safe),
        "major_sources_safe": bool(major_sources_safe),
        "advances_as_sampler_prior_correction": bool(
            full_gate and held_gate and stable and major_sources_safe
        ),
    }


def run_candidate(
    args: argparse.Namespace,
    fraction: float,
    learning_rate: float,
    uniform_replacement: bool = False,
) -> Path:
    fraction_label = f"{fraction:g}".replace(".", "p")
    lr_label = f"{learning_rate:.0e}".replace("e-0", "e-")
    family = "uniform_replacement" if uniform_replacement else f"class_{fraction_label}"
    outdir = args.out_root / family / f"lr_{lr_label}"
    if (outdir / "summary.json").exists():
        print(f"class-sampling refinement complete, skipping: {outdir}", flush=True)
        return outdir
    archived = archive_incomplete_persistent_run(outdir)
    if archived is not None:
        print(f"archived incomplete persistent-worker run: {archived}", flush=True)
    command = [
        sys.executable,
        "scripts/train_hovernet_our_split.py",
        "--prepared", str(args.prepared),
        "--backbone", str(args.backbone),
        "--train-ids", str(args.train_ids),
        "--val-ids", str(args.val_ids),
        "--outdir", str(outdir),
        "--learning-rate", str(learning_rate),
        "--class-sampling-fraction", str(fraction),
        "--epochs-phase1", str(args.epochs),
        "--epochs-phase2", "0",
        "--batch-size", str(args.batch_size),
        "--val-batch-size", str(args.val_batch_size),
        "--workers", str(args.workers),
        "--metric-every", str(args.metric_every),
        "--seed", str(args.seed),
        "--amp", "none",
        "--device", "cuda:0",
    ]
    if uniform_replacement:
        command.append("--uniform-replacement-sampling")
    run_checked(command)
    return outdir


def backfill_selected(
    args: argparse.Namespace,
    selected_by_fraction: dict[str, dict],
) -> list[dict]:
    requested: dict[Path, dict] = {}
    for fraction, selections in selected_by_fraction.items():
        for metric, key, checkpoint_name in (
            ("R2", "selected_R2", "best_r2.pth"),
            ("mPQ+", "selected_mPQ+", "best_mpq.pth"),
        ):
            selection = selections[key]
            checkpoint = Path(selection["run_dir"]) / checkpoint_name
            item = requested.setdefault(checkpoint, {"fraction": fraction, "metrics": []})
            item["metrics"].append(metric)

    diagnostic_root = args.out_root / "diagnostics"
    diagnostic_root.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for checkpoint, request in requested.items():
        metric_label = "_and_".join(metric.replace("+", "plus") for metric in request["metrics"])
        fraction_label = request["fraction"].replace(".", "p")
        output = diagnostic_root / f"class_{fraction_label}_{metric_label}.json"
        prediction_output = output.with_name(f"{output.stem}_predictions.npz")
        if not (output.exists() and prediction_output.exists()):
            run_checked([
                sys.executable,
                "scripts/backfill_hovernet_validation_diagnostics.py",
                "--checkpoint", str(checkpoint),
                "--prepared", str(args.prepared),
                "--val-ids", str(args.val_ids),
                "--out-json", str(output),
                "--workers", str(args.workers),
                "--device", "cuda:0",
                "--update-curve",
            ])
        item = {
            "fraction": float(request["fraction"]),
            "metrics": request["metrics"],
            "diagnostic_json": str(output),
            "prediction_artifact": str(prediction_output),
        }
        if "mPQ+" in request["metrics"]:
            correction = output.with_name(f"{output.stem}_sampler_prior.json")
            prior_payload = None
            if correction.exists():
                try:
                    prior_payload = json.loads(correction.read_text())
                    corrected_path = prior_payload.get("selected_mPQ_corrected_prediction_artifact")
                    if corrected_path is None or not Path(corrected_path).exists():
                        prior_payload = None
                except (OSError, json.JSONDecodeError):
                    prior_payload = None
            if prior_payload is None:
                run_checked([
                    sys.executable,
                    "scripts/sweep_hovernet_sampler_prior.py",
                    "--prepared", str(args.prepared),
                    "--diagnostic-json", str(output),
                    "--train-ids", str(args.train_ids),
                    "--out-report", str(correction),
                ])
                prior_payload = json.loads(correction.read_text())
            item["sampler_prior_report"] = str(correction)
            corrected_artifact = Path(prior_payload["selected_mPQ_corrected_prediction_artifact"])
            if not corrected_artifact.exists():
                raise FileNotFoundError(f"sampler-prior corrected artifact is missing: {corrected_artifact}")
            item["sampler_prior_corrected_prediction_artifact"] = str(corrected_artifact)
        artifacts.append(item)
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--train-ids", type=Path, required=True)
    parser.add_argument("--val-ids", type=Path, required=True)
    parser.add_argument("--control-summary", type=Path, required=True)
    parser.add_argument("--intervention-summary", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--val-batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--metric-every", type=int, default=2)
    parser.add_argument("--seed", type=int, default=206)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.1])
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[3e-5, 1e-4, 3e-4])
    args = parser.parse_args()

    wait_for_json(args.wait_for, args.poll_seconds)
    control = wait_for_json(args.control_summary, args.poll_seconds)
    intervention = wait_for_json(args.intervention_summary, args.poll_seconds)
    gate = refinement_gate(intervention, control)
    if not gate["passes"]:
        report = {
            "status": "skipped by exact E37 causal gate",
            "evaluation_set": "711-patch source-group-disjoint development validation",
            "gate": gate,
            "test_evaluated": False,
        }
        args.out_root.mkdir(parents=True, exist_ok=True)
        output = args.out_root / "e37_class_fraction_summary.json"
        output.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        return
    replacement_run_dirs = {
        learning_rate: run_candidate(args, 0.0, learning_rate, uniform_replacement=True)
        for learning_rate in args.learning_rates
    }
    replacement_rows = [
        row
        for learning_rate, run_dir in replacement_run_dirs.items()
        for row in scored_rows(run_dir, 0.0, learning_rate)
    ]
    replacement_selected = {
        "selected_R2": select(replacement_rows, "val_R2"),
        "selected_mPQ+": select(replacement_rows, "val_mPQ+"),
    }
    replacement_selection_boundaries = {
        metric: selection_boundary_audit(selection, args.learning_rates, args.epochs)
        for metric, selection in replacement_selected.items()
    }
    run_dirs: dict[tuple[float, float], Path] = {}
    for fraction in args.fractions:
        if not 0 < fraction < 0.5:
            raise ValueError("refinement fractions must lie strictly between 0 and the E37 0.5 pilot")
        for learning_rate in args.learning_rates:
            run_dirs[(fraction, learning_rate)] = run_candidate(args, fraction, learning_rate)

    all_rows = [
        row
        for (fraction, learning_rate), run_dir in run_dirs.items()
        for row in scored_rows(run_dir, fraction, learning_rate)
    ]
    selected_by_fraction = {}
    for fraction in args.fractions:
        rows = [row for row in all_rows if row["class_sampling_fraction"] == fraction]
        selected_by_fraction[str(fraction)] = {
            "selected_R2": select(rows, "val_R2"),
            "selected_mPQ+": select(rows, "val_mPQ+"),
        }
    selected_r2 = select(all_rows, "val_R2")
    selected_mpq = select(all_rows, "val_mPQ+")
    control_r2 = control["selected_R2"]
    control_mpq = selection_with_full_metrics(control["selected_mPQ+"])
    class_half_mpq = selection_with_full_metrics(intervention["families"]["e37_class_0.5"]["selected_mPQ+"])
    mechanism_audits = {
        fraction: {
            "vs_independently_selected_uniform_no_replacement": selection_mechanism_audit(
                selections["selected_mPQ+"], control_mpq
            ),
            "vs_independently_selected_uniform_replacement": selection_mechanism_audit(
                selections["selected_mPQ+"], replacement_selected["selected_mPQ+"]
            ),
        }
        for fraction, selections in selected_by_fraction.items()
    }
    promotion_audits = {
        fraction: lower_fraction_promotion_audit(
            selections["selected_mPQ+"],
            control_mpq,
            class_half_mpq,
            selection_boundary_audit(selections["selected_mPQ+"], args.learning_rates, args.epochs),
            uniform_replacement=replacement_selected["selected_mPQ+"],
            uniform_replacement_boundary=replacement_selection_boundaries["selected_mPQ+"],
            major_sources_safe=bool(
                mechanism_audits[fraction]["vs_independently_selected_uniform_no_replacement"]["major_sources_safe"]
                and mechanism_audits[fraction]["vs_independently_selected_uniform_replacement"]["major_sources_safe"]
            ),
        )
        for fraction, selections in selected_by_fraction.items()
    }
    report = {
        "status": "complete",
        "protocol": (
            "Validation-only E37 intensity refinement after the 0.5 fraction passed the initial mPQ/DQ gate. "
            "A uniform-with-replacement LR grid separates replacement/duplication regularization from class weighting. "
            "Each lower fraction receives the full LR bracket and denser two-epoch scoring; locked test is refused."
        ),
        "evaluation_set": "711-patch source-group-disjoint development validation",
        "seed": args.seed,
        "gate": gate,
        "metric_every": args.metric_every,
        "runs": {
            **{
                f"uniform_replacement,lr={learning_rate:g}": str(path)
                for learning_rate, path in sorted(replacement_run_dirs.items())
            },
            **{
                f"fraction={fraction:g},lr={learning_rate:g}": str(path)
                for (fraction, learning_rate), path in sorted(run_dirs.items())
            },
        },
        "selected_by_fraction": selected_by_fraction,
        "selected_R2": selected_r2,
        "selected_mPQ+": selected_mpq,
        "matched_uniform_control": control,
        "uniform_replacement_mechanism_control": replacement_selected,
        "uniform_replacement_selection_boundaries": replacement_selection_boundaries,
        "class_0.5_selected_mPQ+_reference": class_half_mpq,
        "promotion_audits_by_fraction": promotion_audits,
        "mechanism_audits_by_fraction": mechanism_audits,
        "independently_selected_delta_vs_uniform": {
            "R2": float(selected_r2["val_R2"] - control_r2["val_R2"]),
            "mPQ+": float(selected_mpq["val_mPQ+"] - control_mpq["val_mPQ+"]),
            "mDQ+": float(selected_mpq["val_mDQ+"] - control_mpq["val_mDQ+"]),
            "mSQ+": float(selected_mpq["val_mSQ+"] - control_mpq["val_mSQ+"]),
        },
        "independently_selected_delta_vs_uniform_replacement": {
            "R2": float(selected_r2["val_R2"] - replacement_selected["selected_R2"]["val_R2"]),
            "mPQ+": float(selected_mpq["val_mPQ+"] - replacement_selected["selected_mPQ+"]["val_mPQ+"]),
            "mDQ+": float(selected_mpq["val_mDQ+"] - replacement_selected["selected_mPQ+"]["val_mDQ+"]),
            "mSQ+": float(selected_mpq["val_mSQ+"] - replacement_selected["selected_mPQ+"]["val_mSQ+"]),
        },
        "guardrail": (
            "Select R2 and mPQ independently. An mPQ candidate requires mPQ and mDQ improvements above 0.003 versus "
            "both independently selected ordinary uniform and uniform-with-replacement controls, mSQ deltas above "
            "-0.005 versus both, no mPQ/mDQ/mSQ regression below -0.01 in CRAG, DPath, or GLaS versus either "
            "control, and confirmed non-boundary candidate and replacement-control selections; count "
            "tails do not veto that route. A separate tail-repair/Pareto screen additionally requires improved "
            "absolute signed bias, improvement in at least one of the >10/>20 outlier proportions, and no material "
            "MAE or tail regression (>0.25 counts or >0.005). Replacing class-0.5 for mPQ also requires mPQ and mDQ "
            "within 0.005 of its frontier. Sampler-prior correction uses source-excluded typed metrics and strength "
            "stability; directional count tails remain reported only for the independent R2/reliability track."
        ),
        "test_evaluated": False,
    }
    report["selected_checkpoint_artifacts"] = backfill_selected(args, selected_by_fraction)
    report["sampler_prior_correction_audits"] = []
    for artifact in report["selected_checkpoint_artifacts"]:
        prior_path = artifact.get("sampler_prior_report")
        if prior_path is None:
            continue
        prior_report = json.loads(Path(prior_path).read_text())
        report["sampler_prior_correction_audits"].append({
            "fraction": artifact["fraction"],
            "report": prior_path,
            "audit": sampler_prior_promotion_audit(prior_report),
        })
    args.out_root.mkdir(parents=True, exist_ok=True)
    output = args.out_root / "e37_class_fraction_summary.json"
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
