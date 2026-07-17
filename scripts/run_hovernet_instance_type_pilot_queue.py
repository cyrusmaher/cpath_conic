#!/usr/bin/env python3
"""Run the gated validation-only E43 one-loss-per-nucleus type pilot."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cpath_conic.queue_integrity import archive_incomplete_persistent_run


def wait_for_json(path: Path, poll_seconds: int) -> dict:
    while True:
        if path.exists():
            try:
                result = json.loads(path.read_text())
                time.sleep(max(10, poll_seconds))
                return result
            except (OSError, json.JSONDecodeError):
                pass
        print(f"waiting for completed prerequisite: {path}", flush=True)
        time.sleep(max(1, poll_seconds))


def enrich_selected_components(selection: dict) -> dict:
    if selection.get("val_mDQ+") is not None and selection.get("val_mSQ+") is not None:
        return selection
    rows = json.loads((Path(selection["run_dir"]) / "training_curve.json").read_text())
    row = next(item for item in rows if int(item["epoch"]) == int(selection["epoch"]))
    enriched = dict(selection)
    for output_key, per_class_key in (("val_mDQ+", "val_per_class_DQ"), ("val_mSQ+", "val_per_class_SQ")):
        value = row.get(output_key)
        if value is None:
            value = np.mean(list(row[per_class_key].values()))
        enriched[output_key] = float(value)
    return enriched


def gate_evidence(e42: dict, e37_intervention: dict, e37_control: dict) -> dict:
    e42_delta = e42.get("independently_selected_delta", {})
    class_selected = enrich_selected_components(
        e37_intervention["families"]["e37_class_0.5"]["selected_mPQ+"]
    )
    control_selected = enrich_selected_components(e37_control["selected_mPQ+"])
    e37_delta = {
        "mPQ+": float(class_selected["val_mPQ+"] - control_selected["val_mPQ+"]),
        "mDQ+": float(class_selected["val_mDQ+"] - control_selected["val_mDQ+"]),
        "mSQ+": float(class_selected["val_mSQ+"] - control_selected["val_mSQ+"]),
    }

    def passes(delta: dict) -> bool:
        return (
            float(delta.get("mPQ+", -np.inf)) > 0.003
            and float(delta.get("mDQ+", -np.inf)) > 0.003
            and float(delta.get("mSQ+", -np.inf)) > -0.005
        )

    return {
        "predeclared_rule": (
            "Run E43 if E42 or class-balanced E37 improves independently selected validation "
            "mPQ+ and mDQ+ by more than 0.003 without reducing mSQ+ by more than 0.005."
        ),
        "E42_delta": e42_delta,
        "E37_class_sampling_delta": e37_delta,
        "E42_passes": passes(e42_delta),
        "E37_passes": passes(e37_delta),
        "passes": passes(e42_delta) or passes(e37_delta),
    }


def scored_rows(run_dir: Path, weight: float, learning_rate: float) -> list[dict]:
    path = run_dir / "training_curve.json"
    if not path.exists():
        return []
    rows = json.loads(path.read_text())
    return [
        {
            "instance_type_loss_weight": weight,
            "learning_rate": learning_rate,
            "epoch": int(row["epoch"]),
            "val_R2": float(row["val_R2"]),
            "val_mPQ+": float(row["val_mPQ+"]),
            "val_mDQ+": float(row["val_mDQ+"]),
            "val_mSQ+": float(row["val_mSQ+"]),
            "val_gt_instance_type_nll": float(row["val_gt_instance_type_nll"]),
            "val_gt_instance_type_entropy": float(row["val_gt_instance_type_entropy"]),
            "run_dir": str(run_dir),
        }
        for row in rows
        if row.get("val_R2") is not None and row.get("val_mPQ+") is not None
    ]


def select(candidates: list[dict], metric: str) -> dict:
    finite = [row for row in candidates if np.isfinite(row[metric])]
    if not finite:
        raise RuntimeError(f"no finite E43 candidates for {metric}")
    return max(finite, key=lambda row: (row[metric], row["epoch"]))


def checkpoint_row(run_dir: Path, epoch: int) -> dict:
    rows = json.loads((run_dir / "training_curve.json").read_text())
    for row in rows:
        if int(row["epoch"]) == int(epoch):
            return {**row, "run_dir": str(run_dir)}
    raise KeyError(f"epoch {epoch} is absent from {run_dir}")


def matched_control_row(control: dict, selection: dict) -> dict:
    target_lr = float(selection["learning_rate"])
    target_epoch = int(selection["epoch"])
    for run_dir in matched_control_run_dirs(control, target_lr):
        rows = json.loads((run_dir / "training_curve.json").read_text())
        return checkpoint_row(run_dir, target_epoch)
    raise KeyError(f"matched control LR {target_lr:g}, epoch {target_epoch} is absent")


def matched_control_run_dirs(control: dict, target_lr: float) -> list[Path]:
    matches = []
    for run_dir_text in control["runs"].values():
        run_dir = Path(run_dir_text)
        rows = json.loads((run_dir / "training_curve.json").read_text())
        if rows and np.isclose(float(rows[0]["decoder_learning_rate"]), target_lr, rtol=0.0, atol=1e-12):
            matches.append(run_dir)
    return matches


def checkpoint_for_epoch(run_dir: Path, epoch: int) -> Path | None:
    """Find a retained checkpoint for an exact historical LR/epoch pair."""
    for name in ("latest.pth", "best_mpq.pth", "best_r2.pth"):
        path = run_dir / name
        if not path.exists():
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint_epoch = int(payload.get("epoch", payload.get("metrics", {}).get("epoch", -1)))
        del payload
        if checkpoint_epoch == int(epoch):
            return path
    return None


def ensure_matched_control_spatial_diagnostic(
    control: dict,
    selection: dict,
    args: argparse.Namespace,
) -> dict:
    """Backfill the new spatial-disagreement metric at the exact control checkpoint."""
    metric = "val_gt_instance_type_spatial_js_disagreement"
    existing = matched_control_row(control, selection)
    if existing.get(metric) is not None:
        return {"status": "already available", "metric": metric}
    target_lr = float(selection["learning_rate"])
    run_dirs = matched_control_run_dirs(control, target_lr)
    if len(run_dirs) != 1:
        return {"status": "held", "reason": f"expected one matched control run, found {len(run_dirs)}"}
    checkpoint = checkpoint_for_epoch(run_dirs[0], int(selection["epoch"]))
    if checkpoint is None:
        return {
            "status": "held",
            "reason": "the exact historical control checkpoint was not retained",
            "learning_rate": target_lr,
            "epoch": int(selection["epoch"]),
        }
    output = args.out_root / (
        f"matched_control_spatial_lr_{target_lr:.0e}_epoch_{int(selection['epoch'])}.json"
    )
    subprocess.run([
        sys.executable,
        "scripts/backfill_hovernet_validation_diagnostics.py",
        "--checkpoint", str(checkpoint),
        "--prepared", str(args.prepared),
        "--val-ids", str(args.val_ids),
        "--out-json", str(output),
        "--workers", str(args.workers),
        "--device", "cuda:0",
        "--update-curve",
    ], check=True)
    refreshed = matched_control_row(control, selection)
    if refreshed.get(metric) is None:
        raise RuntimeError(f"control spatial diagnostic backfill did not produce {metric}")
    return {"status": "complete", "metric": metric, "checkpoint": str(checkpoint), "report": str(output)}


def exact_checkpoint_comparison(candidate_selection: dict, control: dict) -> dict:
    candidate = checkpoint_row(Path(candidate_selection["run_dir"]), int(candidate_selection["epoch"]))
    matched = matched_control_row(control, candidate_selection)
    scalar_keys = (
        "val_R2", "val_mPQ+", "val_mDQ+", "val_mSQ+", "val_bPQ",
        "val_binary_DQ", "val_binary_SQ", "val_AJI+", "val_boundary_F1",
        "val_gt_instance_type_nll", "val_gt_instance_type_target_probability",
        "val_gt_instance_type_entropy", "val_gt_instance_type_mean_pixel_entropy",
        "val_gt_instance_type_spatial_js_disagreement", "val_gt_instance_pixel_type_accuracy",
    )
    candidate_scalars = {key: float(candidate[key]) for key in scalar_keys if candidate.get(key) is not None}
    control_scalars = {key: float(matched[key]) for key in scalar_keys if matched.get(key) is not None}
    delta = {
        key: float(candidate_scalars[key] - control_scalars[key])
        for key in candidate_scalars.keys() & control_scalars.keys()
    }
    candidate_error = candidate.get("val_count_error") or {}
    control_error = matched.get("val_count_error") or {}
    count_error_delta = {
        key: float(candidate_error[key] - control_error[key])
        for key in candidate_error.keys() & control_error.keys()
        if isinstance(candidate_error[key], (int, float)) and isinstance(control_error[key], (int, float))
    }
    source_delta = {}
    candidate_sources = candidate.get("val_per_source") or {}
    control_sources = matched.get("val_per_source") or {}
    for source in sorted(candidate_sources.keys() & control_sources.keys()):
        source_delta[source] = {
            key: float(candidate_sources[source][key] - control_sources[source][key])
            for key in ("R2", "mPQ+", "mDQ+", "mSQ+")
            if candidate_sources[source].get(key) is not None and control_sources[source].get(key) is not None
        }
    candidate_confusion = candidate.get("val_instance_type_confusion") or {}
    control_confusion = matched.get("val_instance_type_confusion") or {}
    confusion_delta = {
        key: int(candidate_confusion[key] - control_confusion[key])
        for key in ("geometry_matched", "correctly_typed", "missed_truth", "spurious_prediction")
        if key in candidate_confusion and key in control_confusion
    }
    if candidate_confusion.get("matched_type_accuracy") is not None and control_confusion.get("matched_type_accuracy") is not None:
        confusion_delta["matched_type_accuracy"] = float(
            candidate_confusion["matched_type_accuracy"] - control_confusion["matched_type_accuracy"]
        )
    return {
        "learning_rate": float(candidate_selection["learning_rate"]),
        "epoch": int(candidate_selection["epoch"]),
        "candidate": candidate_scalars,
        "matched_control": control_scalars,
        "delta": delta,
        "count_error_delta": count_error_delta,
        "source_delta": source_delta,
        "confusion_delta": confusion_delta,
    }


def promotion_audit(exact_mpq: dict, independently_selected_delta: dict) -> dict:
    delta = exact_mpq["delta"]
    count_delta = exact_mpq["count_error_delta"]
    typed_signal = (
        independently_selected_delta["mPQ+"] > 0.003
        and independently_selected_delta["mDQ+"] > 0.003
        and independently_selected_delta["mSQ+"] > -0.005
        and delta.get("val_mPQ+", -np.inf) > 0.003
        and delta.get("val_mDQ+", -np.inf) > 0.003
        and delta.get("val_mSQ+", -np.inf) > -0.005
    )
    geometry_safe = (
        delta.get("val_bPQ", -np.inf) > -0.005
        and delta.get("val_binary_DQ", -np.inf) > -0.005
        and delta.get("val_binary_SQ", -np.inf) > -0.005
    )
    tails_safe = (
        count_delta.get("MAE", np.inf) <= 0.25
        and count_delta.get("absolute_error_gt_10_fraction", np.inf) <= 0.005
        and count_delta.get("absolute_error_gt_20_fraction", np.inf) <= 0.005
    )
    major_source_safe = all(
        exact_mpq["source_delta"].get(source, {}).get(metric, -np.inf) > -0.01
        for source in ("crag", "dpath", "glas")
        for metric in ("mPQ+", "mDQ+", "mSQ+")
    )
    spatial_delta = delta.get("val_gt_instance_type_spatial_js_disagreement")
    spatial_consistency_available = spatial_delta is not None and np.isfinite(spatial_delta)
    # The diagnostic is normalized by log(number of classes). A +0.005
    # tolerance avoids rejecting negligible paired movement while preventing a
    # pooled-probability gain from concealing materially more contradictory
    # pixel predictions inside each GT nucleus.
    spatial_consistency_safe = bool(spatial_consistency_available and spatial_delta <= 0.005)
    return {
        "predeclared_interpretation": (
            "A standalone E43 recipe needs an independently selected and exact-checkpoint typed gain, "
            "safe binary geometry, stable major sources, and safe within-nucleus spatial consistency. "
            "Count tails are reported for mechanism and reliability but do not veto a model selected "
            "specifically for mPQ+. A typed signal that fails binary geometry may advance only to "
            "E46's fixed-geometry type-probability screen."
        ),
        "typed_signal": bool(typed_signal),
        "binary_geometry_safe": bool(geometry_safe),
        "count_tails_safe": bool(tails_safe),
        "major_sources_safe": bool(major_source_safe),
        "spatial_consistency_available": bool(spatial_consistency_available),
        "spatial_consistency_delta": float(spatial_delta) if spatial_consistency_available else None,
        "spatial_consistency_safe": spatial_consistency_safe,
        "provisional_standalone_passes": bool(
            typed_signal and geometry_safe and major_source_safe and spatial_consistency_safe
        ),
        "admit_to_fixed_geometry_type_screen": bool(
            typed_signal and major_source_safe and spatial_consistency_safe
        ),
    }


def selection_boundaries(
    selection: dict,
    learning_rates: list[float],
    weights: list[float],
    max_epoch: int,
) -> dict:
    rates = np.asarray(sorted(set(learning_rates)), dtype=np.float64)
    searched_weights = np.asarray(sorted(set(weights)), dtype=np.float64)
    rate = float(selection["learning_rate"])
    weight = float(selection["instance_type_loss_weight"])
    at_lower_lr = bool(np.isclose(rate, rates[0]))
    at_upper_lr = bool(np.isclose(rate, rates[-1]))
    at_lower_weight = bool(np.isclose(weight, searched_weights[0]))
    at_upper_weight = bool(np.isclose(weight, searched_weights[-1]))
    at_horizon = int(selection["epoch"]) >= int(max_epoch)
    return {
        "at_lower_learning_rate_boundary": at_lower_lr,
        "at_upper_learning_rate_boundary": at_upper_lr,
        "at_lower_auxiliary_weight_boundary": at_lower_weight,
        "at_upper_auxiliary_weight_boundary": at_upper_weight,
        "at_scored_horizon_boundary": at_horizon,
        "requires_learning_rate_expansion": bool(at_lower_lr or at_upper_lr),
        "requires_weight_expansion": bool(at_lower_weight or at_upper_weight),
        "requires_horizon_extension": bool(at_horizon),
        "requires_boundary_confirmation": bool(
            at_lower_lr or at_upper_lr or at_lower_weight or at_upper_weight or at_horizon
        ),
    }


def run_candidate(args: argparse.Namespace, weight: float, learning_rate: float) -> Path:
    weight_label = f"{weight:g}".replace(".", "p")
    lr_label = f"{learning_rate:.0e}".replace("e-0", "e-")
    outdir = args.out_root / f"weight_{weight_label}" / f"lr_{lr_label}"
    if (outdir / "summary.json").exists():
        print(f"E43 pilot complete, skipping: {outdir}", flush=True)
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
        "--instance-type-loss-weight", str(weight),
        "--epochs-phase1", str(args.epochs),
        "--epochs-phase2", "0",
        "--batch-size", str(args.batch_size),
        "--val-batch-size", str(args.val_batch_size),
        "--workers", str(args.workers),
        "--metric-every", "5",
        "--seed", str(args.seed),
        "--amp", "none",
        "--device", "cuda:0",
    ]
    print("running:", " ".join(command), flush=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise SystemExit(f"E43 candidate failed with exit code {completed.returncode}")
    return outdir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--train-ids", type=Path, required=True)
    parser.add_argument("--val-ids", type=Path, required=True)
    parser.add_argument("--control-summary", type=Path, required=True)
    parser.add_argument("--e42-summary", type=Path, required=True)
    parser.add_argument("--e37-intervention-summary", type=Path, required=True)
    parser.add_argument("--e37-control-summary", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--val-batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=205)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[3e-5, 1e-4, 3e-4])
    args = parser.parse_args()

    wait_for_json(args.wait_for, args.poll_seconds)
    e42 = wait_for_json(args.e42_summary, args.poll_seconds)
    e37_intervention = wait_for_json(args.e37_intervention_summary, args.poll_seconds)
    e37_control = wait_for_json(args.e37_control_summary, args.poll_seconds)
    evidence = gate_evidence(e42, e37_intervention, e37_control)
    args.out_root.mkdir(parents=True, exist_ok=True)
    output = args.out_root / "e43_instance_type_summary.json"
    if not evidence["passes"]:
        report = {
            "status": "skipped by predeclared causal gate",
            "evaluation_set": "711-patch source-group-disjoint development validation",
            "gate_evidence": evidence,
            "test_evaluated": False,
        }
        output.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        return

    control = json.loads(args.control_summary.read_text())
    control["selected_R2"] = enrich_selected_components(control["selected_R2"])
    control["selected_mPQ+"] = enrich_selected_components(control["selected_mPQ+"])
    run_dirs: dict[tuple[float, float], Path] = {}
    for learning_rate in args.learning_rates:
        run_dirs[(0.1, learning_rate)] = run_candidate(args, 0.1, learning_rate)
    stage_a = [
        row
        for (weight, learning_rate), run_dir in run_dirs.items()
        for row in scored_rows(run_dir, weight, learning_rate)
    ]
    selected_lrs = {
        float(select(stage_a, "val_R2")["learning_rate"]),
        float(select(stage_a, "val_mPQ+")["learning_rate"]),
    }
    for weight in (0.05, 0.25):
        for learning_rate in sorted(selected_lrs):
            run_dirs[(weight, learning_rate)] = run_candidate(args, weight, learning_rate)

    candidates = [
        row
        for (weight, learning_rate), run_dir in run_dirs.items()
        for row in scored_rows(run_dir, weight, learning_rate)
    ]
    selected_r2 = select(candidates, "val_R2")
    selected_mpq = select(candidates, "val_mPQ+")
    searched_weights = sorted({weight for weight, _ in run_dirs})
    independently_selected_delta = {
        "R2": float(selected_r2["val_R2"] - control["selected_R2"]["val_R2"]),
        "mPQ+": float(selected_mpq["val_mPQ+"] - control["selected_mPQ+"]["val_mPQ+"]),
        "mDQ+": float(selected_mpq["val_mDQ+"] - control["selected_mPQ+"]["val_mDQ+"]),
        "mSQ+": float(selected_mpq["val_mSQ+"] - control["selected_mPQ+"]["val_mSQ+"]),
    }
    spatial_control_backfill = ensure_matched_control_spatial_diagnostic(control, selected_mpq, args)
    exact_r2 = exact_checkpoint_comparison(selected_r2, control)
    exact_mpq = exact_checkpoint_comparison(selected_mpq, control)
    report = {
        "status": "complete",
        "protocol": (
            "Validation-only staged grid. Weight 0.1 brackets LR; weights 0.05 and 0.25 are tested only "
            "at the union of the independently selected R2/mPQ+ LRs. No internal-test inference."
        ),
        "evaluation_set": "711-patch source-group-disjoint development validation",
        "seed": args.seed,
        "gate_evidence": evidence,
        "runs": {f"weight={weight:g},lr={lr:g}": str(path) for (weight, lr), path in sorted(run_dirs.items())},
        "selected_R2": selected_r2,
        "selected_mPQ+": selected_mpq,
        "selection_boundaries": {
            "R2": selection_boundaries(selected_r2, args.learning_rates, searched_weights, args.epochs),
            "mPQ+": selection_boundaries(selected_mpq, args.learning_rates, searched_weights, args.epochs),
        },
        "matched_control": control,
        "independently_selected_delta": independently_selected_delta,
        "exact_matched_control": {"R2": exact_r2, "mPQ+": exact_mpq},
        "matched_control_spatial_diagnostic": spatial_control_backfill,
        "promotion_audit": promotion_audit(exact_mpq, independently_selected_delta),
        "promotion_guard": (
            "Expand any selected weight/LR boundary before promotion. Require mPQ+/DQ improvement without "
            "SQ, within-instance spatial-consistency, binary-geometry, or major-source regression. Report "
            "signed bias and count tails without using the R2 objective to veto an mPQ-specific model."
        ),
    }
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
