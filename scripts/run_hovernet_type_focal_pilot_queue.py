#!/usr/bin/env python3
"""Run isolated HoVer-Net type-loss ablations after the current validation queue."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cpath_conic.queue_integrity import archive_incomplete_persistent_run
from scripts.run_hovernet_instance_type_pilot_queue import (
    exact_checkpoint_comparison,
    promotion_audit,
)


def wait_for_json(path: Path, poll_seconds: int) -> None:
    while True:
        if path.exists():
            try:
                json.loads(path.read_text())
                time.sleep(max(10, poll_seconds))
                return
            except (OSError, json.JSONDecodeError):
                pass
        print(f"waiting for completed prerequisite: {path}", flush=True)
        time.sleep(max(1, poll_seconds))


def scored_rows(run_dir: Path, family: str, learning_rate: float) -> list[dict]:
    path = run_dir / "training_curve.json"
    if not path.exists():
        return []
    rows = json.loads(path.read_text())
    return [
        {
            "family": family,
            "learning_rate": learning_rate,
            "epoch": int(row["epoch"]),
            "val_R2": float(row["val_R2"]),
            "val_mPQ+": float(row["val_mPQ+"]),
            "val_mDQ+": float(row["val_mDQ+"]),
            "val_mSQ+": float(row["val_mSQ+"]),
            "run_dir": str(run_dir),
        }
        for row in rows
        if row.get("val_R2") is not None and row.get("val_mPQ+") is not None
    ]


def select(candidates: list[dict], metric: str) -> dict:
    finite = [row for row in candidates if np.isfinite(row[metric])]
    if not finite:
        raise RuntimeError(f"no finite type-loss candidates for {metric}")
    return max(finite, key=lambda row: (row[metric], row["epoch"]))


def validate_control_seed(control: dict, expected_seed: int) -> int:
    """Refuse a nominally matched control whose completed runs use another seed."""
    seeds = set()
    for run_dir in control["runs"].values():
        summary_path = Path(run_dir) / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"matched-control summary is missing: {summary_path}")
        payload = json.loads(summary_path.read_text())
        seed = payload.get("args", {}).get("seed", payload.get("seed"))
        if seed is None:
            raise KeyError(f"matched-control run does not declare its seed: {summary_path}")
        seeds.add(int(seed))
    if seeds != {int(expected_seed)}:
        raise ValueError(
            f"type-loss seed {expected_seed} requires a seed-matched control; found {sorted(seeds)}"
        )
    return next(iter(seeds))


def selection_boundaries(selection: dict, learning_rates: list[float], max_epoch: int) -> dict:
    rates = np.asarray(sorted(set(learning_rates)), dtype=np.float64)
    rate = float(selection["learning_rate"])
    at_lower_lr = bool(np.isclose(rate, rates[0]))
    at_upper_lr = bool(np.isclose(rate, rates[-1]))
    at_horizon = int(selection["epoch"]) >= int(max_epoch)
    return {
        "at_lower_learning_rate_boundary": at_lower_lr,
        "at_upper_learning_rate_boundary": at_upper_lr,
        "at_scored_horizon_boundary": at_horizon,
        "requires_learning_rate_expansion": bool(at_lower_lr or at_upper_lr),
        "requires_horizon_extension": bool(at_horizon),
        "requires_boundary_confirmation": bool(at_lower_lr or at_upper_lr or at_horizon),
    }


def independently_selected_delta(selected_r2: dict, selected_mpq: dict, control: dict) -> dict:
    return {
        "R2": float(selected_r2["val_R2"] - control["selected_R2"]["val_R2"]),
        "mPQ+": float(selected_mpq["val_mPQ+"] - control["selected_mPQ+"]["val_mPQ+"]),
        "mDQ+": float(selected_mpq["val_mDQ+"] - control["selected_mPQ+"]["val_mDQ+"]),
        "mSQ+": float(selected_mpq["val_mSQ+"] - control["selected_mPQ+"]["val_mSQ+"]),
    }


def passes_segmentation_gate(delta: dict) -> bool:
    """Use the same typed-detection/SQ guardrail as the E43 causal gate."""
    return (
        float(delta["mPQ+"]) > 0.003
        and float(delta["mDQ+"]) > 0.003
        and float(delta["mSQ+"]) > -0.005
    )


def run_candidate(
    args: argparse.Namespace,
    family: str,
    loss_arguments: list[str],
    learning_rate: float,
) -> Path:
    lr_label = f"{learning_rate:.0e}".replace("e-0", "e-")
    outdir = args.out_root / family / f"lr_{lr_label}"
    if (outdir / "summary.json").exists():
        print(f"type-loss pilot complete, skipping: {outdir}", flush=True)
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
        "--epochs-phase1", str(args.epochs),
        "--epochs-phase2", "0",
        "--batch-size", str(args.batch_size),
        "--val-batch-size", str(args.val_batch_size),
        "--workers", str(args.workers),
        "--metric-every", "5",
        "--seed", str(args.seed),
        "--amp", "none",
        "--device", "cuda:0",
        *loss_arguments,
    ]
    print("running:", " ".join(command), flush=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise SystemExit(f"type-loss candidate failed with exit code {completed.returncode}")
    return outdir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--train-ids", type=Path, required=True)
    parser.add_argument("--val-ids", type=Path, required=True)
    parser.add_argument("--control-summary", type=Path, required=True)
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
    control = json.loads(args.control_summary.read_text())
    control_seed = validate_control_seed(control, args.seed)
    families = {
        "weight_rho3": ["--type-class-balance-rho", "3"],
        "focal_gamma2": ["--type-focal-gamma", "2"],
        "weight_rho3_focal_gamma2": [
            "--type-class-balance-rho", "3",
            "--type-focal-gamma", "2",
        ],
    }
    report = {
        "protocol": (
            "Validation-only, seed-205 isolated HoVer type-loss ablations. Fixed complement-frequency "
            "weighting, focal emphasis, and their pure combination each receive the same three-LR bracket "
            "and 10-epoch optimizer-step budget. Label smoothing is a conditional marginal add-on to the "
            "validation-selected pure combination, so it cannot confound the weighting/focal attribution. "
            "R2 and mPQ+ select independently; no test inference."
        ),
        "evaluation_set": "711-patch source-group-disjoint development validation",
        "frequency_source": "development-training nucleus counts only",
        "seed": args.seed,
        "matched_control_seed": control_seed,
        "families": {},
        "matched_control": control,
        "promotion_guard": (
            "Require a seed-matched DQ gain, especially for neutrophil/eosinophil, without unstable false-positive "
            "counts, source-specific collapse, SQ loss, or a selected LR on an unexplored boundary. Test sampling+loss "
            "composition only after both isolated interventions pass."
        ),
    }
    for family, loss_arguments in families.items():
        runs: dict[float, Path] = {}
        for learning_rate in args.learning_rates:
            runs[learning_rate] = run_candidate(args, family, loss_arguments, learning_rate)
        candidates = [
            row
            for learning_rate, run_dir in runs.items()
            for row in scored_rows(run_dir, family, learning_rate)
        ]
        selected_r2 = select(candidates, "val_R2")
        selected_mpq = select(candidates, "val_mPQ+")
        delta = independently_selected_delta(selected_r2, selected_mpq, control)
        exact_r2 = exact_checkpoint_comparison(selected_r2, control)
        exact_mpq = exact_checkpoint_comparison(selected_mpq, control)
        report["families"][family] = {
            "loss_arguments": loss_arguments,
            "runs": {str(rate): str(path) for rate, path in runs.items()},
            "selected_R2": selected_r2,
            "selected_mPQ+": selected_mpq,
            "selection_boundaries": {
                "R2": selection_boundaries(selected_r2, args.learning_rates, args.epochs),
                "mPQ+": selection_boundaries(selected_mpq, args.learning_rates, args.epochs),
            },
            "independently_selected_delta": delta,
            "exact_matched_control": {"R2": exact_r2, "mPQ+": exact_mpq},
            "promotion_audit": promotion_audit(exact_mpq, delta),
        }
        args.out_root.mkdir(parents=True, exist_ok=True)
        (args.out_root / "e44_type_focal_progress.json").write_text(json.dumps(report, indent=2))

    # Label smoothing is deliberately not folded into the primary combination:
    # doing so would leave any win attributable to either smoothing or synergy.
    # Test it only as a marginal add-on, and only if the pure combination clears
    # the same validation-only mPQ/DQ/SQ gate used elsewhere in this queue.
    pure_combination = report["families"]["weight_rho3_focal_gamma2"]
    smoothing_family = "weight_rho3_focal_gamma2_smooth005"
    smoothing_gate = {
        "rule": (
            "Pure weighting+focal must improve mPQ+ and mDQ+ by >0.003 with mSQ+ delta >-0.005 "
            "both at its selected exact-control checkpoint and against independently selected controls."
        ),
        "pure_combination_delta": pure_combination["independently_selected_delta"],
        "pure_combination_exact_delta": pure_combination["exact_matched_control"]["mPQ+"]["delta"],
        "passes": pure_combination["promotion_audit"]["typed_signal"],
    }
    if smoothing_gate["passes"]:
        selected_rates = sorted({
            float(pure_combination["selected_R2"]["learning_rate"]),
            float(pure_combination["selected_mPQ+"]["learning_rate"]),
        })
        smoothing_arguments = [
            "--type-class-balance-rho", "3",
            "--type-focal-gamma", "2",
            "--type-label-smoothing", "0.05",
        ]
        runs = {
            learning_rate: run_candidate(
                args, smoothing_family, smoothing_arguments, learning_rate
            )
            for learning_rate in selected_rates
        }
        candidates = [
            row
            for learning_rate, run_dir in runs.items()
            for row in scored_rows(run_dir, smoothing_family, learning_rate)
        ]
        selected_r2 = select(candidates, "val_R2")
        selected_mpq = select(candidates, "val_mPQ+")
        delta = independently_selected_delta(selected_r2, selected_mpq, control)
        exact_r2 = exact_checkpoint_comparison(selected_r2, control)
        exact_mpq = exact_checkpoint_comparison(selected_mpq, control)
        report["families"][smoothing_family] = {
            "status": "completed conditional marginal smoothing screen",
            "loss_arguments": smoothing_arguments,
            "gate": smoothing_gate,
            "runs": {str(rate): str(path) for rate, path in runs.items()},
            "selected_R2": selected_r2,
            "selected_mPQ+": selected_mpq,
            "selection_boundaries": {
                "R2": selection_boundaries(selected_r2, args.learning_rates, args.epochs),
                "mPQ+": selection_boundaries(selected_mpq, args.learning_rates, args.epochs),
            },
            "independently_selected_delta": delta,
            "exact_matched_control": {"R2": exact_r2, "mPQ+": exact_mpq},
            "promotion_audit": promotion_audit(exact_mpq, delta),
            "delta_vs_pure_combination_at_independently_selected_endpoints": {
                "R2": float(selected_r2["val_R2"] - pure_combination["selected_R2"]["val_R2"]),
                "mPQ+": float(selected_mpq["val_mPQ+"] - pure_combination["selected_mPQ+"]["val_mPQ+"]),
                "mDQ+": float(selected_mpq["val_mDQ+"] - pure_combination["selected_mPQ+"]["val_mDQ+"]),
                "mSQ+": float(selected_mpq["val_mSQ+"] - pure_combination["selected_mPQ+"]["val_mSQ+"]),
            },
        }
    else:
        report["families"][smoothing_family] = {
            "status": "skipped by predeclared pure-combination causal gate",
            "gate": smoothing_gate,
            "runs": {},
        }
    (args.out_root / "e44_type_focal_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
