#!/usr/bin/env python3
"""Expand E44's upper LR boundary with seed-matched ordinary-loss controls."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cpath_conic.queue_integrity import archive_incomplete_persistent_run
from scripts.run_hovernet_instance_type_pilot_queue import exact_checkpoint_comparison, promotion_audit
from scripts.run_hovernet_type_focal_pilot_queue import (
    independently_selected_delta,
    scored_rows,
    select,
    selection_boundaries,
    validate_control_seed,
)


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


def expansion_recipe(e44: dict) -> tuple[str, list[str], str]:
    """Use smoothing only when it adds a practical typed gain over the pure combination."""
    pure_name = "weight_rho3_focal_gamma2"
    pure = e44["families"][pure_name]
    smooth_name = "weight_rho3_focal_gamma2_smooth005"
    smooth = e44["families"].get(smooth_name)
    if smooth is not None:
        delta = smooth["delta_vs_pure_combination_at_independently_selected_endpoints"]
        if delta["mPQ+"] > 0.003 and delta["mDQ+"] > 0.003 and delta["mSQ+"] > -0.005:
            return smooth_name, list(smooth["loss_arguments"]), "smoothing adds a practical typed gain"
    return pure_name, list(pure["loss_arguments"]), "smoothing does not add a practical typed gain"


def upper_lr_bracket(candidate_rows: list[dict], learning_rate: float) -> dict:
    """Stop an upper-LR sweep once both target metrics turn down."""
    current = [
        row for row in candidate_rows
        if np.isclose(float(row["learning_rate"]), float(learning_rate))
    ]
    lower = [
        row for row in candidate_rows
        if float(row["learning_rate"]) < float(learning_rate)
    ]
    metrics = {}
    for metric in ("val_R2", "val_mPQ+"):
        current_best = max((float(row[metric]) for row in current), default=float("-inf"))
        lower_best = max((float(row[metric]) for row in lower), default=float("-inf"))
        metrics[metric] = {
            "current_best": current_best,
            "best_lower_lr": lower_best,
            "delta": current_best - lower_best,
            "turned_down": bool(current and lower and current_best <= lower_best),
        }
    return {
        "learning_rate": float(learning_rate),
        "metrics": metrics,
        "bracketed": all(item["turned_down"] for item in metrics.values()),
        "rule": (
            "Stop after the seed-matched control when the best observed candidate R2 and mPQ+ at the "
            "new upper learning rate are both no better than their best values at lower learning rates."
        ),
    }


def run_fit(
    args: argparse.Namespace,
    family: str,
    loss_arguments: list[str],
    learning_rate: float,
) -> Path:
    lr_label = f"{learning_rate:.0e}".replace("e-0", "e-")
    outdir = args.out_root / family / f"lr_{lr_label}"
    if (outdir / "summary.json").exists():
        print(f"E44 LR-expansion fit complete, skipping: {outdir}", flush=True)
        return outdir
    archived = archive_incomplete_persistent_run(outdir)
    if archived is not None:
        print(f"archived incomplete persistent-worker run: {archived}", flush=True)
    command = [
        sys.executable, "scripts/train_hovernet_our_split.py",
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
        raise SystemExit(f"E44 LR-expansion fit failed with exit code {completed.returncode}")
    return outdir


def plot_expanded_lr_curves(
    candidate_rows: list[dict],
    control_rows: list[dict],
    output: Path,
) -> None:
    """Show metric trajectories for every candidate/control LR on one validation figure."""
    panels = (
        ("val_R2", "Count R²"),
        ("val_mPQ+", "Typed panoptic quality (mPQ+)"),
        ("val_mDQ+", "Typed detection quality (mDQ+)"),
        ("val_mSQ+", "Typed matched-mask quality (mSQ+)"),
    )
    rates = sorted({float(row["learning_rate"]) for row in candidate_rows})
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(rates)))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for axis, (metric, title) in zip(axes.flat, panels):
        for color, rate in zip(colors, rates):
            for rows, style, suffix in (
                (candidate_rows, "-", "combined loss"),
                (control_rows, "--", "ordinary loss"),
            ):
                selected = sorted(
                    (row for row in rows if np.isclose(float(row["learning_rate"]), rate)),
                    key=lambda row: int(row["epoch"]),
                )
                if not selected:
                    continue
                axis.plot(
                    [row["epoch"] for row in selected],
                    [row[metric] for row in selected],
                    marker="o", linestyle=style, color=color,
                    label=f"LR {rate:g} · {suffix}",
                )
        axis.set(title=title, xlabel="epoch", ylabel=metric)
        axis.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=8)
    fig.suptitle("E44 validation-only learning-rate expansion · candidate versus seed-matched control")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


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
    parser.add_argument("--lower-learning-rate", type=float, default=1e-4)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[6e-4, 1e-3])
    args = parser.parse_args()

    e44 = wait_for_json(args.wait_for, args.poll_seconds)
    control = json.loads(args.control_summary.read_text())
    validate_control_seed(control, args.seed)
    family, loss_arguments, recipe_reason = expansion_recipe(e44)
    original = e44["families"][family]

    candidate_runs = {float(rate): Path(path) for rate, path in original["runs"].items()}
    control_runs = {float(rate): Path(path) for rate, path in control["runs"].items()}
    if args.lower_learning_rate not in candidate_runs:
        candidate_runs[args.lower_learning_rate] = run_fit(
            args, family, loss_arguments, args.lower_learning_rate
        )
    adaptive_stop = None
    requested_learning_rates = sorted(set(args.learning_rates))
    for learning_rate in requested_learning_rates:
        candidate_runs[learning_rate] = run_fit(args, family, loss_arguments, learning_rate)
        control_runs[learning_rate] = run_fit(args, "matched_uniform_control", [], learning_rate)
        rows_so_far = [
            row
            for rate, run_dir in candidate_runs.items()
            for row in scored_rows(run_dir, family, rate)
        ]
        adaptive_stop = upper_lr_bracket(rows_so_far, learning_rate)
        if adaptive_stop["bracketed"]:
            print(
                f"E44 upper LR bracket established at {learning_rate:g}; "
                "skipping higher requested rates.",
                flush=True,
            )
            break

    candidate_rows = [
        row
        for learning_rate, run_dir in candidate_runs.items()
        for row in scored_rows(run_dir, family, learning_rate)
    ]
    control_rows = [
        row
        for learning_rate, run_dir in control_runs.items()
        for row in scored_rows(run_dir, "matched_uniform_control", learning_rate)
    ]
    selected_r2 = select(candidate_rows, "val_R2")
    selected_mpq = select(candidate_rows, "val_mPQ+")
    expanded_control = {
        **control,
        "runs": {str(rate): str(path) for rate, path in sorted(control_runs.items())},
        "selected_R2": select(control_rows, "val_R2"),
        "selected_mPQ+": select(control_rows, "val_mPQ+"),
    }
    exact_r2 = exact_checkpoint_comparison(selected_r2, expanded_control)
    exact_mpq = exact_checkpoint_comparison(selected_mpq, expanded_control)
    delta = independently_selected_delta(selected_r2, selected_mpq, expanded_control)
    audit = promotion_audit(exact_mpq, delta)
    all_rates = sorted(candidate_runs)
    boundaries = {
        "R2": selection_boundaries(selected_r2, all_rates, args.epochs),
        "mPQ+": selection_boundaries(selected_mpq, all_rates, args.epochs),
    }
    curve_path = args.out_root / "e44_lr_expansion_curves.png"
    plot_expanded_lr_curves(candidate_rows, control_rows, curve_path)
    mpq_specific_admission = bool(
        audit["typed_signal"]
        and audit["binary_geometry_safe"]
        and audit["major_sources_safe"]
        and audit["spatial_consistency_safe"]
        and not boundaries["mPQ+"]["requires_learning_rate_expansion"]
    )
    report = {
        "protocol": (
            "Validation-only upper-LR expansion for the E44 mPQ-specific recipe. Every new candidate LR has "
            "a seed-, split-, horizon-, and initialization-matched ordinary-loss fit. Locked test is refused."
        ),
        "evaluation_set": "711-patch source-group-disjoint development validation",
        "seed": args.seed,
        "selected_recipe_family": family,
        "selected_recipe_reason": recipe_reason,
        "loss_arguments": loss_arguments,
        "requested_upper_learning_rates": requested_learning_rates,
        "adaptive_lr_stop": adaptive_stop,
        "candidate_runs": {str(rate): str(path) for rate, path in sorted(candidate_runs.items())},
        "matched_control_runs": {str(rate): str(path) for rate, path in sorted(control_runs.items())},
        "selected_R2": selected_r2,
        "selected_mPQ+": selected_mpq,
        "matched_control_selected_R2": expanded_control["selected_R2"],
        "matched_control_selected_mPQ+": expanded_control["selected_mPQ+"],
        "independently_selected_delta": delta,
        "exact_matched_control": {"R2": exact_r2, "mPQ+": exact_mpq},
        "promotion_audit": audit,
        "selection_boundaries": boundaries,
        "training_curve_figure": str(curve_path),
        "mPQ_specific_admission": mpq_specific_admission,
        "mPQ_specific_interpretation": (
            "Count tails are reported but do not veto a segmentation model selected specifically for mPQ+. "
            "Typed gain, binary geometry, within-nucleus spatial consistency, major-source stability, and an "
            "interior LR do gate admission. If the matched historical checkpoint lacks the spatial diagnostic, "
            "admission remains held until deterministic backfill. "
            "The fixed 10-epoch endpoint is a pilot budget; mature schedule duration remains validation-selected."
        ),
        "test_evaluated": False,
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    output = args.out_root / "e44_lr_expansion_summary.json"
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
