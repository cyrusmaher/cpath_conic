#!/usr/bin/env python3
"""Backfill pre-diagnostic HoVer checkpoints after all queued training is idle."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


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


def checkpoint_for(selection: dict, metric: str) -> Path:
    checkpoint = Path(selection["run_dir"]) / ("best_r2.pth" if metric == "R2" else "best_mpq.pth")
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def run(command: list[str]) -> None:
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def add_selected_checkpoint(
    requested: dict[Path, list[str]],
    selection: dict,
    metric: str,
    label: str,
) -> None:
    requested.setdefault(checkpoint_for(selection, metric), []).append(label)


def training_selection_constraint(selection: dict, boundary: dict | None = None) -> dict:
    """Hold E46 admission for an unexpanded training-hyperparameter boundary."""
    boundary = boundary or {}
    reasons = []
    learning_rate = float(selection["learning_rate"])
    if boundary.get("requires_learning_rate_expansion") or any(
        abs(learning_rate - edge) <= 1e-12 for edge in (3e-5, 3e-4)
    ):
        reasons.append("learning_rate")
    if boundary.get("requires_weight_expansion"):
        reasons.append("intervention_strength")
    return {
        "selection": {
            key: selection[key]
            for key in ("learning_rate", "epoch", "instance_type_loss_weight")
            if key in selection
        },
        "unconfirmed_boundaries": reasons,
    }


def single_dose_selection_constraint(
    selection: dict,
    boundary: dict | None = None,
) -> dict:
    """Hold an intervention whose strength was never bracketed."""
    constraint = training_selection_constraint(selection, boundary)
    constraint["unconfirmed_boundaries"] = sorted(set([
        *constraint["unconfirmed_boundaries"],
        "intervention_strength",
    ]))
    return constraint


def matched_resnet_control_label(candidate_label: str, metric: str) -> tuple[str, str]:
    """Route each intervention to its exact-seed ordinary ResNet control."""
    if metric not in {"R2", "mPQ+"}:
        raise ValueError(f"unsupported control metric: {metric}")
    if candidate_label.startswith("e37_"):
        return f"e37_no_sampling_seed206_{metric}", "seed-206 uniform ResNet-50"
    return f"e36_no_hed_seed205_{metric}", "seed-205 ResNet-50"


def is_complementarity_candidate_label(label: str, metric: str) -> bool:
    """Exclude ordinary controls while admitting declared intervention families."""
    return label.endswith(f"_{metric}") and label.startswith((
        "e37_class_", "e37_source_", "e41_", "e42_", "e43_", "e44_",
    ))


def selected_type_families(type_payload: dict) -> list[tuple[str, dict]]:
    """Return completed E44 families while honoring conditional skip records."""
    selected = []
    for family, payload in type_payload.get("families", {}).items():
        if payload.get("status", "").startswith("skipped"):
            continue
        missing = [key for key in ("selected_R2", "selected_mPQ+") if key not in payload]
        if missing:
            raise KeyError(f"completed E44 family {family} is missing {', '.join(missing)}")
        selected.append((family, payload))
    return selected


def e44_backfill_eligible(expanded_type: dict) -> bool:
    """Materialize E44 composition artifacts only after a metric-specific signal survives."""
    audit = expanded_type.get("promotion_audit", {})
    delta = expanded_type.get("independently_selected_delta", {})
    typed_candidate = bool(audit.get("typed_signal", False))
    count_candidate = bool(
        float(delta.get("R2", float("-inf"))) > 0.01
        and audit.get("count_tails_safe", False)
    )
    return typed_candidate or count_candidate


def training_prerequisites(args: argparse.Namespace) -> list[Path]:
    """Terminal summaries that must exist before GPU diagnostic inference."""
    return [args.wait_for, args.instance_type_summary, args.type_lr_expansion_summary]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for", type=Path, required=True)
    parser.add_argument("--intervention-summary", type=Path, required=True)
    parser.add_argument(
        "--backbone-summary",
        type=Path,
        default=Path("outputs/conic_backbone_pilots/e41_seresnext101_summary.json"),
    )
    parser.add_argument(
        "--instance-summary",
        type=Path,
        default=Path("outputs/conic_instance_loss_pilots/e42_instance_loss_summary.json"),
    )
    parser.add_argument(
        "--instance-type-summary",
        type=Path,
        default=Path("outputs/conic_instance_type_pilots/e43_instance_type_summary.json"),
    )
    parser.add_argument(
        "--type-summary",
        type=Path,
        default=Path("outputs/conic_type_loss_pilots/e44_type_focal_summary.json"),
    )
    parser.add_argument(
        "--type-lr-expansion-summary",
        type=Path,
        default=Path("outputs/conic_type_loss_lr_expansion/e44_lr_expansion_summary.json"),
    )
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--train-ids", type=Path, default=None)
    parser.add_argument("--val-ids", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    for prerequisite in training_prerequisites(args):
        wait_for_json(prerequisite, args.poll_seconds)
    # E43 may be launched by a serialized companion waiter when the already-
    # running master process predates its insertion into stage_commands().
    # Always wait for its terminal summary before any GPU backfill so training
    # and inference cannot race. A causal-gate skip still writes this summary.
    summary = wait_for_json(args.intervention_summary, args.poll_seconds)
    requested: dict[Path, list[str]] = {}
    selection_constraints: dict[str, dict] = {}
    for family in ("e36_empirical_hed", "e37_source_0.5", "e37_class_0.5"):
        for metric, key in (("R2", "selected_R2"), ("mPQ+", "selected_mPQ+")):
            selection = summary["families"][family][key]
            label = f"{family}_{metric}"
            add_selected_checkpoint(requested, selection, metric, label)
            # Each historical intervention was tested at one fixed dose.  A
            # complementary validation residual can motivate a dose bracket,
            # but it cannot make that unexpanded dose composition-ready.
            selection_constraints[label] = single_dose_selection_constraint(selection)
    for family, summary_name in (
        ("e36_no_hed_seed205", "e36_matched_control_summary.json"),
        ("e37_no_sampling_seed206", "e37_matched_control_summary.json"),
    ):
        control = wait_for_json(args.intervention_summary.parent / summary_name, args.poll_seconds)
        for metric, key in (("R2", "selected_R2"), ("mPQ+", "selected_mPQ+")):
            selection = control[key]
            add_selected_checkpoint(requested, selection, metric, f"{family}_{metric}")

    # Preserve per-patch predictions for every selected modern HoVer endpoint.
    # These validation-only artifacts are required to test error complementarity
    # before any heterogeneous raw-map ensemble or locked-test inference.
    modern_summaries: list[tuple[str, Path]] = [
        ("e41_seresnext101", args.backbone_summary),
        ("e42_instance_equalized", args.instance_summary),
    ]
    e43_included = True
    modern_summaries.append(("e43_instance_type", args.instance_type_summary))
    for family, path in modern_summaries:
        payload = wait_for_json(path, args.poll_seconds)
        if payload.get("status", "complete").startswith("skipped"):
            continue
        for metric, key in (("R2", "selected_R2"), ("mPQ+", "selected_mPQ+")):
            label = f"{family}_{metric}"
            selection = payload[key]
            add_selected_checkpoint(requested, selection, metric, label)
            boundary = payload.get("selection_boundaries", {}).get(metric, {})
            # E44 brackets LR but each family uses one fixed rho/gamma dose.
            # A positive family needs a local strength bracket before E46 can
            # admit either its standalone endpoint or a composition.
            selection_constraints[label] = single_dose_selection_constraint(selection, boundary)

    expanded_type = wait_for_json(args.type_lr_expansion_summary, args.poll_seconds)
    if e44_backfill_eligible(expanded_type):
        type_payload = wait_for_json(args.type_summary, args.poll_seconds)
        selected_type = selected_type_families(type_payload)
        selected_type_family_names = {family for family, _ in selected_type}
        for family in type_payload.get("families", {}):
            if family not in selected_type_family_names:
                print(f"type-loss family skipped by causal gate, no backfill needed: {family}", flush=True)
        for family, payload in selected_type:
            for metric, key in (("R2", "selected_R2"), ("mPQ+", "selected_mPQ+")):
                label = f"e44_{family}_{metric}"
                selection = payload[key]
                add_selected_checkpoint(requested, selection, metric, label)
                boundary = payload.get("selection_boundaries", {}).get(metric, {})
                selection_constraints[label] = training_selection_constraint(selection, boundary)
        for metric, key in (("R2", "selected_R2"), ("mPQ+", "selected_mPQ+")):
            label = f"e44_lr_expansion_{metric}"
            selection = expanded_type[key]
            add_selected_checkpoint(requested, selection, metric, label)
            boundary = expanded_type.get("selection_boundaries", {}).get(metric, {})
            selection_constraints[label] = training_selection_constraint(selection, boundary)
    else:
        print(
            "E44 rejected after expanded matched-control selection; skipping all E44 composition backfills",
            flush=True,
        )

    args.out_root.mkdir(parents=True, exist_ok=True)
    prediction_by_label: dict[str, Path] = {}
    for checkpoint, labels in requested.items():
        output = args.out_root / f"{'__'.join(labels).replace('+', 'plus')}.json"
        prediction_output = output.with_name(f"{output.stem}_predictions.npz")
        diagnostic_complete = output.exists() and prediction_output.exists()
        if diagnostic_complete:
            print(f"diagnostic backfill complete, skipping: {output}", flush=True)
        else:
            run([
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
        for label in labels:
            prediction_by_label[label] = prediction_output
        if any(label.startswith("e37_class_0.5_") for label in labels):
            correction_output = output.with_name(f"{output.stem}_sampler_prior.json")
            if correction_output.exists():
                print(f"sampler-prior correction complete, skipping: {correction_output}", flush=True)
                continue
            command = [
                sys.executable,
                "scripts/sweep_hovernet_sampler_prior.py",
                "--prepared", str(args.prepared),
                "--diagnostic-json", str(output),
                "--out-report", str(correction_output),
            ]
            if args.train_ids is not None:
                command.extend(["--train-ids", str(args.train_ids)])
            run(command)

    # Quantify count-residual complementarity before any candidate is admitted
    # to the metric-specific E46 blend.  The analyzer independently refuses
    # locked-test patch IDs and reselects class weights leave-one-source-out.
    for label, candidate_prediction in sorted(prediction_by_label.items()):
        if not is_complementarity_candidate_label(label, "R2"):
            continue
        control_label, control_name = matched_resnet_control_label(label, "R2")
        count_control = prediction_by_label.get(control_label)
        if count_control is None:
            raise RuntimeError(f"missing exact count control {control_label} for {label}")
        complementarity_output = args.audit_root / f"e46_{label}_vs_resnet_count_complementarity.json"
        if complementarity_output.exists():
            print(f"count-complementarity audit complete, skipping: {complementarity_output}", flush=True)
            continue
        run([
            sys.executable,
            "scripts/analyze_hovernet_count_complementarity.py",
            "--artifacts", str(count_control), str(candidate_prediction),
            "--names", control_name, label,
            "--prepared", str(args.prepared),
            "--out", str(complementarity_output),
        ])

    # A candidate can improve type discrimination while weakening NP/HV
    # geometry (E41 is the motivating case).  Match its type probabilities to
    # the selected ResNet instances and sweep only the type weight, keeping the
    # control instance map bitwise fixed so binary geometry cannot confound the
    # result.
    for label, candidate_prediction in sorted(prediction_by_label.items()):
        if not is_complementarity_candidate_label(label, "mPQ+"):
            continue
        control_label, control_name = matched_resnet_control_label(label, "mPQ+")
        type_control = prediction_by_label.get(control_label)
        if type_control is None:
            raise RuntimeError(f"missing exact type control {control_label} for {label}")
        safe_label = label.replace("+", "plus")
        complementarity_output = args.audit_root / f"e46_{safe_label}_vs_resnet_type_complementarity.json"
        if complementarity_output.exists():
            print(f"type-complementarity audit complete, skipping: {complementarity_output}", flush=True)
            continue
        run([
            sys.executable,
            "scripts/analyze_hovernet_type_complementarity.py",
            "--control-artifact", str(type_control),
            "--candidate-artifact", str(candidate_prediction),
            "--control-name", control_name,
            "--candidate-name", label,
            "--prepared", str(args.prepared),
            "--out", str(complementarity_output),
        ])

    # E42's causal claim is specifically that equal loss mass per nucleus helps
    # small-cell detection.  Test that claim on paired GT nuclei rather than
    # accepting a pooled DQ movement with an unknown size/FP mechanism.
    size_control = prediction_by_label.get("e36_no_hed_seed205_mPQ+")
    size_candidate = prediction_by_label.get("e42_instance_equalized_mPQ+")
    if size_control is not None and size_candidate is not None:
        size_output = args.audit_root / "e42_instance_equalized_detection_by_size.json"
        if size_output.exists():
            print(f"size-stratified detection audit complete, skipping: {size_output}", flush=True)
        else:
            run([
                sys.executable,
                "scripts/analyze_hovernet_detection_by_size.py",
                "--control-artifact", str(size_control),
                "--candidate-artifact", str(size_candidate),
                "--control-name", "seed-205 ResNet-50",
                "--candidate-name", "E42 instance-equalized loss",
                "--prepared", str(args.prepared),
                "--out", str(size_output),
            ])

    # Convert the pairwise diagnostics into an explicit pre-E45 membership
    # decision. E45 is serialized after this backfill and will be appended in
    # the final E46 admission report; this interim report prevents aggregate
    # gains from being described as blend complementarity without surviving
    # source-held-out and directional-tail gates.
    constraint_path = args.audit_root / "e46_training_selection_constraints.json"
    constraint_path.parent.mkdir(parents=True, exist_ok=True)
    constraint_path.write_text(json.dumps(selection_constraints, indent=2))
    run([
        sys.executable,
        "scripts/select_hovernet_e46_candidates.py",
        "--audit-root", str(args.audit_root),
        "--out", str(args.audit_root / "e46_candidate_admission_pre_e45.json"),
        "--selection-constraints", str(constraint_path),
    ])

    # Refresh causal audits now that historical selected rows carry source SSE
    # and decoded-instance confusion alongside the newly trained controls.
    for family, candidate_label, stem, control in (
        ("e36_empirical_hed", "empirical H/E", "e36_hed_vs_matched_control", "e36_no_hed_seed205"),
        ("e37_source_0.5", "source-balanced 0.5", "e37_source_0.5_vs_matched_control", "e37_no_sampling_seed206"),
        ("e37_class_0.5", "class-balanced 0.5", "e37_class_0.5_vs_matched_control", "e37_no_sampling_seed206"),
    ):
        run([
            sys.executable,
            "scripts/analyze_hovernet_pilot_pair.py",
            "--candidate-root", str(args.intervention_summary.parent / family),
            "--control-root", str(args.intervention_summary.parent / control),
            "--candidate-label", candidate_label,
            "--control-label", "matched uniform control",
            "--out-json", str(args.audit_root / f"{stem}.json"),
            "--out-plot", str(args.audit_root / f"{stem}.png"),
        ])
    marker = args.out_root / "backfill_complete.json"
    marker.write_text(json.dumps({
        "protocol": "development-validation-only deterministic backfill after serialized training",
        "checkpoints": [str(path) for path in requested],
        "artifacts": [str(path) for path in sorted(args.out_root.glob("*.json"))],
    }, indent=2))
    print(marker, flush=True)
    if e43_included:
        e43_marker = args.out_root / "backfill_e43_complete.json"
        e43_marker.write_text(json.dumps({
            "protocol": (
                "E43 selected-checkpoint validation diagnostics complete, or no E43 inference required "
                "because its predeclared causal gate skipped training; GPU backfill exited"
            ),
            "instance_type_summary": str(args.instance_type_summary),
            "artifacts": [str(path) for path in sorted(args.out_root.glob("*e43*.json"))],
        }, indent=2))
        print(e43_marker, flush=True)


if __name__ == "__main__":
    main()
