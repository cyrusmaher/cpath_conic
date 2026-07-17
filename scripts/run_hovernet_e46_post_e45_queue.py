#!/usr/bin/env python3
"""Append E45 validation diagnostics to E46 and finalize candidate admission."""
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


def run_checked(command: list[str]) -> None:
    print("running:", " ".join(command), flush=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise SystemExit(f"command failed with exit code {completed.returncode}: {' '.join(command)}")


def unique_artifact(root: Path, label: str) -> Path:
    matches = sorted(root.glob(f"*{label}*predictions.npz"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {label} prediction artifact in {root}, found {matches}")
    return matches[0]


def selection_constraint(selection: dict) -> dict:
    learning_rate = float(selection["learning_rate"])
    boundary = any(abs(learning_rate - edge) <= 1e-12 for edge in (3e-5, 3e-4))
    return {
        "selection": {key: selection[key] for key in ("learning_rate", "epoch")},
        "unconfirmed_boundaries": ["learning_rate"] if boundary else [],
    }


def type_candidate_artifact(artifact: dict, correction_audit: dict | None) -> tuple[Path, bool]:
    """Use prior-corrected labels only after their held-source promotion audit passes."""
    raw = Path(artifact["prediction_artifact"])
    if not correction_audit or not correction_audit.get("audit", {}).get(
        "advances_as_sampler_prior_correction", False
    ):
        return raw, False
    corrected = artifact.get("sampler_prior_corrected_prediction_artifact")
    if corrected is None:
        raise KeyError("promoted sampler-prior correction has no corrected prediction artifact")
    return Path(corrected), True


def run_selection(args: argparse.Namespace, constraints_path: Path) -> None:
    run_checked([
        sys.executable, "scripts/select_hovernet_e46_candidates.py",
        "--audit-root", str(args.audit_root), "--out", str(args.out),
        "--selection-constraints", str(constraints_path),
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for", type=Path, required=True)
    parser.add_argument("--e45-summary", type=Path, required=True)
    parser.add_argument("--backfill-root", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()

    wait_for_json(args.wait_for, args.poll_seconds)
    e45 = wait_for_json(args.e45_summary, args.poll_seconds)
    args.audit_root.mkdir(parents=True, exist_ok=True)
    constraints_path = args.audit_root / "e46_training_selection_constraints.json"
    constraints = json.loads(constraints_path.read_text()) if constraints_path.exists() else {}
    if e45.get("status", "").startswith("skipped"):
        constraints_path.write_text(json.dumps(constraints, indent=2))
        run_selection(args, constraints_path)
        return

    control_r2 = unique_artifact(args.backfill_root, "e37_no_sampling_seed206_R2")
    control_mpq = unique_artifact(args.backfill_root, "e37_no_sampling_seed206_mPQplus")
    correction_audits = {
        float(item["fraction"]): item for item in e45.get("sampler_prior_correction_audits", [])
    }
    for artifact in e45["selected_checkpoint_artifacts"]:
        fraction = float(artifact["fraction"])
        fraction_label = f"{fraction:g}".replace(".", "p")
        prediction = Path(artifact["prediction_artifact"])
        if not prediction.exists():
            raise FileNotFoundError(f"E45 prediction artifact is missing: {prediction}")
        if "R2" in artifact["metrics"]:
            candidate_name = f"E45 class sampling {fraction:g} R2 endpoint"
            selection = e45["selected_by_fraction"][str(fraction)]["selected_R2"]
            constraints[candidate_name] = selection_constraint(selection)
            output = args.audit_root / f"e46_e45_class_{fraction_label}_R2_vs_uniform_count_complementarity.json"
            if not output.exists():
                run_checked([
                    sys.executable, "scripts/analyze_hovernet_count_complementarity.py",
                    "--artifacts", str(control_r2), str(prediction),
                    "--names", "seed-206 uniform ResNet-50", candidate_name,
                    "--prepared", str(args.prepared), "--out", str(output),
                ])
        if "mPQ+" in artifact["metrics"]:
            type_prediction, uses_prior_correction = type_candidate_artifact(
                artifact, correction_audits.get(fraction)
            )
            if not type_prediction.exists():
                raise FileNotFoundError(f"E45 type prediction artifact is missing: {type_prediction}")
            correction_label = " + source-audited prior correction" if uses_prior_correction else ""
            candidate_name = f"E45 class sampling {fraction:g} mPQ+ endpoint{correction_label}"
            selection = e45["selected_by_fraction"][str(fraction)]["selected_mPQ+"]
            constraints[candidate_name] = selection_constraint(selection)
            output = args.audit_root / f"e46_e45_class_{fraction_label}_mPQplus_vs_uniform_type_complementarity.json"
            if not output.exists():
                run_checked([
                    sys.executable, "scripts/analyze_hovernet_type_complementarity.py",
                    "--control-artifact", str(control_mpq),
                    "--candidate-artifact", str(type_prediction),
                    "--control-name", "seed-206 uniform ResNet-50",
                    "--candidate-name", candidate_name,
                    "--prepared", str(args.prepared), "--out", str(output),
                ])

    constraints_path.write_text(json.dumps(constraints, indent=2))
    run_selection(args, constraints_path)
    marker = args.out.with_name("e46_post_e45_complete.json")
    marker.write_text(json.dumps({
        "protocol": "development-validation-only E46 admission after E45; no test inference",
        "e45_summary": str(args.e45_summary),
        "admission_report": str(args.out),
    }, indent=2))
    print(marker, flush=True)


if __name__ == "__main__":
    main()
