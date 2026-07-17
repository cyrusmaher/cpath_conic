#!/usr/bin/env python3
"""Turn validation complementarity audits into explicit E46 admissions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


COUNT_TOLERANCES = {
    "MAE": 0.25,
    "absolute_mean_signed_error": 0.25,
    "absolute_error_gt_10_fraction": 0.005,
    "absolute_error_gt_20_fraction": 0.005,
}
ZERO_TRUTH_MIN_SUPPORT = 20
ZERO_TRUTH_TOLERANCES = {
    "over_10_fraction": 0.02,
    "over_20_fraction": 0.01,
}


def finite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def directional_tail_audit(candidate: dict, reference: dict) -> dict:
    candidate_bias = abs(float(candidate["mean_signed_error"]))
    reference_bias = abs(float(reference["mean_signed_error"]))
    deltas = {
        "MAE": float(candidate["MAE"] - reference["MAE"]),
        "absolute_mean_signed_error": float(candidate_bias - reference_bias),
        "absolute_error_gt_10_fraction": float(
            candidate["absolute_error_gt_10_fraction"]
            - reference["absolute_error_gt_10_fraction"]
        ),
        "absolute_error_gt_20_fraction": float(
            candidate["absolute_error_gt_20_fraction"]
            - reference["absolute_error_gt_20_fraction"]
        ),
    }
    passes = all(deltas[key] <= tolerance for key, tolerance in COUNT_TOLERANCES.items())
    return {
        "delta": deltas,
        "tolerances": COUNT_TOLERANCES,
        "passes": bool(passes),
    }


def zero_truth_tail_audit(
    candidate: dict,
    reference: dict,
    candidate_by_source: dict | None = None,
    reference_by_source: dict | None = None,
) -> dict:
    """Reject supported true-zero strata that acquire materially more large false counts."""
    candidate_zero = candidate.get("zero_truth_overcount") or {}
    reference_zero = reference.get("zero_truth_overcount") or {}
    if not candidate_zero or not reference_zero:
        return {
            "available": False,
            "passes": True,
            "reason": "True-zero tail diagnostics are absent from this legacy report.",
        }

    strata = []

    def add_scope(scope: str, left: dict, right: dict) -> None:
        if not left or not right:
            return
        pairs = [(scope, left.get("all_zero_truth_points"), right.get("all_zero_truth_points"))]
        for class_name in sorted(set((left.get("per_class") or {})) & set((right.get("per_class") or {}))):
            pairs.append((f"{scope}/class={class_name}", left["per_class"][class_name], right["per_class"][class_name]))
        for label, candidate_row, reference_row in pairs:
            if not candidate_row or not reference_row:
                continue
            support = min(int(candidate_row.get("support", 0)), int(reference_row.get("support", 0)))
            if support < ZERO_TRUTH_MIN_SUPPORT:
                continue
            deltas = {
                key: float(candidate_row[key] - reference_row[key])
                for key in ZERO_TRUTH_TOLERANCES
                if candidate_row.get(key) is not None and reference_row.get(key) is not None
            }
            if len(deltas) == len(ZERO_TRUTH_TOLERANCES):
                effective_tolerances = {
                    key: max(base_tolerance, 2.0 / support)
                    for key, base_tolerance in ZERO_TRUTH_TOLERANCES.items()
                }
                strata.append({
                    "stratum": label,
                    "support": support,
                    "delta": deltas,
                    "effective_tolerance": effective_tolerances,
                })

    add_scope("pooled", candidate_zero, reference_zero)
    for source in sorted(set(candidate_by_source or {}) & set(reference_by_source or {})):
        add_scope(
            f"source={source}",
            (candidate_by_source[source].get("zero_truth_overcount") or {}),
            (reference_by_source[source].get("zero_truth_overcount") or {}),
        )

    violations = [
        row for row in strata
        if any(
            row["delta"][key] > row["effective_tolerance"][key]
            for key in ZERO_TRUTH_TOLERANCES
        )
    ]
    worst = sorted(
        strata,
        key=lambda row: max(
            row["delta"][key] - row["effective_tolerance"][key]
            for key in ZERO_TRUTH_TOLERANCES
        ),
        reverse=True,
    )[:10]
    return {
        "available": True,
        "minimum_support": ZERO_TRUTH_MIN_SUPPORT,
        "base_tolerances": ZERO_TRUTH_TOLERANCES,
        "effective_tolerance_rule": "max(base tolerance, two patches / stratum support)",
        "eligible_strata": len(strata),
        "violating_strata": len(violations),
        "worst_strata": worst,
        "passes": not violations,
    }
def hyperparameter_boundary_audit(selection_constraint: dict | None) -> dict:
    if selection_constraint is None:
        return {
            "available": False,
            "passes": True,
            "reason": "No boundary constraint supplied; legacy diagnostic-only behavior.",
        }
    reasons = list(selection_constraint.get("unconfirmed_boundaries", []))
    return {
        "available": True,
        "passes": not reasons,
        "unconfirmed_boundaries": reasons,
        "selection": selection_constraint.get("selection", {}),
        "reason": (
            "Selected training hyperparameters are interior to the searched grid."
            if not reasons else
            "Hold composition admission until the selected boundary is expanded: " + ", ".join(reasons)
        ),
    }


def count_admission(
    report: dict,
    minimum_gain: float = 0.003,
    selection_constraint: dict | None = None,
) -> dict:
    first = report["first_model"]
    second = report["second_model"]
    endpoint_delta = float(second["R2"] - first["R2"])
    endpoint_tail = directional_tail_audit(second["count_error"], first["count_error"])
    endpoint_zero_tail = zero_truth_tail_audit(
        second, first, second.get("by_source"), first.get("by_source")
    )
    boundary = hyperparameter_boundary_audit(selection_constraint)
    endpoint_advances = (
        endpoint_delta > minimum_gain
        and endpoint_tail["passes"]
        and endpoint_zero_tail["passes"]
        and boundary["passes"]
    )

    best = max((first, second), key=lambda item: float(item["R2"]))
    held_out = report["leave_one_source_out"]["pooled_out_of_source"]
    blend_delta = float(held_out["R2"] - best["R2"])
    blend_tail = directional_tail_audit(held_out["count_error"], best["count_error"])
    blend_zero_tail = zero_truth_tail_audit(
        held_out,
        best,
        report["leave_one_source_out"].get("held_out_sources"),
        best.get("by_source"),
    )
    full_delta = float(report["stability"]["full_validation_per_class_blend_delta_R2"])
    full_minus_held_out = float(report["stability"]["full_minus_cross_source_R2"])
    blend_advances = (
        blend_delta > minimum_gain
        and full_delta > minimum_gain
        and full_minus_held_out <= 0.02
        and blend_tail["passes"]
        and blend_zero_tail["passes"]
        and boundary["passes"]
    )
    return {
        "candidate": second["name"],
        "control": first["name"],
        "evaluation_set": report["evaluation_set"],
        "training_hyperparameter_boundary_audit": boundary,
        "standalone_candidate": {
            "delta_R2_vs_control": endpoint_delta,
            "directional_tail_audit": endpoint_tail,
            "zero_truth_tail_audit": endpoint_zero_tail,
            "advances_to_mature_training_screen": bool(endpoint_advances),
        },
        "source_held_out_blend": {
            "delta_R2_vs_best_endpoint": blend_delta,
            "full_validation_delta_R2_vs_best_endpoint": full_delta,
            "full_minus_source_held_out_R2": full_minus_held_out,
            "selected_second_model_weights_by_held_source": report["leave_one_source_out"][
                "selected_second_model_weights_by_held_source"
            ],
            "directional_tail_audit": blend_tail,
            "zero_truth_tail_audit": blend_zero_tail,
            "advances_to_raw_map_or_count_composition": bool(blend_advances),
        },
        "passes_any_count_gate": bool(endpoint_advances or blend_advances),
    }


def type_admission(
    report: dict,
    minimum_gain: float = 0.003,
    selection_constraint: dict | None = None,
) -> dict:
    delta = report["selected_delta_vs_control"]
    rows = report["candidates"]
    baseline_row = next(row for row in rows if float(row["candidate_type_weight"]) == 0.0)
    baseline = baseline_row["overall"]
    selected_row = report["selected"]
    selected = selected_row["overall"]
    tail = directional_tail_audit(selected["count_error"], baseline["count_error"])
    zero_tail = zero_truth_tail_audit(
        selected,
        baseline,
        report["selected"].get("by_source"),
        baseline_row.get("by_source"),
    )
    metric_gate = (
        float(delta["mPQ+"]) > minimum_gain
        and float(delta["mDQ+"]) > minimum_gain
        and float(delta["mSQ+"]) > -0.005
    )
    stable = bool(report["weight_stable_across_source_exclusions"])
    source_delta = {
        source: {
            metric: float(
                selected_row["by_source"][source][metric]
                - baseline_row["by_source"][source][metric]
            )
            for metric in ("mPQ+", "mDQ+", "mSQ+")
        }
        for source in selected_row.get("by_source", {}).keys() & baseline_row.get("by_source", {}).keys()
    }
    major_sources_safe = all(
        source_delta.get(source, {}).get(metric, -np.inf) > -0.01
        for source in ("crag", "dpath", "glas")
        for metric in ("mPQ+", "mDQ+", "mSQ+")
    )
    boundary = hyperparameter_boundary_audit(selection_constraint)
    advances = metric_gate and stable and major_sources_safe and boundary["passes"]
    return {
        "candidate": report["candidate"],
        "control": report["control"],
        "evaluation_set": report["evaluation_set"],
        "training_hyperparameter_boundary_audit": boundary,
        "selected_candidate_type_weight": float(report["selected"]["candidate_type_weight"]),
        "metric_delta_vs_fixed_geometry_control": {
            key: float(delta[key]) for key in ("R2", "mPQ+", "mDQ+", "mSQ+")
        },
        "passes_metric_gate": bool(metric_gate),
        "source_excluded_selected_weights": report["selected_candidate_weight_excluding_each_source"],
        "weight_stable_across_source_exclusions": stable,
        "typed_delta_by_source_vs_fixed_geometry_control": source_delta,
        "major_sources_safe": bool(major_sources_safe),
        "directional_tail_audit": tail,
        "zero_truth_tail_audit": zero_tail,
        "count_tail_interpretation": (
            "Directional and true-zero count tails are reported for the independent R2/reliability track; "
            "they do not veto a fixed-geometry mPQ type-probability candidate."
        ),
        "advances_to_raw_map_type_composition": bool(advances),
    }


def validation_protocol(report: dict) -> bool:
    protocol = str(report.get("protocol", "")).lower()
    evaluation = str(report.get("evaluation_set", "")).lower()
    return "validation" in protocol and "test" not in evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--minimum-gain", type=float, default=0.003)
    parser.add_argument("--selection-constraints", type=Path)
    args = parser.parse_args()
    if args.minimum_gain < 0:
        parser.error("--minimum-gain must be non-negative")

    constraints = (
        json.loads(args.selection_constraints.read_text())
        if args.selection_constraints is not None else {}
    )
    count_reports = []
    for path in sorted(args.audit_root.glob("e46_*_count_complementarity.json")):
        payload = json.loads(path.read_text())
        if not validation_protocol(payload):
            raise RuntimeError(f"E46 selection refuses non-validation report: {path}")
        candidate = payload["second_model"]["name"]
        count_reports.append({
            "report": str(path),
            **count_admission(payload, args.minimum_gain, constraints.get(candidate)),
        })

    type_reports = []
    for path in sorted(args.audit_root.glob("e46_*_type_complementarity.json")):
        payload = json.loads(path.read_text())
        if not validation_protocol(payload):
            raise RuntimeError(f"E46 selection refuses non-validation report: {path}")
        candidate = payload["candidate"]
        type_reports.append({
            "report": str(path),
            **type_admission(payload, args.minimum_gain, constraints.get(candidate)),
        })

    report = {
        "status": "complete" if count_reports or type_reports else "no complementarity reports found",
        "protocol": (
            "Development-validation-only E46 candidate admission. Standalone R2 improvement and source-held-out "
            "blend complementarity are separate claims. Type blending keeps control geometry fixed and requires "
            "source-exclusion stability. Training recipes selected on an unexpanded LR or intervention-strength "
            "boundary are held even if their complementarity screen passes. Locked-test evidence is refused."
        ),
        "minimum_practical_gain": args.minimum_gain,
        "count_tail_tolerances": COUNT_TOLERANCES,
        "zero_truth_tail_tolerances": ZERO_TRUTH_TOLERANCES,
        "zero_truth_minimum_support": ZERO_TRUTH_MIN_SUPPORT,
        "count_candidates": count_reports,
        "type_candidates": type_reports,
        "admitted_standalone_count_candidates": [
            row["candidate"]
            for row in count_reports
            if row["standalone_candidate"]["advances_to_mature_training_screen"]
        ],
        "admitted_count_compositions": [
            row["candidate"]
            for row in count_reports
            if row["source_held_out_blend"]["advances_to_raw_map_or_count_composition"]
        ],
        "admitted_type_compositions": [
            row["candidate"]
            for row in type_reports
            if row["advances_to_raw_map_type_composition"]
        ],
        "test_evaluated": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({
        key: report[key]
        for key in (
            "status",
            "admitted_standalone_count_candidates",
            "admitted_count_compositions",
            "admitted_type_compositions",
        )
    }, indent=2))


if __name__ == "__main__":
    main()
