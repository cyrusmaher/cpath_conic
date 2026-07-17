import unittest
import json
import tempfile
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from PIL import Image

from cpath_conic.constants import CLASS_NAMES
from cpath_conic.data import central_crop_counts, official_hovernet_fold
from cpath_conic.directional import expand_hv_head, instance_directional_map
from cpath_conic.experiment_metrics import _feature_signature, _file_signature, evaluate_fixed_masks
from cpath_conic.hv import decode_hv, fast_binary_pq_stats
from cpath_conic.lora import LoRALinear
from cpath_conic.metrics import (
    binary_instance_segmentation_metrics,
    instance_type_confusion,
    multiclass_pq_plus,
    multiclass_r2,
    pq_stats,
)
from cpath_conic.queue_integrity import archive_incomplete_persistent_run
from cpath_conic.segmentation import instance_hv_map
from cpath_conic.sampling import (
    effective_sample_size,
    expected_unique_draws,
    minority_patch_weights,
    source_class_patch_weights,
    source_patch_weights,
)
from cpath_conic.tta import (
    invert_hv_horizontal_flip,
    invert_hv_rotation,
    invert_hv_vertical_flip,
    invert_spatial_rotation,
)
from cpath_conic.visuals import render_cutout_strip, render_panel
from scripts.train_lora_segmentation import (
    color_blur_augmentation,
    hed_stain_augmentation,
    select_train_validation_ids,
)
from scripts.audit_lora_detection_by_class import paired_bootstrap_bpq
from scripts.analyze_hovernet_count_complementarity import (
    blend_counts,
    leave_one_source_out,
    select_per_class_weights,
    zero_truth_overcount_summary,
)
from scripts.analyze_hovernet_type_complementarity import (
    blended_class_lookup_for_patch,
    blended_classes_for_patch,
    central_counts_from_lookup,
    decoded_class_lookup,
    fixed_geometry_lookup_pq_statistics,
    fixed_geometry_patch_pq_statistics,
    fixed_geometry_pq_cache,
    match_instances,
    patch_pq_statistics,
    pq_summary_from_statistics,
)
from scripts.sweep_robust_count_blend import robust_blend, true_zero_tail_gate
from scripts.analyze_hovernet_detection_by_size import grouped_detection, quantile_bin_labels
from scripts.select_raw_map_ensemble import evaluate_subset
from scripts.run_hovernet_control import (
    build_stain_views,
    instance_class_probabilities,
    normalize_model_weights,
    process_prediction as process_hovernet_prediction,
    resolve_branch_model_weights,
)
from scripts.run_hovernet_instance_type_pilot_queue import (
    exact_checkpoint_comparison,
    gate_evidence,
    promotion_audit as instance_type_promotion_audit,
    selection_boundaries as instance_type_selection_boundaries,
)
from scripts.run_hovernet_type_focal_pilot_queue import (
    independently_selected_delta as type_loss_delta,
    passes_segmentation_gate as type_loss_passes_gate,
    selection_boundaries as type_loss_selection_boundaries,
    validate_control_seed,
)
from scripts.run_hovernet_diagnostic_backfill_queue import (
    e44_backfill_eligible,
    is_complementarity_candidate_label,
    matched_resnet_control_label,
    selected_type_families,
    single_dose_selection_constraint,
    training_prerequisites,
    training_selection_constraint,
)
from scripts.run_hovernet_experiment_queue import stage_commands as hovernet_stage_commands
from scripts.run_hovernet_e46_post_e45_queue import type_candidate_artifact
from scripts.run_final_hovernet_stack_queue import prior_eligible, type_blend_eligible
from scripts.run_hovernet_e44_lr_expansion_queue import expansion_recipe, upper_lr_bracket
from scripts.select_hovernet_e46_candidates import hyperparameter_boundary_audit
from scripts.train_hovernet_our_split import (
    EmpiricalHEDTargetBank,
    capture_rng_state,
    complement_frequency_type_weights,
    count_error_stats,
    zero_truth_count_stats,
    hed_stain_augmentation_array,
    instance_equalized_pixel_weights,
    instance_pooled_type_loss,
    instance_type_diagnostic_sums,
    mse_loss,
    sampling_exposure_summary,
    require_resumable_worker_policy,
    restore_rng_state,
    sampling_mode,
    weighted_mse_loss,
    weighted_xentropy_loss,
    weighted_focal_xentropy_loss,
    xentropy_loss,
)
from scripts.plot_experiment_curve import selected_row
from scripts.plot_hovernet_family_progress import load_runs as load_hovernet_family_runs
from scripts.run_cellvit_conic import pool_instance_token
from scripts.compare_fixed_mask_methods import signed_error_summary
from scripts.predict_metric_aligned_classifier import counts_from_assignments
from scripts.sweep_class_confidence_rejection import apply_class_thresholds
from scripts.select_source_gated_ensemble import pooled_pq
from scripts.materialize_source_routed_predictions import routed_patch_ids
from scripts.bootstrap_source_route import pooled_mpq
from scripts.sweep_type_probability_ensemble import calibrated_logits, softmax
from scripts.sweep_count_ensemble import blended_counts
from scripts.select_prediction_source_route import pooled_pq_from_stats
from scripts.fit_count_stacker import design_matrix, fit_ridge, predict_ridge
from scripts.run_hovernet_intervention_pilot_queue import selected as select_intervention_lr
from scripts.analyze_hovernet_pilot_pair import (
    count_error_deltas,
    load_family as load_hovernet_analysis_family,
    mpq_source_counterfactuals,
    maybe_at_checkpoint,
    r2_sse_drivers,
    selection_boundary_flags,
    type_confusion_drivers,
)
from scripts.run_hovernet_instance_loss_pilot_queue import select as select_instance_loss
from scripts.run_hovernet_class_sampling_fraction_queue import (
    lower_fraction_promotion_audit,
    refinement_gate,
    sampler_prior_promotion_audit,
    selection_boundary_audit,
)
from scripts.render_review import build_subgroup_breakdown, e43_interim_note, require_complete_dashboard_summary
from scripts.sweep_hovernet_sampler_prior import (
    apply_log_prior_correction,
    relabel_predictions,
    sampled_exposure_prior,
    source_excluded_strength_audit,
)
from scripts.select_hovernet_e46_candidates import count_admission, type_admission


class DashboardModelTests(unittest.TestCase):
    def test_classify_outcome_maps_free_text_status_to_vocabulary(self):
        from cpath_conic.dashboard_model import classify_outcome

        cases = [
            ("baseline", {"kind": "baseline", "status": "complete"}),
            ("external", {"kind": "benchmark", "status": "complete — exact public-fold reproduction"}),
            ("promoted", {"kind": "combination", "status": "complete — promoted R2 recipe"}),
            ("rejected", {"kind": "isolated", "status": "complete — not promoted"}),
            ("rejected", {"kind": "failed", "status": "failed guard"}),
            ("rejected", {"kind": "isolated", "status": "complete"}),  # scored but not adopted
            ("cancelled", {"kind": "isolated", "status": "cancelled after fold 1"}),
            ("running", {"kind": "isolated", "status": "training · epoch 14/50"}),
            ("planned", {"kind": "future-best", "status": "not run yet"}),
        ]
        for expected, row in cases:
            self.assertEqual(classify_outcome(row), expected, msg=row["status"])

    def test_split_note_separates_recipe_from_findings(self):
        from cpath_conic.dashboard_model import split_note

        row = {"recipe": "Do the thing.", "findings": "It worked on validation."}
        self.assertEqual(split_note(row), ("Do the thing.", "It worked on validation."))
        # Legacy summary: only `notes`, recipe recovered by known prefix.
        legacy = {"id": "E01", "notes": "Do the thing. It worked."}
        self.assertEqual(split_note(legacy, {"E01": "Do the thing."}), ("Do the thing.", "It worked."))

    def test_build_trajectory_frontier_is_monotone_and_ends_on_best(self):
        from cpath_conic.dashboard_model import build_trajectory, normalize_rows

        rows = normalize_rows([
            {"id": "E00", "kind": "baseline", "status": "complete", "r2": 0.30, "mpq": 0.20},
            {"id": "E01", "kind": "isolated", "status": "complete", "r2": 0.55, "mpq": 0.19},
            {"id": "E02", "kind": "isolated", "status": "failed guard", "r2": 0.10, "mpq": 0.25},
            {"id": "E08", "kind": "benchmark", "status": "complete", "r2": 0.99, "mpq": 0.99},
            {"id": "E32", "kind": "isolated", "status": "complete", "r2": 0.80, "mpq": 0.44},
        ])
        trajectory = build_trajectory(rows, {"r2": 0.76, "mpq": 0.457})
        # External benchmark is excluded from the internal-test frontier.
        self.assertNotIn("E08", [point["id"] for point in trajectory["points"]])
        r2_values = [step["value"] for step in trajectory["series"]["r2"]["frontier"]]
        self.assertEqual(r2_values, sorted(r2_values))
        self.assertAlmostEqual(trajectory["series"]["r2"]["best"], 0.80)
        self.assertAlmostEqual(trajectory["series"]["mpq"]["best"], 0.44)


class PipelineTests(unittest.TestCase):
    def test_sampling_mode_separates_weighting_from_replacement(self):
        self.assertEqual(sampling_mode(0.0, 0.0), "without_replacement")
        self.assertEqual(sampling_mode(0.0, 0.0, True), "uniform_with_replacement")
        self.assertEqual(sampling_mode(0.0, 0.1), "weighted_with_replacement")
        with self.assertRaises(ValueError):
            sampling_mode(0.0, 0.1, True)

    def test_e43_dashboard_note_uses_only_e43_schema(self):
        matrix = json.loads(Path("experiments/conic_matrix.json").read_text())
        e42 = next(item for item in matrix["experiments"] if item["id"] == "E42")
        e43 = next(item for item in matrix["experiments"] if item["id"] == "E43")
        self.assertEqual(e43_interim_note(e42), "")
        note = e43_interim_note(e43)
        self.assertIn("completed low-LR endpoint", note)
        self.assertIn("Eosinophil R² is -3.6825", note)
        self.assertIn("At upper LR 0.0003, epoch 5", note)
        self.assertIn("correctly typed nuclei fall by 131", note)
        self.assertIn("1,167 extra spurious nuclei", note)
        self.assertIn("selects interior LR 0.0001", note)
        self.assertIn("446 more correctly typed", note)
        self.assertIn("123 fewer spurious nuclei", note)
        self.assertIn("711-patch source-group-disjoint development validation", note)

    def test_modern_hovernet_pilots_flag_lr_weight_and_horizon_boundaries(self):
        e43 = instance_type_selection_boundaries(
            {"learning_rate": 3e-4, "instance_type_loss_weight": 0.25, "epoch": 10},
            [3e-5, 1e-4, 3e-4], [0.05, 0.1, 0.25], 10,
        )
        self.assertTrue(e43["requires_learning_rate_expansion"])
        self.assertTrue(e43["requires_weight_expansion"])
        self.assertTrue(e43["requires_horizon_extension"])
        e44 = type_loss_selection_boundaries(
            {"learning_rate": 1e-4, "epoch": 5}, [3e-5, 1e-4, 3e-4], 10
        )
        self.assertFalse(e44["requires_boundary_confirmation"])

    def test_e43_exact_checkpoint_audit_uses_matched_lr_and_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate_dir = root / "candidate"
            control_dir = root / "control"
            candidate_dir.mkdir()
            control_dir.mkdir()

            def row(mpq, bpq, mae, source_mpq, spurious):
                return {
                    "epoch": 5, "decoder_learning_rate": 1e-4,
                    "val_R2": 0.5, "val_mPQ+": mpq, "val_mDQ+": mpq + 0.1,
                    "val_mSQ+": 0.8, "val_bPQ": bpq, "val_binary_DQ": bpq + 0.1,
                    "val_binary_SQ": 0.8, "val_AJI+": bpq, "val_boundary_F1": 0.85,
                    "val_gt_instance_type_nll": 0.9,
                    "val_gt_instance_type_target_probability": 0.7,
                    "val_gt_instance_type_entropy": 0.25,
                    "val_gt_instance_pixel_type_accuracy": 0.75,
                    "val_count_error": {
                        "MAE": mae, "mean_signed_error": 1.0,
                        "absolute_error_gt_10_fraction": 0.1,
                        "absolute_error_gt_20_fraction": 0.05,
                    },
                    "val_per_source": {
                        "crag": {"R2": 0.4, "mPQ+": source_mpq, "mDQ+": 0.5, "mSQ+": 0.8}
                    },
                    "val_instance_type_confusion": {
                        "geometry_matched": 100, "correctly_typed": 80,
                        "missed_truth": 20, "spurious_prediction": spurious,
                        "matched_type_accuracy": 0.8,
                    },
                }

            (candidate_dir / "training_curve.json").write_text(__import__("json").dumps([row(0.41, 0.55, 4.9, 0.40, 21)]))
            (control_dir / "training_curve.json").write_text(__import__("json").dumps([row(0.40, 0.56, 5.0, 0.39, 20)]))
            comparison = exact_checkpoint_comparison(
                {"run_dir": str(candidate_dir), "learning_rate": 1e-4, "epoch": 5},
                {"runs": {"0.0001": str(control_dir)}},
            )
            self.assertAlmostEqual(comparison["delta"]["val_mPQ+"], 0.01)
            self.assertAlmostEqual(comparison["count_error_delta"]["MAE"], -0.1)
            self.assertEqual(comparison["confusion_delta"]["spurious_prediction"], 1)

    def test_e43_geometry_failure_routes_typed_signal_to_fixed_geometry_screen(self):
        exact = {
            "delta": {
                "val_mPQ+": 0.01, "val_mDQ+": 0.012, "val_mSQ+": 0.0,
                "val_bPQ": -0.01, "val_binary_DQ": -0.012, "val_binary_SQ": 0.0,
                "val_gt_instance_type_spatial_js_disagreement": -0.001,
            },
            "count_error_delta": {
                "MAE": 0.1, "absolute_error_gt_10_fraction": 0.0,
                "absolute_error_gt_20_fraction": 0.0,
            },
            "source_delta": {
                source: {"mPQ+": 0.002, "mDQ+": 0.002, "mSQ+": 0.002}
                for source in ("crag", "dpath", "glas")
            },
        }
        audit = instance_type_promotion_audit(
            exact, {"mPQ+": 0.01, "mDQ+": 0.012, "mSQ+": 0.0}
        )
        self.assertTrue(audit["typed_signal"])
        self.assertFalse(audit["provisional_standalone_passes"])
        self.assertTrue(audit["admit_to_fixed_geometry_type_screen"])

    def test_e43_spatial_disagreement_regression_blocks_both_promotion_routes(self):
        exact = {
            "delta": {
                "val_mPQ+": 0.01, "val_mDQ+": 0.012, "val_mSQ+": 0.0,
                "val_bPQ": 0.0, "val_binary_DQ": 0.0, "val_binary_SQ": 0.0,
                "val_gt_instance_type_spatial_js_disagreement": 0.006,
            },
            "count_error_delta": {
                "MAE": 0.0, "absolute_error_gt_10_fraction": 0.0,
                "absolute_error_gt_20_fraction": 0.0,
            },
            "source_delta": {
                source: {"mPQ+": 0.0, "mDQ+": 0.0, "mSQ+": 0.0}
                for source in ("crag", "dpath", "glas")
            },
        }
        audit = instance_type_promotion_audit(
            exact, {"mPQ+": 0.01, "mDQ+": 0.012, "mSQ+": 0.0}
        )
        self.assertTrue(audit["typed_signal"])
        self.assertFalse(audit["spatial_consistency_safe"])
        self.assertFalse(audit["provisional_standalone_passes"])
        self.assertFalse(audit["admit_to_fixed_geometry_type_screen"])

    def test_e43_count_tails_are_reported_without_vetoing_mpq_recipe(self):
        exact = {
            "delta": {
                "val_mPQ+": 0.01, "val_mDQ+": 0.012, "val_mSQ+": 0.0,
                "val_bPQ": 0.0, "val_binary_DQ": 0.0, "val_binary_SQ": 0.0,
                "val_gt_instance_type_spatial_js_disagreement": 0.0,
            },
            "count_error_delta": {
                "MAE": 2.0, "absolute_error_gt_10_fraction": 0.05,
                "absolute_error_gt_20_fraction": 0.02,
            },
            "source_delta": {
                source: {"mPQ+": 0.0, "mDQ+": 0.0, "mSQ+": 0.0}
                for source in ("crag", "dpath", "glas")
            },
        }
        audit = instance_type_promotion_audit(
            exact, {"mPQ+": 0.01, "mDQ+": 0.012, "mSQ+": 0.0}
        )

        self.assertFalse(audit["count_tails_safe"])
        self.assertTrue(audit["provisional_standalone_passes"])
        self.assertTrue(audit["admit_to_fixed_geometry_type_screen"])

    def test_e43_major_source_sq_regression_blocks_mpq_promotion(self):
        exact = {
            "delta": {
                "val_mPQ+": 0.01, "val_mDQ+": 0.012, "val_mSQ+": 0.0,
                "val_bPQ": 0.0, "val_binary_DQ": 0.0, "val_binary_SQ": 0.0,
                "val_gt_instance_type_spatial_js_disagreement": 0.0,
            },
            "count_error_delta": {
                "MAE": 0.0, "absolute_error_gt_10_fraction": 0.0,
                "absolute_error_gt_20_fraction": 0.0,
            },
            "source_delta": {
                source: {"mPQ+": 0.0, "mDQ+": 0.0, "mSQ+": -0.011 if source == "glas" else 0.0}
                for source in ("crag", "dpath", "glas")
            },
        }
        audit = instance_type_promotion_audit(
            exact, {"mPQ+": 0.01, "mDQ+": 0.012, "mSQ+": 0.0}
        )

        self.assertFalse(audit["major_sources_safe"])
        self.assertFalse(audit["provisional_standalone_passes"])
        self.assertFalse(audit["admit_to_fixed_geometry_type_screen"])

    def test_pilot_pair_partial_grid_marks_absent_control_checkpoint(self):
        runs = {
            1e-4: {
                "path": "candidate",
                "rows": [{"epoch": 5, "val_R2": 0.5, "val_mPQ+": 0.4}],
            }
        }
        row, reason = maybe_at_checkpoint(runs, {"learning_rate": 3e-4, "epoch": 5})
        self.assertIsNone(row)
        self.assertIn("absent", reason)

    def test_e46_count_admission_separates_endpoint_and_source_held_out_blend(self):
        def error(bias, mae, gt10, gt20):
            return {
                "mean_signed_error": bias, "MAE": mae,
                "absolute_error_gt_10_fraction": gt10,
                "absolute_error_gt_20_fraction": gt20,
            }

        report = {
            "first_model": {"name": "control", "R2": 0.60, "count_error": error(2.0, 5.0, 0.15, 0.08)},
            "second_model": {"name": "candidate", "R2": 0.64, "count_error": error(1.5, 4.8, 0.14, 0.07)},
            "leave_one_source_out": {
                "pooled_out_of_source": {"R2": 0.65, "count_error": error(1.4, 4.7, 0.14, 0.07)},
                "selected_second_model_weights_by_held_source": {"a": {"epithelial": 0.5}},
            },
            "stability": {
                "full_validation_per_class_blend_delta_R2": 0.012,
                "full_minus_cross_source_R2": 0.002,
            },
            "evaluation_set": {"patches": 100},
        }
        audit = count_admission(report)
        self.assertTrue(audit["standalone_candidate"]["advances_to_mature_training_screen"])
        self.assertTrue(audit["source_held_out_blend"]["advances_to_raw_map_or_count_composition"])
        boundary = {"selection": {"learning_rate": 3e-4}, "unconfirmed_boundaries": ["learning_rate"]}
        held = count_admission(report, selection_constraint=boundary)
        self.assertFalse(held["standalone_candidate"]["advances_to_mature_training_screen"])
        self.assertFalse(held["source_held_out_blend"]["advances_to_raw_map_or_count_composition"])
        report["leave_one_source_out"]["pooled_out_of_source"]["count_error"] = error(4.0, 5.5, 0.17, 0.10)
        audit = count_admission(report)
        self.assertFalse(audit["source_held_out_blend"]["advances_to_raw_map_or_count_composition"])

    def test_e46_type_admission_requires_source_stability_and_sq_guardrail(self):
        baseline_error = {
            "mean_signed_error": 2.0, "MAE": 5.0,
            "absolute_error_gt_10_fraction": 0.15,
            "absolute_error_gt_20_fraction": 0.08,
        }
        selected_error = {
            "mean_signed_error": 1.8, "MAE": 4.9,
            "absolute_error_gt_10_fraction": 0.14,
            "absolute_error_gt_20_fraction": 0.08,
        }
        def typed(mpq=0.40, dq=0.50, sq=0.80, count_error=baseline_error):
            return {
                "overall": {
                    "mPQ+": mpq, "mDQ+": dq, "mSQ+": sq, "count_error": count_error,
                },
                "by_source": {
                    source: {"mPQ+": mpq, "mDQ+": dq, "mSQ+": sq}
                    for source in ("crag", "dpath", "glas")
                },
            }

        report = {
            "candidate": "candidate", "control": "control", "evaluation_set": "validation",
            "selected_delta_vs_control": {"R2": 0.0, "mPQ+": 0.01, "mDQ+": 0.012, "mSQ+": -0.002},
            "selected": {"candidate_type_weight": 0.5, **typed(0.41, 0.512, 0.798, selected_error)},
            "candidates": [{"candidate_type_weight": 0.0, **typed()}],
            "selected_candidate_weight_excluding_each_source": {"a": 0.5, "b": 0.5},
            "weight_stable_across_source_exclusions": True,
        }
        self.assertTrue(type_admission(report)["advances_to_raw_map_type_composition"])
        report["selected"]["overall"]["count_error"] = {
            "mean_signed_error": 4.0, "MAE": 6.0,
            "absolute_error_gt_10_fraction": 0.18,
            "absolute_error_gt_20_fraction": 0.10,
        }
        tail_audit = type_admission(report)
        self.assertFalse(tail_audit["directional_tail_audit"]["passes"])
        self.assertTrue(tail_audit["advances_to_raw_map_type_composition"])
        report["selected"]["by_source"]["dpath"]["mSQ+"] = 0.789
        source_audit = type_admission(report)
        self.assertFalse(source_audit["major_sources_safe"])
        self.assertFalse(source_audit["advances_to_raw_map_type_composition"])
        report["selected"]["by_source"]["dpath"]["mSQ+"] = 0.798
        boundary = {"selection": {"learning_rate": 3e-5}, "unconfirmed_boundaries": ["learning_rate"]}
        self.assertFalse(
            type_admission(report, selection_constraint=boundary)["advances_to_raw_map_type_composition"]
        )
        report["weight_stable_across_source_exclusions"] = False
        self.assertFalse(type_admission(report)["advances_to_raw_map_type_composition"])
    def test_class_sampling_refinement_gate_uses_same_lr_and_epoch_control(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate"
            control = root / "control"
            candidate.mkdir()
            control.mkdir()
            candidate_row = {
                "epoch": 10, "val_R2": 0.55, "val_mPQ+": 0.40,
                "val_mDQ+": 0.50, "val_mSQ+": 0.795,
            }
            control_row = {
                "epoch": 10, "val_R2": 0.62, "val_mPQ+": 0.35,
                "val_mDQ+": 0.44, "val_mSQ+": 0.798,
            }
            (candidate / "training_curve.json").write_text(__import__("json").dumps([candidate_row]))
            (control / "training_curve.json").write_text(__import__("json").dumps([control_row]))
            intervention = {"families": {"e37_class_0.5": {"selected_mPQ+": {
                "learning_rate": 1e-4, "epoch": 10, "run_dir": str(candidate),
            }}}}
            control_summary = {"runs": {"0.0001": str(control)}}
            gate = refinement_gate(intervention, control_summary)
            self.assertTrue(gate["passes"])
            self.assertAlmostEqual(gate["delta"]["mPQ+"], 0.05)
            self.assertLess(gate["delta"]["R2"], 0)

    def test_lower_class_sampling_promotion_requires_metric_and_directional_tail_gates(self):
        uniform = {
            "val_mPQ+": 0.35, "val_mDQ+": 0.44, "val_mSQ+": 0.80,
        }
        class_half = {
            "val_mPQ+": 0.40, "val_mDQ+": 0.50, "val_MAE": 6.0,
            "val_mean_signed_error": 3.5,
            "val_absolute_error_gt_10_fraction": 0.16,
            "val_absolute_error_gt_20_fraction": 0.08,
            "val_under_error_lt_minus_10_fraction": 0.03,
            "val_under_error_lt_minus_20_fraction": 0.01,
            "val_over_error_gt_10_fraction": 0.13,
            "val_over_error_gt_20_fraction": 0.07,
        }
        candidate = {
            "val_mPQ+": 0.397, "val_mDQ+": 0.496, "val_mSQ+": 0.798,
            "val_MAE": 5.9, "val_mean_signed_error": 2.5,
            "val_absolute_error_gt_10_fraction": 0.15,
            "val_absolute_error_gt_20_fraction": 0.075,
            "val_under_error_lt_minus_10_fraction": 0.04,
            "val_under_error_lt_minus_20_fraction": 0.015,
            "val_over_error_gt_10_fraction": 0.11,
            "val_over_error_gt_20_fraction": 0.06,
        }
        audit = lower_fraction_promotion_audit(candidate, uniform, class_half)
        self.assertTrue(audit["passes_uniform_metric_gate"])
        self.assertTrue(audit["passes_tail_gate_vs_class_0.5"])
        self.assertTrue(audit["advances_as_mpq_candidate"])
        self.assertTrue(audit["promotable_over_class_0.5_for_mpq"])
        self.assertTrue(audit["promotable_over_class_0.5"])
        candidate["val_mean_signed_error"] = 4.0
        failed_tail = lower_fraction_promotion_audit(candidate, uniform, class_half)
        self.assertFalse(failed_tail["passes_tail_gate_vs_class_0.5"])
        self.assertFalse(failed_tail["advances_to_tail_repair_schedule_screen"])
        self.assertTrue(failed_tail["advances_as_mpq_candidate"])
        self.assertTrue(failed_tail["promotable_over_class_0.5_for_mpq"])
        replacement_control = {
            "val_mPQ+": 0.396, "val_mDQ+": 0.495, "val_mSQ+": 0.799,
        }
        confounded = lower_fraction_promotion_audit(
            candidate, uniform, class_half, uniform_replacement=replacement_control
        )
        self.assertTrue(confounded["passes_practical_uniform_metric_gate"])
        self.assertFalse(confounded["passes_uniform_replacement_mechanism_gate"])
        self.assertFalse(confounded["advances_as_mpq_candidate"])
        source_unsafe = lower_fraction_promotion_audit(
            candidate, uniform, class_half, major_sources_safe=False
        )
        self.assertFalse(source_unsafe["major_sources_safe_vs_both_controls"])
        self.assertFalse(source_unsafe["passes_uniform_metric_gate"])
        self.assertFalse(source_unsafe["advances_as_mpq_candidate"])
        under_tuned_replacement = {
            "val_mPQ+": 0.38, "val_mDQ+": 0.48, "val_mSQ+": 0.80,
        }
        replacement_boundary = {
            "requires_boundary_confirmation": True,
        }
        held_for_control_expansion = lower_fraction_promotion_audit(
            candidate,
            uniform,
            class_half,
            uniform_replacement=under_tuned_replacement,
            uniform_replacement_boundary=replacement_boundary,
        )
        self.assertTrue(held_for_control_expansion["passes_uniform_replacement_mechanism_gate"])
        self.assertFalse(held_for_control_expansion["advances_as_mpq_candidate"])

    def test_lower_class_sampling_promotion_holds_boundary_winners(self):
        selection = {"learning_rate": 3e-4, "epoch": 10}
        boundary = selection_boundary_audit(selection, [3e-5, 1e-4, 3e-4], 10)
        self.assertTrue(boundary["requires_learning_rate_expansion"])
        self.assertTrue(boundary["requires_horizon_extension"])
        uniform = {"val_mPQ+": 0.35, "val_mDQ+": 0.44, "val_mSQ+": 0.80}
        half = {
            "val_mPQ+": 0.40, "val_mDQ+": 0.50, "val_MAE": 6.0,
            "val_mean_signed_error": 3.5,
            "val_absolute_error_gt_10_fraction": 0.16, "val_absolute_error_gt_20_fraction": 0.08,
            "val_under_error_lt_minus_10_fraction": 0.03, "val_under_error_lt_minus_20_fraction": 0.01,
            "val_over_error_gt_10_fraction": 0.13, "val_over_error_gt_20_fraction": 0.07,
        }
        candidate = {
            **half, "val_mPQ+": 0.399, "val_mDQ+": 0.499, "val_mSQ+": 0.799,
            "val_MAE": 5.9, "val_mean_signed_error": 2.5,
            "val_absolute_error_gt_10_fraction": 0.15,
        }
        audit = lower_fraction_promotion_audit(candidate, uniform, half, boundary)
        self.assertTrue(audit["passes_metric_and_tail_gates"])
        self.assertFalse(audit["advances_as_mpq_candidate"])
        self.assertFalse(audit["promotable_over_class_0.5_for_mpq"])
        self.assertFalse(audit["advances_to_schedule_screen"])
        self.assertFalse(audit["promotable_over_class_0.5"])

    def test_sampler_prior_correction_normalizes_and_moves_toward_target(self):
        probabilities = np.asarray([[0.45, 0.45, 0.025, 0.025, 0.025, 0.025]], dtype=np.float32)
        sampled = np.asarray([0.4, 0.4, 0.05, 0.05, 0.05, 0.05])
        target = np.asarray([0.1, 0.7, 0.05, 0.05, 0.05, 0.05])
        unchanged = apply_log_prior_correction(probabilities, sampled, target, 0.0)
        corrected = apply_log_prior_correction(probabilities, sampled, target, 1.0)
        np.testing.assert_allclose(unchanged, probabilities, atol=1e-7)
        np.testing.assert_allclose(corrected.sum(axis=1), 1.0, atol=1e-7)
        self.assertGreater(corrected[0, 1], corrected[0, 0])

    def test_sampler_prior_promotion_requires_source_held_out_strength_stability(self):
        def metric(mpq, dq, sq, mae=5.0, gt10=0.1, gt20=0.05):
            return {
                "R2": 0.5, "mPQ+": mpq, "mDQ+": dq, "mSQ+": sq,
                "count_error": {
                    "MAE": mae, "mean_signed_error": 1.0,
                    "absolute_error_gt_10_fraction": gt10,
                    "absolute_error_gt_20_fraction": gt20,
                },
                "per_source": {
                    source: {"mPQ+": mpq, "mDQ+": dq, "mSQ+": sq}
                    for source in ("crag", "dpath", "glas")
                },
            }

        delta = {"R2": 0.0, "mPQ+": 0.01, "mDQ+": 0.012, "mSQ+": 0.0}
        report = {
            "pooled_instance_strength_0": metric(0.40, 0.50, 0.80),
            "selected": {"mPQ+": {
                "strength": 0.5,
                "delta_vs_raw": delta,
                "delta_vs_pooled_strength_0": delta,
            }},
            "leave_one_source_out": {"mPQ+": {
                "delta_vs_raw": delta,
                "delta_vs_pooled_strength_0": delta,
                "selected_strength_excluding_each_source": {
                    "crag": 0.5, "dpath": 0.5, "glas": 0.75,
                },
                "stable_within_one_grid_step": True,
                "pooled_out_of_source": metric(0.41, 0.512, 0.80),
            }},
        }
        self.assertTrue(sampler_prior_promotion_audit(report)["advances_as_sampler_prior_correction"])
        report["leave_one_source_out"]["mPQ+"]["pooled_out_of_source"] = metric(
            0.41, 0.512, 0.80, mae=5.5, gt10=0.12, gt20=0.06
        )
        tail_audit = sampler_prior_promotion_audit(report)
        self.assertFalse(tail_audit["count_tails_safe"])
        self.assertTrue(tail_audit["advances_as_sampler_prior_correction"])
        report["leave_one_source_out"]["mPQ+"]["pooled_out_of_source"]["per_source"]["dpath"]["mSQ+"] = 0.789
        source_audit = sampler_prior_promotion_audit(report)
        self.assertFalse(source_audit["major_sources_safe"])
        self.assertFalse(source_audit["advances_as_sampler_prior_correction"])
        report["leave_one_source_out"]["mPQ+"]["pooled_out_of_source"] = metric(0.41, 0.512, 0.80)
        report["leave_one_source_out"]["mPQ+"]["stable_within_one_grid_step"] = False
        self.assertFalse(sampler_prior_promotion_audit(report)["advances_as_sampler_prior_correction"])

    def test_sampler_prior_source_exclusion_selects_unbiased_strength(self):
        patches = 10
        truth = np.zeros((patches, 256, 256, 2), dtype=np.int32)
        true_counts = np.zeros((patches, len(CLASS_NAMES)), dtype=np.int32)
        for patch in range(patches):
            instance_id = 1
            for class_index in range(len(CLASS_NAMES)):
                count = 1 + ((patch + class_index) % 2)
                true_counts[patch, class_index] = count
                for offset in range(count):
                    row = 32 + class_index * 24
                    column = 32 + offset * 8
                    truth[patch, row, column, 0] = instance_id
                    truth[patch, row, column, 1] = class_index + 1
                    instance_id += 1
        biased = truth.copy()
        biased[..., 1][truth[..., 1] == 1] = 2
        sources = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"])
        audit = source_excluded_strength_audit(
            truth,
            {0.0: truth.copy(), 1.0: biased},
            true_counts,
            sources,
        )
        self.assertEqual(set(audit["mPQ+"]["selected_strength_excluding_each_source"].values()), {0.0})
        self.assertTrue(audit["mPQ+"]["stable_within_one_grid_step"])
        self.assertAlmostEqual(audit["mPQ+"]["pooled_out_of_source"]["mPQ+"], 1.0, places=6)

    def test_sampler_exposure_prior_uses_all_epochs_through_checkpoint(self):
        first = {name: float(index + 1) for index, name in enumerate(CLASS_NAMES)}
        second = {name: float(6 - index) for index, name in enumerate(CLASS_NAMES)}
        rows = [
            {"epoch": 1, "train_sampling_actual": {"draws": 2, "mean_nuclei_per_draw": first}},
            {"epoch": 2, "train_sampling_actual": {"draws": 1, "mean_nuclei_per_draw": second}},
            {"epoch": 3, "train_sampling_actual": {"draws": 99, "mean_nuclei_per_draw": first}},
        ]
        prior, used = sampled_exposure_prior(rows, 2)
        expected = 2 * np.arange(1, 7) + np.arange(6, 0, -1)
        np.testing.assert_allclose(prior, expected / expected.sum())
        self.assertEqual(used, 2)

    def test_sampler_prior_relabel_preserves_geometry_and_maps_exact_instances(self):
        predictions = np.zeros((1, 3, 4, 2), dtype=np.int32)
        predictions[0, 0:2, 0:2, 0] = 7
        predictions[0, 0:2, 0:2, 1] = 2
        predictions[0, 1:3, 2:4, 0] = 11
        predictions[0, 1:3, 2:4, 1] = 4
        relabeled = relabel_predictions(
            predictions,
            np.asarray([42]),
            np.asarray([42, 42]),
            np.asarray([7, 11]),
            np.asarray([6, 1]),
        )
        np.testing.assert_array_equal(relabeled[..., 0], predictions[..., 0])
        self.assertTrue(np.all(relabeled[0, ..., 1][predictions[0, ..., 0] == 7] == 6))
        self.assertTrue(np.all(relabeled[0, ..., 1][predictions[0, ..., 0] == 11] == 1))
        np.testing.assert_array_equal(predictions[0, ..., 1][predictions[0, ..., 0] == 7], 2)

    def test_rng_checkpoint_restores_next_sampler_draw_without_replay(self):
        import random

        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)
        generators = {
            "sampler": torch.Generator().manual_seed(206),
            "train_workers": torch.Generator().manual_seed(1_000_209),
            "val_workers": torch.Generator().manual_seed(2_000_209),
        }
        weights = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.double)
        first_worker_seed = torch.empty((), dtype=torch.int64).random_(generator=generators["train_workers"])
        first = torch.multinomial(weights, 20, replacement=True, generator=generators["sampler"])
        state = capture_rng_state(generators)
        expected_worker_seed = torch.empty((), dtype=torch.int64).random_(generator=generators["train_workers"])
        expected_val_seed = torch.empty((), dtype=torch.int64).random_(generator=generators["val_workers"])
        expected_draw = torch.multinomial(weights, 20, replacement=True, generator=generators["sampler"])
        expected_aux = (random.random(), float(np.random.random()), float(torch.rand(())))

        resumed_generators = {
            "sampler": torch.Generator(),
            "train_workers": torch.Generator(),
            "val_workers": torch.Generator(),
        }
        restore_rng_state(state, resumed_generators)
        actual_worker_seed = torch.empty((), dtype=torch.int64).random_(generator=resumed_generators["train_workers"])
        actual_val_seed = torch.empty((), dtype=torch.int64).random_(generator=resumed_generators["val_workers"])
        actual_draw = torch.multinomial(weights, 20, replacement=True, generator=resumed_generators["sampler"])
        actual_aux = (random.random(), float(np.random.random()), float(torch.rand(())))
        self.assertNotEqual(first_worker_seed, expected_worker_seed)
        self.assertFalse(torch.equal(first, expected_draw))
        self.assertEqual(actual_worker_seed, expected_worker_seed)
        self.assertEqual(actual_val_seed, expected_val_seed)
        self.assertTrue(torch.equal(actual_draw, expected_draw))
        self.assertEqual(actual_aux, expected_aux)

    def test_rng_restore_rejects_legacy_checkpoint_without_sampler_state(self):
        with self.assertRaisesRegex(RuntimeError, "restart this candidate from epoch 0"):
            restore_rng_state({}, {"sampler": torch.Generator()})

    def test_persistent_worker_policy_forces_clean_restart(self):
        with self.assertRaisesRegex(RuntimeError, "persistent-worker augmentation RNG"):
            require_resumable_worker_policy("persistent")
        self.assertIsNone(require_resumable_worker_policy("epoch_reseed"))

    def test_queue_archives_partial_persistent_run_without_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            outdir = Path(directory) / "lr_1e-4"
            outdir.mkdir()
            (outdir / "latest.pth").write_bytes(b"partial")
            first = archive_incomplete_persistent_run(outdir)
            self.assertEqual(first, Path(directory) / "lr_1e-4_partial_archive")
            self.assertFalse(outdir.exists())

            outdir.mkdir()
            (outdir / "latest.pth").write_bytes(b"second")
            second = archive_incomplete_persistent_run(outdir)
            self.assertEqual(second, Path(directory) / "lr_1e-4_partial_archive_2")
            self.assertEqual((second / "latest.pth").read_bytes(), b"second")

    def test_queue_leaves_complete_or_empty_run_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            outdir = Path(directory) / "lr_1e-4"
            outdir.mkdir()
            self.assertIsNone(archive_incomplete_persistent_run(outdir))
            (outdir / "summary.json").write_text("{}")
            self.assertIsNone(archive_incomplete_persistent_run(outdir))
            self.assertTrue(outdir.exists())

    def test_persistent_queue_launchers_import_from_script_entrypoint(self):
        launchers = (
            "run_hovernet_matched_control_queue.py",
            "run_hovernet_sampling_control_queue.py",
            "run_hovernet_backbone_pilot_queue.py",
            "run_hovernet_instance_loss_pilot_queue.py",
            "run_hovernet_instance_type_pilot_queue.py",
            "run_hovernet_type_focal_pilot_queue.py",
            "run_hovernet_e44_lr_expansion_queue.py",
            "run_hovernet_intervention_pilot_queue.py",
            "run_hovernet_class_sampling_fraction_queue.py",
            "run_hovernet_e46_post_e45_queue.py",
        )
        for launcher in launchers:
            completed = subprocess.run(
                [sys.executable, str(Path("scripts") / launcher), "--help"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(completed.returncode, 0, msg=f"{launcher}: {completed.stderr}")

    def test_e46_uses_sampler_prior_artifact_only_after_source_audit_passes(self):
        artifact = {
            "prediction_artifact": "raw.npz",
            "sampler_prior_corrected_prediction_artifact": "corrected.npz",
        }
        corrected, used = type_candidate_artifact(artifact, {
            "audit": {"advances_as_sampler_prior_correction": True},
        })
        self.assertEqual(corrected, Path("corrected.npz"))
        self.assertTrue(used)

        raw, used = type_candidate_artifact(artifact, {
            "audit": {"advances_as_sampler_prior_correction": False},
        })
        self.assertEqual(raw, Path("raw.npz"))
        self.assertFalse(used)

    def test_e43_gate_requires_practical_mpq_dq_gain_without_sq_collapse(self):
        control = {"selected_mPQ+": {
            "val_mPQ+": 0.39, "val_mDQ+": 0.49, "val_mSQ+": 0.802,
        }}
        intervention = {"families": {"e37_class_0.5": {"selected_mPQ+": {
            "val_mPQ+": 0.40, "val_mDQ+": 0.50, "val_mSQ+": 0.800,
        }}}}
        passed = gate_evidence({"independently_selected_delta": {}}, intervention, control)
        self.assertTrue(passed["E37_passes"])
        self.assertTrue(passed["passes"])

        intervention["families"]["e37_class_0.5"]["selected_mPQ+"] = {
            "val_mPQ+": 0.391, "val_mDQ+": 0.491, "val_mSQ+": 0.790,
        }
        failed = gate_evidence({"independently_selected_delta": {}}, intervention, control)
        self.assertFalse(failed["E37_passes"])
        self.assertFalse(failed["passes"])

    def test_e44_smoothing_gate_uses_independent_endpoints_and_sq_guardrail(self):
        control = {
            "selected_R2": {"val_R2": 0.60},
            "selected_mPQ+": {"val_mPQ+": 0.39, "val_mDQ+": 0.49, "val_mSQ+": 0.80},
        }
        selected_r2 = {"val_R2": 0.63}
        selected_mpq = {"val_mPQ+": 0.40, "val_mDQ+": 0.50, "val_mSQ+": 0.797}
        delta = type_loss_delta(selected_r2, selected_mpq, control)
        self.assertAlmostEqual(delta["R2"], 0.03)
        self.assertTrue(type_loss_passes_gate(delta))

        selected_mpq["val_mSQ+"] = 0.79
        self.assertFalse(type_loss_passes_gate(type_loss_delta(selected_r2, selected_mpq, control)))

    def test_e44_refuses_a_control_from_another_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "control"
            run.mkdir()
            (run / "summary.json").write_text(__import__("json").dumps({
                "args": {"seed": 206},
            }))
            control = {"runs": {"0.0001": str(run)}}
            with self.assertRaisesRegex(ValueError, "seed-matched control"):
                validate_control_seed(control, 205)
            self.assertEqual(validate_control_seed(control, 206), 206)

    def test_e44_lr_expansion_uses_smoothing_only_for_practical_typed_gain(self):
        pure = {"loss_arguments": ["--type-focal-gamma", "2"]}
        smooth = {
            "loss_arguments": ["--type-focal-gamma", "2", "--type-label-smoothing", "0.05"],
            "delta_vs_pure_combination_at_independently_selected_endpoints": {
                "mPQ+": 0.004, "mDQ+": 0.004, "mSQ+": -0.001,
            },
        }
        report = {"families": {
            "weight_rho3_focal_gamma2": pure,
            "weight_rho3_focal_gamma2_smooth005": smooth,
        }}
        family, _, _ = expansion_recipe(report)
        self.assertEqual(family, "weight_rho3_focal_gamma2_smooth005")
        smooth["delta_vs_pure_combination_at_independently_selected_endpoints"]["mPQ+"] = 0.002
        family, _, _ = expansion_recipe(report)
        self.assertEqual(family, "weight_rho3_focal_gamma2")

    def test_e44_upper_lr_sweep_stops_after_both_metrics_turn_down(self):
        rows = [
            {"learning_rate": 3e-4, "val_R2": 0.63, "val_mPQ+": 0.387},
            {"learning_rate": 6e-4, "val_R2": 0.54, "val_mPQ+": 0.376},
        ]
        audit = upper_lr_bracket(rows, 6e-4)

        self.assertTrue(audit["bracketed"])
        self.assertAlmostEqual(audit["metrics"]["val_R2"]["delta"], -0.09)
        rows[1]["val_mPQ+"] = 0.390
        self.assertFalse(upper_lr_bracket(rows, 6e-4)["bracketed"])

    def test_diagnostic_backfill_ignores_causally_skipped_e44_family(self):
        complete = {"selected_R2": {}, "selected_mPQ+": {}}
        payload = {"families": {
            "weight_rho3": complete,
            "weight_rho3_focal_gamma2_smooth005": {
                "status": "skipped by predeclared pure-combination causal gate",
                "runs": {},
            },
        }}
        self.assertEqual(selected_type_families(payload), [("weight_rho3", complete)])
        with self.assertRaises(KeyError):
            selected_type_families({"families": {"broken_complete_family": {}}})

    def test_diagnostic_backfill_skips_e44_after_expanded_control_rejection(self):
        rejected = {
            "independently_selected_delta": {"R2": 0.0037, "mPQ+": -0.0038},
            "promotion_audit": {"typed_signal": False, "count_tails_safe": False},
        }
        typed = {
            "independently_selected_delta": {"R2": -0.1, "mPQ+": 0.01},
            "promotion_audit": {"typed_signal": True, "count_tails_safe": False},
        }

        self.assertFalse(e44_backfill_eligible(rejected))
        self.assertTrue(e44_backfill_eligible(typed))

    def test_diagnostic_backfill_waits_for_e43_terminal_summary(self):
        args = SimpleNamespace(
            wait_for=Path("e44_summary.json"),
            instance_type_summary=Path("e43_summary.json"),
            type_lr_expansion_summary=Path("e44_lr_expansion_summary.json"),
        )
        self.assertEqual(
            training_prerequisites(args),
            [
                Path("e44_summary.json"),
                Path("e43_summary.json"),
                Path("e44_lr_expansion_summary.json"),
            ],
        )

    def test_historical_single_dose_candidate_constraint_holds_composition(self):
        selection = {"learning_rate": 1e-4, "epoch": 10}
        constraint = single_dose_selection_constraint(selection)
        self.assertEqual(constraint["unconfirmed_boundaries"], ["intervention_strength"])
        boundary = hyperparameter_boundary_audit(constraint)
        self.assertFalse(boundary["passes"])
        self.assertIn("intervention_strength", boundary["unconfirmed_boundaries"])

    def test_single_dose_constraint_preserves_learning_rate_boundary(self):
        constraint = single_dose_selection_constraint(
            {"learning_rate": 3e-4, "epoch": 5},
            {"requires_learning_rate_expansion": True},
        )
        self.assertEqual(
            constraint["unconfirmed_boundaries"],
            ["intervention_strength", "learning_rate"],
        )

    def test_e37_complementarity_uses_exact_seed206_control(self):
        self.assertEqual(
            matched_resnet_control_label("e37_class_0.5_mPQ+", "mPQ+"),
            ("e37_no_sampling_seed206_mPQ+", "seed-206 uniform ResNet-50"),
        )
        self.assertEqual(
            matched_resnet_control_label("e41_seresnext101_R2", "R2"),
            ("e36_no_hed_seed205_R2", "seed-205 ResNet-50"),
        )
        self.assertTrue(is_complementarity_candidate_label("e37_class_0.5_mPQ+", "mPQ+"))
        self.assertTrue(is_complementarity_candidate_label("e37_source_0.5_R2", "R2"))
        self.assertFalse(is_complementarity_candidate_label("e37_no_sampling_seed206_R2", "R2"))

    def test_master_queue_serializes_e43_before_e44_and_backfill(self):
        root = Path("/tmp/conic_queue_order_test")
        args = SimpleNamespace(
            prepared=root / "prepared",
            train_ids=root / "train.npy",
            val_ids=root / "val.npy",
            poll_seconds=30,
            epochs=10,
            batch_size=6,
            workers=8,
            learning_rates=[3e-5, 1e-4, 3e-4],
            intervention_root=root / "interventions",
            backbone_root=root / "backbones",
            instance_root=root / "instance",
            instance_type_root=root / "instance_type",
            type_root=root / "type",
            backfill_root=root / "backfill",
            audit_root=root / "audits",
            analysis_root=root / "analysis",
            resnet_backbone=root / "resnet.pth",
            seresnext_backbone=root / "seresnext.pth",
            val_batch_size=24,
            backbone_val_batch_size=12,
        )
        stages = hovernet_stage_commands(args)
        labels = [label for label, _ in stages]

        self.assertLess(labels.index("E42 instance-equalized foreground loss"), labels.index("E43 one-loss-per-nucleus pooled type supervision"))
        self.assertLess(labels.index("E43 one-loss-per-nucleus pooled type supervision"), labels.index("E44 type-loss imbalance ablations"))
        self.assertLess(labels.index("E44 type-loss imbalance ablations"), labels.index("selected-checkpoint diagnostic backfill and causal audits"))
        e44_command = dict(stages)["E44 type-loss imbalance ablations"]
        self.assertIn(str(root / "instance_type" / "e43_instance_type_summary.json"), e44_command)

    def test_hovernet_family_plot_excludes_archived_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("lr_3e-5", "lr_3e-5_resume_rng_bug_archive"):
                run = root / name
                run.mkdir()
                (run / "training_curve.json").write_text(
                    '[{"learning_rate": 0.00003, "decoder_learning_rate": 0.00003}]'
                )
            self.assertEqual(set(load_hovernet_family_runs(root)), {"lr_3e-5"})
            self.assertEqual(set(load_hovernet_analysis_family(root)), {3e-5})

    def test_count_error_delta_retains_signed_and_directional_tail_changes(self):
        candidate = {"val_count_error": {
            "points": 10, "mean_signed_error": -1.0,
            "under_error_lt_minus_10_fraction": 0.2, "over_error_gt_10_fraction": 0.1,
        }}
        control = {"val_count_error": {
            "points": 10, "mean_signed_error": 1.0,
            "under_error_lt_minus_10_fraction": 0.1, "over_error_gt_10_fraction": 0.2,
        }}
        result = count_error_deltas(candidate, control)
        self.assertAlmostEqual(result["delta"]["mean_signed_error"], -2.0)
        self.assertAlmostEqual(result["delta"]["under_error_lt_minus_10_fraction"], 0.1)
        self.assertAlmostEqual(result["delta"]["over_error_gt_10_fraction"], -0.1)

    def test_lr_selection_boundary_flags_require_expansion_and_horizon_extension(self):
        runs = {
            3e-5: {"rows": [{"epoch": 1, "val_R2": 0.1, "val_mPQ+": 0.2}]},
            1e-4: {"rows": [{"epoch": 1, "val_R2": 0.2, "val_mPQ+": 0.3}]},
            3e-4: {"rows": [
                {"epoch": 1, "val_R2": 0.3, "val_mPQ+": 0.3},
                {"epoch": 10, "val_R2": 0.5, "val_mPQ+": 0.4},
            ]},
        }
        flags = selection_boundary_flags(runs, {"learning_rate": 3e-4, "epoch": 10})
        self.assertTrue(flags["at_upper_learning_rate_boundary"])
        self.assertTrue(flags["at_scored_horizon_boundary"])
        self.assertTrue(flags["requires_lr_expansion_before_promotion"])
        self.assertTrue(flags["requires_horizon_extension_before_promotion"])

    def test_sampling_exposure_summary_uses_realized_draws_and_duplicates(self):
        rows = pd.DataFrame({
            "patch_id": [10, 11, 12],
            "source": ["a", "a", "b"],
            **{
                f"count_{name}": [1 if index == class_index else 0 for index in range(3)]
                for class_index, name in enumerate(CLASS_NAMES)
            },
        })
        summary = sampling_exposure_summary([10, 10, 12], rows)
        self.assertEqual(summary["draws"], 3)
        self.assertEqual(summary["unique_patches"], 2)
        self.assertAlmostEqual(summary["unique_patch_fraction"], 2 / 3)
        self.assertAlmostEqual(summary["source_draw_fraction"]["a"], 2 / 3)
        self.assertAlmostEqual(summary["class_positive_patch_fraction"]["neutrophil"], 2 / 3)
        self.assertAlmostEqual(summary["nucleus_class_fraction"]["neutrophil"], 2 / 3)
        self.assertEqual(
            summary["draw_sequence_sha256"],
            sampling_exposure_summary([10, 10, 12], rows)["draw_sequence_sha256"],
        )
        self.assertNotEqual(
            summary["draw_sequence_sha256"],
            sampling_exposure_summary([10, 12, 10], rows)["draw_sequence_sha256"],
        )

    def test_instance_type_confusion_separates_swaps_misses_and_spurious_detections(self):
        truth = np.zeros((1, 8, 8, 2), dtype=np.int32)
        prediction = np.zeros_like(truth)
        truth[0, 1:3, 1:3, 0] = 1
        truth[0, 1:3, 1:3, 1] = 2
        truth[0, 5:7, 1:3, 0] = 2
        truth[0, 5:7, 1:3, 1] = 4
        prediction[0, 1:3, 1:3, 0] = 8
        prediction[0, 1:3, 1:3, 1] = 5
        prediction[0, 5:7, 5:7, 0] = 9
        prediction[0, 5:7, 5:7, 1] = 1
        result = instance_type_confusion(truth, prediction)
        matrix = np.asarray(result["matrix"])
        self.assertEqual(matrix[1, 4], 1)  # GT epithelial, predicted eosinophil.
        self.assertEqual(matrix[3, -1], 1)  # GT plasma was missed.
        self.assertEqual(matrix[-1, 0], 1)  # Spurious neutrophil prediction.
        self.assertEqual(result["geometry_matched"], 1)
        self.assertEqual(result["correctly_typed"], 0)

    def test_type_confusion_driver_reports_directional_error_changes(self):
        labels = [*CLASS_NAMES, "unmatched"]
        control_matrix = np.zeros((7, 7), dtype=int)
        candidate_matrix = np.zeros((7, 7), dtype=int)
        control_matrix[0, 1] = 4
        candidate_matrix[0, 1] = 2
        control_matrix[4, -1] = 3
        candidate_matrix[4, -1] = 1
        control = {"val_instance_type_confusion": {
            "labels": labels, "matrix": control_matrix.tolist(), "matched_type_accuracy": 0.5
        }}
        candidate = {"val_instance_type_confusion": {
            "labels": labels, "matrix": candidate_matrix.tolist(), "matched_type_accuracy": 0.7
        }}
        result = type_confusion_drivers(candidate, control)
        self.assertTrue(result["available"])
        self.assertEqual(result["largest_typing_swap_changes"][0]["delta"], -2)
        self.assertEqual(result["missed_truth_delta_by_class"]["eosinophil"], -2)
        self.assertAlmostEqual(result["delta_matched_type_accuracy"], 0.2)

    def test_complement_frequency_type_weights_upweight_rare_foreground_classes(self):
        weights = complement_frequency_type_weights(
            np.asarray([10, 1000, 300, 80, 8, 300], dtype=np.float64), rho=3.0
        )
        self.assertEqual(weights.shape, (7,))
        self.assertAlmostEqual(float(weights[0]), 1.0)
        self.assertAlmostEqual(float(weights[1:].mean()), 1.0, places=6)
        self.assertGreater(float(weights[1]), float(weights[2]))
        self.assertGreater(float(weights[5]), float(weights[2]))

    def test_weighted_focal_type_loss_preserves_ce_at_zero_settings(self):
        target = torch.nn.functional.one_hot(torch.tensor([[[0, 2]]]), num_classes=3).float()
        probability = torch.tensor([[[[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]]]], dtype=torch.float32)
        expected = -(torch.log(torch.tensor(0.7)) + torch.log(torch.tensor(0.7))) / 2
        observed = weighted_focal_xentropy_loss(
            target, probability, torch.ones(3), gamma=0.0, label_smoothing=0.0
        )
        self.assertAlmostEqual(float(observed), float(expected), places=6)

    def test_weighted_focal_type_loss_emphasizes_hard_and_upweighted_targets(self):
        target = torch.nn.functional.one_hot(torch.tensor([[[1, 2]]]), num_classes=3).float()
        probability = torch.tensor([[[[0.01, 0.90, 0.09], [0.01, 0.89, 0.10]]]], dtype=torch.float32)
        ordinary = weighted_focal_xentropy_loss(target, probability, torch.ones(3), gamma=2.0)
        upweighted = weighted_focal_xentropy_loss(
            target, probability, torch.tensor([1.0, 1.0, 3.0]), gamma=2.0
        )
        self.assertGreater(float(upweighted), float(ordinary))

    def test_instance_type_diagnostics_are_equal_nucleus_averages(self):
        instances = torch.tensor([[[1, 1, 1, 2]]], dtype=torch.long)
        types = torch.tensor([[[1, 1, 1, 2]]], dtype=torch.long)
        probabilities = torch.tensor(
            [[[[0.01, 0.98, 0.01], [0.01, 0.98, 0.01], [0.01, 0.98, 0.01], [0.01, 0.01, 0.98]]]],
            dtype=torch.float32,
        )
        stats = instance_type_diagnostic_sums(instances, types, probabilities)
        self.assertEqual(stats["nuclei"], 2)
        self.assertAlmostEqual(stats["target_probability"] / stats["nuclei"], 0.98, places=6)
        self.assertAlmostEqual(stats["pixel_accuracy"] / stats["nuclei"], 1.0, places=6)
        self.assertLess(stats["entropy"] / stats["nuclei"], 0.2)
        self.assertAlmostEqual(stats["spatial_js_disagreement"], 0.0, places=6)

    def test_instance_type_spatial_disagreement_distinguishes_equal_pooled_probabilities(self):
        instances = torch.tensor([[[1, 1]]], dtype=torch.long)
        types = torch.tensor([[[1, 1]]], dtype=torch.long)
        consistent = torch.tensor(
            [[[[0.01, 0.495, 0.495], [0.01, 0.495, 0.495]]]], dtype=torch.float32
        )
        contradictory = torch.tensor(
            [[[[0.01, 0.98, 0.01], [0.01, 0.01, 0.98]]]], dtype=torch.float32
        )
        consistent_stats = instance_type_diagnostic_sums(instances, types, consistent)
        contradictory_stats = instance_type_diagnostic_sums(instances, types, contradictory)
        self.assertAlmostEqual(consistent_stats["entropy"], contradictory_stats["entropy"], places=6)
        self.assertAlmostEqual(consistent_stats["spatial_js_disagreement"], 0.0, places=6)
        self.assertGreater(contradictory_stats["spatial_js_disagreement"], 0.4)

    def test_instance_pooled_type_loss_scores_each_nucleus_after_pooling(self):
        instances = torch.tensor([[[1, 1, 0], [2, 2, 2]]], dtype=torch.long)
        types = torch.tensor([[[2, 2, 0], [4, 4, 4]]], dtype=torch.long)
        correct = torch.full((1, 2, 3, 7), 1.0e-4, dtype=torch.float32)
        correct[0, 0, :2, 2] = 0.9994
        correct[0, 1, :, 4] = 0.9994
        wrong = correct.clone()
        wrong[0, 1, :, 4] = 1.0e-4
        wrong[0, 1, :, 3] = 0.9994
        self.assertLess(
            float(instance_pooled_type_loss(instances, types, correct)),
            float(instance_pooled_type_loss(instances, types, wrong)),
        )

    def test_instance_pooled_type_loss_is_invariant_to_nucleus_area_duplication(self):
        small_instances = torch.tensor([[[1, 2]]], dtype=torch.long)
        small_types = torch.tensor([[[1, 2]]], dtype=torch.long)
        small_probs = torch.tensor([[[[0.0, 0.8, 0.2], [0.0, 0.4, 0.6]]]], dtype=torch.float32)
        large_instances = torch.tensor([[[1, 1, 1, 2]]], dtype=torch.long)
        large_types = torch.tensor([[[1, 1, 1, 2]]], dtype=torch.long)
        large_probs = torch.tensor(
            [[[[0.0, 0.8, 0.2], [0.0, 0.8, 0.2], [0.0, 0.8, 0.2], [0.0, 0.4, 0.6]]]],
            dtype=torch.float32,
        )
        self.assertAlmostEqual(
            float(instance_pooled_type_loss(small_instances, small_types, small_probs)),
            float(instance_pooled_type_loss(large_instances, large_types, large_probs)),
            places=6,
        )

    def test_instance_pooled_type_loss_backpropagates_to_every_nucleus(self):
        instances = torch.tensor([[[1, 1, 2, 2]]], dtype=torch.long)
        types = torch.tensor([[[1, 1, 2, 2]]], dtype=torch.long)
        probabilities = torch.tensor(
            [[[[0.1, 0.7, 0.2], [0.1, 0.6, 0.3], [0.1, 0.2, 0.7], [0.1, 0.3, 0.6]]]],
            dtype=torch.float32,
            requires_grad=True,
        )
        loss = instance_pooled_type_loss(instances, types, probabilities)
        loss.backward()
        self.assertTrue(torch.isfinite(probabilities.grad).all())
        self.assertTrue((probabilities.grad[0, 0, :2, 1] < 0).all())
        self.assertTrue((probabilities.grad[0, 0, 2:, 2] < 0).all())

    def test_instance_loss_selection_keeps_r2_and_mpq_endpoints_independent(self):
        rows = [
            {"val_R2": 0.72, "val_mPQ+": 0.40, "epoch": 10},
            {"val_R2": 0.68, "val_mPQ+": 0.45, "epoch": 5},
        ]
        self.assertEqual(select_instance_loss(rows, "val_R2")["val_R2"], 0.72)
        self.assertEqual(select_instance_loss(rows, "val_mPQ+")["val_mPQ+"], 0.45)

    def test_r2_sse_driver_decomposition_reconstructs_macro_delta(self):
        control = {
            "val_R2": 0.0,
            "val_per_class_SST": {name: 10.0 for name in CLASS_NAMES},
            "val_per_source": {
                "a": {"per_class_SSE": {name: 5.0 for name in CLASS_NAMES}, "count_error": {"mean_signed_error": 1.0, "MAE": 2.0, "absolute_error_gt_5_fraction": 0.2, "absolute_error_gt_10_fraction": 0.1}},
                "b": {"per_class_SSE": {name: 5.0 for name in CLASS_NAMES}, "count_error": {"mean_signed_error": -1.0, "MAE": 2.0, "absolute_error_gt_5_fraction": 0.2, "absolute_error_gt_10_fraction": 0.1}},
            },
        }
        candidate = {
            "val_R2": 0.3,
            "val_per_class_SST": control["val_per_class_SST"],
            "val_per_source": {
                "a": {"per_class_SSE": {name: 2.0 for name in CLASS_NAMES}, "count_error": {"mean_signed_error": 0.0, "MAE": 1.0, "absolute_error_gt_5_fraction": 0.1, "absolute_error_gt_10_fraction": 0.0}},
                "b": control["val_per_source"]["b"],
            },
        }
        report = r2_sse_drivers(candidate, control)
        self.assertTrue(report["available"])
        self.assertAlmostEqual(report["reconstructed_delta_R2"], 0.3)
        self.assertAlmostEqual(report["by_source"]["a"]["delta_macro_R2_contribution"], 0.3)

    def test_mpq_source_counterfactual_uses_pooled_sufficient_statistics(self):
        def source(tp: int, fp: int, fn: int, iou: float) -> dict:
            return {"per_class_PQ_stats": {name: {"tp": tp, "fp": fp, "fn": fn, "sum_iou": iou} for name in CLASS_NAMES}}
        control_sources = {"a": source(5, 5, 5, 4.0), "b": source(5, 5, 5, 4.0)}
        candidate_sources = {"a": source(8, 2, 2, 6.4), "b": control_sources["b"]}
        control = {"val_mPQ+": 0.4, "val_per_source": control_sources}
        candidate = {"val_mPQ+": 0.56, "val_per_source": candidate_sources}
        report = mpq_source_counterfactuals(candidate, control)
        self.assertTrue(report["available"])
        self.assertGreater(report["one_source_counterfactuals"][0]["one_source_delta_mPQ+"], 0)

    def test_count_error_stats_preserve_direction_and_outlier_proportions(self):
        truth = np.asarray([[0, 10], [4, 2]], dtype=np.int64)
        predicted = np.asarray([[3, 4], [4, 13]], dtype=np.int64)
        stats = count_error_stats(truth, predicted)
        self.assertEqual(stats["points"], 4)
        self.assertAlmostEqual(stats["mean_signed_error"], 2.0)
        self.assertAlmostEqual(stats["MAE"], 5.0)
        self.assertAlmostEqual(stats["under_fraction"], 0.25)
        self.assertAlmostEqual(stats["exact_fraction"], 0.25)
        self.assertAlmostEqual(stats["over_fraction"], 0.5)
        self.assertAlmostEqual(stats["absolute_error_gt_5_fraction"], 0.5)
        self.assertAlmostEqual(stats["under_error_lt_minus_5_fraction"], 0.25)
        self.assertAlmostEqual(stats["over_error_gt_10_fraction"], 0.25)

    def test_training_zero_truth_logger_matches_complementarity_audit(self):
        truth = np.zeros((4, 6), dtype=np.float64)
        truth[:, 0] = [0, 1, 0, 2]
        predicted = np.zeros_like(truth)
        predicted[:, 1] = [0, 6, 11, 21]
        logged = zero_truth_count_stats(truth, predicted)
        audited = zero_truth_overcount_summary(truth, predicted)
        self.assertEqual(logged, audited)
        self.assertEqual(logged["per_class"]["epithelial"]["support"], 4)

    def test_instance_equalized_pixel_weights_preserve_mass_and_equalize_instances(self):
        instance_map = torch.tensor(
            [[[0, 1, 1, 1, 1], [0, 2, 2, 0, 0], [0, 2, 2, 0, 0]]], dtype=torch.long
        )
        ordinary = instance_equalized_pixel_weights(instance_map, blend=0.0)
        equalized = instance_equalized_pixel_weights(instance_map, blend=1.0)
        self.assertTrue(torch.equal(ordinary, torch.ones_like(ordinary)))
        foreground = instance_map > 0
        self.assertAlmostEqual(float(equalized[foreground].sum()), float(foreground.sum()))
        mass_one = float(equalized[instance_map == 1].sum())
        mass_two = float(equalized[instance_map == 2].sum())
        self.assertAlmostEqual(mass_one, mass_two)
        self.assertTrue(torch.equal(equalized[instance_map == 0], torch.ones_like(equalized[instance_map == 0])))

    def test_instance_equalized_pixel_weights_blend_is_bounded(self):
        instance_map = torch.tensor([[[1, 1, 1, 2]]], dtype=torch.long)
        halfway = instance_equalized_pixel_weights(instance_map, blend=0.5)
        expected = 0.5 * (
            instance_equalized_pixel_weights(instance_map, blend=0.0)
            + instance_equalized_pixel_weights(instance_map, blend=1.0)
        )
        self.assertTrue(torch.allclose(halfway, expected))
        with self.assertRaises(ValueError):
            instance_equalized_pixel_weights(instance_map, blend=1.1)

    def test_capped_instance_weights_preserve_mass_and_cap(self):
        instance_map = torch.zeros((1, 12, 12), dtype=torch.long)
        instance_map[0, 0, 0] = 1
        instance_map[0, 2:10, 2:10] = 2
        capped = instance_equalized_pixel_weights(instance_map, blend=0.5, max_weight=4.0)
        foreground = instance_map > 0
        self.assertAlmostEqual(
            float(capped[foreground].sum()),
            float(foreground.sum()),
            places=4,
        )
        self.assertLessEqual(float(capped[foreground].max()), 4.0 + 1.0e-6)
        self.assertTrue(torch.equal(capped[~foreground], torch.ones_like(capped[~foreground])))
        self.assertTrue(torch.allclose(
            instance_equalized_pixel_weights(instance_map, blend=0.5, max_weight=1.0),
            torch.ones_like(capped),
        ))
        with self.assertRaises(ValueError):
            instance_equalized_pixel_weights(instance_map, blend=0.5, max_weight=0.5)

    def test_capped_instance_weights_hold_across_size_ratios_and_blends(self):
        sizes = [1, 2, 5, 20, 100]
        labels = torch.cat([
            torch.full((size,), instance_id, dtype=torch.long)
            for instance_id, size in enumerate(sizes, start=1)
        ])
        instance_map = torch.cat([torch.zeros(17, dtype=torch.long), labels]).reshape(1, 5, 29)
        foreground = instance_map > 0
        for blend in (0.25, 0.5, 1.0):
            for cap in (1.0, 1.5, 4.0, 8.0):
                weights = instance_equalized_pixel_weights(
                    instance_map,
                    blend=blend,
                    max_weight=cap,
                )
                self.assertAlmostEqual(
                    float(weights[foreground].mean()),
                    1.0,
                    places=5,
                    msg=f"blend={blend}, cap={cap}",
                )
                self.assertLessEqual(
                    float(weights[foreground].max()),
                    cap + 1.0e-5,
                    msg=f"blend={blend}, cap={cap}",
                )
                self.assertTrue(torch.equal(
                    weights[~foreground],
                    torch.ones_like(weights[~foreground]),
                ))

    def test_instance_equalized_loss_zero_blend_recovers_ordinary_losses(self):
        truth_class = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]]])
        probabilities = torch.tensor([[[[0.8, 0.2], [0.3, 0.7]], [[0.4, 0.6], [0.9, 0.1]]]])
        weights = instance_equalized_pixel_weights(torch.tensor([[[0, 1], [2, 2]]]), blend=0.0)
        self.assertTrue(torch.allclose(
            weighted_xentropy_loss(truth_class, probabilities, weights),
            xentropy_loss(truth_class, probabilities),
        ))
        truth_hv = torch.zeros((1, 2, 2, 2), dtype=torch.float32)
        pred_hv = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2) / 8
        self.assertTrue(torch.allclose(
            weighted_mse_loss(truth_hv, pred_hv, weights),
            mse_loss(truth_hv, pred_hv),
        ))

    def test_binary_segmentation_metrics_separate_coverage_from_instances(self):
        truth = np.zeros((1, 6, 6), dtype=np.int32)
        truth[0, 1:5, 1:3] = 1
        truth[0, 1:5, 3:5] = 2
        merged = np.zeros_like(truth)
        merged[0, 1:5, 1:5] = 1
        metrics = binary_instance_segmentation_metrics(truth, merged, boundary_tolerance=0)
        self.assertAlmostEqual(metrics["foreground_jaccard"], 1.0)
        self.assertAlmostEqual(metrics["foreground_dice"], 1.0)
        self.assertEqual(metrics["bPQ"], 0.0)
        self.assertLess(metrics["AJI+"], 1.0)
        self.assertLess(metrics["boundary_F1"], 1.0)

    def test_binary_segmentation_metrics_are_one_for_perfect_instances(self):
        truth = np.zeros((1, 8, 8), dtype=np.int32)
        truth[0, 1:4, 1:4] = 4
        truth[0, 4:7, 4:7] = 9
        metrics = binary_instance_segmentation_metrics(truth, truth.copy())
        for key in ("foreground_jaccard", "foreground_dice", "bPQ", "binary_DQ", "binary_SQ", "AJI+", "boundary_F1"):
            self.assertAlmostEqual(metrics[key], 1.0, msg=key)

    def test_multiclass_pq_surfaces_detection_and_segmentation_quality(self):
        truth = np.zeros((1, 8, 8, 2), dtype=np.int32)
        prediction = np.zeros_like(truth)
        for class_id in range(1, 7):
            row = class_id - 1
            truth[0, row, 1:3, 0] = class_id
            truth[0, row, 1:3, 1] = class_id
            prediction[0, row, 1:3, 0] = class_id
            prediction[0, row, 1:3, 1] = class_id
        metrics = multiclass_pq_plus(truth, prediction)
        self.assertAlmostEqual(metrics["mPQ+"], 1.0, places=5)
        self.assertAlmostEqual(metrics["mDQ+"], 1.0, places=5)
        self.assertAlmostEqual(metrics["mSQ+"], 1.0, places=5)

    def test_intervention_lr_selection_keeps_metrics_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = {3e-5: root / "slow", 3e-4: root / "fast"}
            for path in runs.values():
                path.mkdir()
            (runs[3e-5] / "training_curve.json").write_text(
                '[{"epoch": 10, "val_mPQ+": 0.44, "val_R2": 0.70}]'
            )
            (runs[3e-4] / "training_curve.json").write_text(
                '[{"epoch": 10, "val_mPQ+": 0.41, "val_R2": 0.78}]'
            )
            self.assertEqual(select_intervention_lr(runs, "val_mPQ+")["learning_rate"], 3e-5)
            self.assertEqual(select_intervention_lr(runs, "val_R2")["learning_rate"], 3e-4)

    def test_hovernet_stain_tta_builds_exactly_two_deterministic_views(self):
        image = np.broadcast_to(np.asarray([185, 115, 155], dtype=np.uint8), (2, 24, 24, 3)).copy()
        image[:, :, :, 0] += np.arange(24, dtype=np.uint8)[None, None, :]
        target = np.asarray([0.08, 0.025], dtype=np.float32)
        first, first_changed = build_stain_views(image, target)
        second, second_changed = build_stain_views(image, target)
        self.assertEqual(len(first), 2)
        self.assertTrue(np.array_equal(first[0], image))
        self.assertTrue(np.array_equal(first[1], second[1]))
        self.assertEqual(first_changed, second_changed)
        self.assertGreater(first_changed, 0)

        native, changed = build_stain_views(image, None)
        self.assertEqual(len(native), 1)
        self.assertEqual(changed, 0)
        self.assertTrue(np.array_equal(native[0], image))

        styled, styled_changed = build_stain_views(image, target, mode="styled")
        self.assertEqual(len(styled), 1)
        self.assertTrue(np.array_equal(styled[0], first[1]))
        self.assertEqual(styled_changed, first_changed)

    def test_hovernet_model_weights_are_normalized_and_uniform_by_default(self):
        self.assertTrue(np.allclose(normalize_model_weights(2, None), [0.5, 0.5]))
        self.assertTrue(np.allclose(normalize_model_weights(2, [1, 3]), [0.25, 0.75]))
        with self.assertRaises(ValueError):
            normalize_model_weights(2, [1])
        with self.assertRaises(ValueError):
            normalize_model_weights(2, [0, 0])
        with self.assertRaises(ValueError):
            normalize_model_weights(2, [-1, 2])

    def test_hovernet_branch_weights_override_only_requested_maps(self):
        weights = resolve_branch_model_weights(
            2,
            model_weights=[1, 3],
            np_weights=[1, 0],
            tp_weights=[2, 2],
        )
        self.assertTrue(np.allclose(weights["np"], [1, 0]))
        self.assertTrue(np.allclose(weights["hv"], [0.25, 0.75]))
        self.assertTrue(np.allclose(weights["tp"], [0.5, 0.5]))
        with self.assertRaises(ValueError):
            resolve_branch_model_weights(2, tp_weights=[1])

    def test_count_complementarity_selects_and_crosschecks_cancelling_residuals(self):
        base = np.arange(12, dtype=np.float64)[:, None]
        truth = base + np.arange(6, dtype=np.float64)[None, :]
        first = truth + 2.0
        second = truth - 2.0
        grid = np.asarray([0.0, 0.5, 1.0])
        selected = select_per_class_weights(truth, first, second, grid)
        self.assertTrue(np.allclose(selected, 0.5))
        self.assertTrue(np.allclose(blend_counts(first, second, selected), truth))
        sources = np.asarray(["a"] * 6 + ["b"] * 6)
        audit = leave_one_source_out(truth, first, second, sources, grid)
        self.assertAlmostEqual(audit["pooled_out_of_source"]["R2"], 1.0)
        for weights in audit["selected_second_model_weights_by_held_source"].values():
            self.assertTrue(all(value == 0.5 for value in weights.values()))

    def test_zero_truth_overcount_summary_reports_outlier_prevalence_with_support(self):
        truth = np.zeros((4, 6), dtype=np.float64)
        truth[:, 0] = [0, 1, 0, 2]
        prediction = np.zeros_like(truth)
        prediction[:, 1] = [0, 6, 11, 21]
        summary = zero_truth_overcount_summary(truth, prediction)
        epithelial = summary["per_class"]["epithelial"]
        self.assertEqual(epithelial["support"], 4)
        self.assertAlmostEqual(epithelial["over_5_fraction"], 0.75)
        self.assertAlmostEqual(epithelial["over_10_fraction"], 0.5)
        self.assertAlmostEqual(epithelial["over_20_fraction"], 0.25)

    def test_e46_rejects_supported_source_class_zero_truth_tail_regression(self):
        def error():
            return {
                "mean_signed_error": 1.0, "MAE": 4.0,
                "absolute_error_gt_10_fraction": 0.10,
                "absolute_error_gt_20_fraction": 0.05,
            }

        def zero(over_10, over_20):
            row = {
                "support": 30, "nonzero_fraction": 0.5,
                "over_5_fraction": max(over_10, 0.2),
                "over_10_fraction": over_10, "over_20_fraction": over_20,
                "mean_prediction": 3.0, "max_prediction": 25.0,
            }
            return {"all_zero_truth_points": row, "per_class": {"epithelial": row}}

        first = {
            "name": "control", "R2": 0.60, "count_error": error(),
            "zero_truth_overcount": zero(0.10, 0.03),
            "by_source": {"dpath": {"zero_truth_overcount": zero(0.10, 0.03)}},
        }
        second = {
            "name": "candidate", "R2": 0.64, "count_error": error(),
            "zero_truth_overcount": zero(0.10, 0.03),
            "by_source": {"dpath": {"zero_truth_overcount": zero(0.40, 0.15)}},
        }
        report = {
            "first_model": first, "second_model": second,
            "leave_one_source_out": {
                "pooled_out_of_source": first,
                "held_out_sources": {"dpath": first["by_source"]["dpath"]},
                "selected_second_model_weights_by_held_source": {"dpath": {"epithelial": 0.5}},
            },
            "stability": {
                "full_validation_per_class_blend_delta_R2": 0.01,
                "full_minus_cross_source_R2": 0.0,
            },
            "evaluation_set": {"patches": 100},
        }
        audit = count_admission(report)
        self.assertFalse(audit["standalone_candidate"]["zero_truth_tail_audit"]["passes"])
        self.assertFalse(audit["standalone_candidate"]["advances_to_mature_training_screen"])

    def test_type_complementarity_relabels_only_iou_matched_control_instances(self):
        control = np.zeros((6, 6, 2), dtype=np.int32)
        candidate = np.zeros_like(control)
        control[1:3, 1:3, 0], control[1:3, 1:3, 1] = 1, 1
        control[3:5, 3:5, 0], control[3:5, 3:5, 1] = 2, 2
        candidate[1:3, 1:3, 0], candidate[1:3, 1:3, 1] = 7, 2
        self.assertEqual(match_instances(control[..., 0], candidate[..., 0]), {1: 7})
        control_probs = {
            (11, 1): np.asarray([0.9, 0.1, 0, 0, 0, 0]),
            (11, 2): np.asarray([0.1, 0.9, 0, 0, 0, 0]),
        }
        candidate_probs = {(11, 7): np.asarray([0.1, 0.9, 0, 0, 0, 0])}
        classes, blended = blended_classes_for_patch(
            11, control, candidate, control_probs, candidate_probs, candidate_weight=1.0
        )
        self.assertEqual(blended, 1)
        self.assertTrue(np.all(classes[control[..., 0] == 1] == 2))
        self.assertTrue(np.all(classes[control[..., 0] == 2] == 2))
        self.assertTrue(np.all(classes[control[..., 0] == 0] == 0))

    def test_type_complementarity_zero_weight_is_exact_decoder_control(self):
        control = np.zeros((6, 6, 2), dtype=np.int32)
        candidate = np.zeros_like(control)
        control[1:3, 1:3, 0], control[1:3, 1:3, 1] = 1, 1
        candidate[1:3, 1:3, 0], candidate[1:3, 1:3, 1] = 7, 2
        # Deliberately make the pooled control probability disagree with the
        # decoder class.  Weight zero must still be the bitwise raw control.
        control_probs = {(11, 1): np.asarray([0.1, 0.9, 0, 0, 0, 0])}
        candidate_probs = {(11, 7): np.asarray([0.1, 0.9, 0, 0, 0, 0])}
        classes, blended = blended_classes_for_patch(
            11, control, candidate, control_probs, candidate_probs, candidate_weight=0.0
        )
        self.assertEqual(blended, 0)
        np.testing.assert_array_equal(classes, control[..., 1])

    def test_type_complementarity_weight_one_is_candidate_decoder_endpoint(self):
        control = np.zeros((6, 6, 2), dtype=np.int32)
        candidate = np.zeros_like(control)
        control[1:3, 1:3, 0], control[1:3, 1:3, 1] = 1, 1
        candidate[1:3, 1:3, 0], candidate[1:3, 1:3, 1] = 7, 2
        control_probs = {(11, 1): np.asarray([0.9, 0.1, 0, 0, 0, 0])}
        # Mean probability deliberately conflicts with the candidate decoder's
        # majority-pixel class.  The endpoint must remain interpretable.
        candidate_probs = {(11, 7): np.asarray([0.6, 0.4, 0, 0, 0, 0])}
        classes, blended = blended_classes_for_patch(
            11, control, candidate, control_probs, candidate_probs, candidate_weight=1.0
        )
        self.assertEqual(blended, 1)
        self.assertTrue(np.all(classes[control[..., 0] == 1] == 2))

        blended_probabilities = {}
        blended_class_lookup_for_patch(
            11,
            control,
            candidate,
            control_probs,
            candidate_probs,
            candidate_weight=1.0,
            assignment_probabilities=blended_probabilities,
        )
        self.assertEqual(int(np.argmax(blended_probabilities[1])) + 1, 2)

    def test_type_complementarity_preserves_unmatched_untyped_control_instance(self):
        control = np.zeros((6, 6, 2), dtype=np.int32)
        candidate = np.zeros_like(control)
        control[1:3, 1:3, 0] = 1
        control_probs = {(11, 1): np.asarray([0.1, 0.9, 0, 0, 0, 0])}
        classes, blended = blended_classes_for_patch(
            11, control, candidate, control_probs, {}, candidate_weight=1.0
        )
        self.assertEqual(blended, 0)
        self.assertTrue(np.all(classes[control[..., 0] == 1] == 0))

    def test_size_detection_audit_reports_paired_recall_with_support(self):
        bins, labels, boundaries = quantile_bin_labels(np.asarray([1, 2, 3, 4, 5, 6, 7, 8]))
        self.assertEqual(len(labels), 4)
        self.assertEqual(len(boundaries), 5)
        self.assertEqual(set(bins.tolist()), {0, 1, 2, 3})
        frame = pd.DataFrame({
            "size_bin": ["small", "small", "large", "large"],
            "control_matched": [True, False, True, False],
            "candidate_matched": [True, True, False, False],
            "control_iou": [0.8, np.nan, 0.7, np.nan],
            "candidate_iou": [0.9, 0.6, np.nan, np.nan],
        })
        rows = {row["size_bin"]: row for row in grouped_detection(frame, ["size_bin"])}
        self.assertEqual(rows["small"]["gt_instances"], 2)
        self.assertAlmostEqual(rows["small"]["delta_detection_recall"], 0.5)
        self.assertAlmostEqual(rows["large"]["delta_detection_recall"], -0.5)

    def test_weighted_sampling_diagnostics_capture_duplicate_cost(self):
        uniform = np.ones(4, dtype=np.float64)
        concentrated = np.asarray([7, 1, 1, 1], dtype=np.float64)
        self.assertAlmostEqual(effective_sample_size(uniform), 4.0)
        self.assertLess(effective_sample_size(concentrated), 2.0)
        self.assertLess(expected_unique_draws(concentrated, 4), expected_unique_draws(uniform, 4))
        self.assertEqual(expected_unique_draws(concentrated, 0), 0.0)

    def test_count_stacker_feature_contracts_and_source_interactions(self):
        counts_a = np.asarray([1, 2, 3, 4], dtype=np.float64)
        counts_b = np.asarray([4, 3, 2, 1], dtype=np.float64)
        sources = np.asarray(["a", "a", "b", "b"])
        levels = ["a", "b"]
        self.assertEqual(design_matrix(counts_a, counts_b, sources, levels, "global_linear").shape, (4, 2))
        self.assertEqual(design_matrix(counts_a, counts_b, sources, levels, "global_quadratic").shape, (4, 5))
        source_matrix = design_matrix(counts_a, counts_b, sources, levels, "source_linear")
        self.assertEqual(source_matrix.shape, (4, 5))
        self.assertTrue(np.all(source_matrix[:2, 2:] == 0))
        self.assertTrue(np.all(source_matrix[2:, 2] == 1))

    def test_count_stacker_rounds_and_clips_predictions(self):
        x = np.arange(8, dtype=np.float64)[:, None]
        y = 2 * x[:, 0] + 1
        model = fit_ridge(x, y, alpha=1.0e-9)
        prediction = predict_ridge(model, np.asarray([[-2.0], [1.2], [10.0]]))
        self.assertEqual(prediction.tolist(), [0, 3, 21])

    def test_hovernet_hed_augmentation_is_seeded_and_geometry_preserving(self):
        image = np.zeros((24, 24, 3), dtype=np.uint8)
        image[..., 0] = np.arange(24, dtype=np.uint8)[None] * 8
        image[..., 1] = 110
        image[..., 2] = 175
        target = np.asarray([0.04, 0.03], dtype=np.float32)
        first = hed_stain_augmentation_array(image, np.random.default_rng(11), probability=1.0, target_concentration=target)
        second = hed_stain_augmentation_array(image, np.random.default_rng(11), probability=1.0, target_concentration=target)
        self.assertEqual(first.shape, image.shape)
        self.assertEqual(first.dtype, np.uint8)
        self.assertTrue(np.array_equal(first, second))
        self.assertFalse(np.array_equal(first, image))

    def test_hovernet_hed_augmentation_does_not_bleach_pale_tissue(self):
        ramp = np.arange(32, dtype=np.uint8)[None, :, None]
        image = np.broadcast_to(np.asarray([225, 205, 220], dtype=np.uint8), (32, 32, 3)).copy()
        image = np.clip(image.astype(np.int16) - ramp.astype(np.int16), 0, 255).astype(np.uint8)
        augmented = hed_stain_augmentation_array(
            image,
            np.random.default_rng(7),
            probability=1.0,
            target_concentration=np.asarray([0.02, 0.015], dtype=np.float32),
        )
        original_luminance = image.mean(axis=-1)
        augmented_luminance = augmented.mean(axis=-1)
        self.assertGreaterEqual(augmented_luminance.std(), 0.5 * original_luminance.std())
        self.assertLess(abs(augmented_luminance.mean() - original_luminance.mean()), 0.2 * 255)

    def test_empirical_hed_target_bank_samples_joint_observations_by_source(self):
        concentrations = np.asarray([[0.01, 0.02], [0.02, 0.04], [0.08, 0.03]], dtype=np.float32)
        bank = EmpiricalHEDTargetBank(concentrations, np.asarray(["major", "major", "minor"]), jitter=0.0)
        rng = np.random.default_rng(9)
        sampled = [bank.sample(rng) for _ in range(1000)]
        minor_fraction = np.mean([source == "minor" for _, source in sampled])
        self.assertGreater(minor_fraction, 0.45)
        self.assertLess(minor_fraction, 0.55)
        observed = {tuple(row) for row in concentrations.tolist()}
        self.assertTrue(all(tuple(target.tolist()) in observed for target, _ in sampled))

    def test_hed_rng_does_not_perturb_official_imgaug_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared = Path(directory)
            (prepared / "images").mkdir()
            (prepared / "labels").mkdir()
            image = np.full((256, 256, 3), [180, 110, 150], dtype=np.uint8)
            Image.fromarray(image).save(prepared / "images" / "00000.png")
            instance = np.zeros((256, 256), dtype=np.int32)
            instance[96:160, 96:160] = 1
            class_map = (instance > 0).astype(np.int32)
            np.save(prepared / "labels" / "00000.npy", {"inst_map": instance, "class_map": class_map})
            code = """
import hashlib, sys
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch
import numpy as np, pandas as pd
from scripts.train_hovernet_our_split import PreparedHoverDataset, seed_everything
prepared, mode = Path(sys.argv[1]), sys.argv[2]
seed_everything(17)
kwargs = {}
if mode == 'hed':
    kwargs = dict(hed_probability=1.0, hed_target_concentrations=np.asarray([[0.2, 0.1]], np.float32), hed_target_sources=np.asarray(['source']))
dataset = PreparedHoverDataset(prepared, pd.DataFrame({'patch_id': [0]}), train=True, seed=17, **kwargs)
digest = hashlib.sha256()
context = patch('scripts.train_hovernet_our_split.hed_stain_augmentation_array', side_effect=lambda image, *args: image) if mode == 'hed' else nullcontext()
with context:
    for _ in range(3):
        item = dataset[0]
        for key in ('img', 'inst_map', 'np_map', 'hv_map', 'tp_map'):
            digest.update(np.ascontiguousarray(item[key]).tobytes())
print(digest.hexdigest())
"""
            hashes = []
            for mode in ("clean", "hed"):
                completed = subprocess.run(
                    [sys.executable, "-c", code, str(prepared), mode],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                hashes.append(completed.stdout.strip())
            self.assertEqual(hashes[0], hashes[1])

    def test_prediction_route_pools_additive_pq_statistics(self):
        stats = np.tile(np.asarray([5, 2, 3, 4.0], dtype=np.float64), (6, 1))
        pooled = pooled_pq_from_stats(stats)
        self.assertAlmostEqual(pooled["mPQ+"], 4.0 / (5 + 0.5 * 2 + 0.5 * 3))
        self.assertEqual(pooled["per_class"][CLASS_NAMES[0]]["tp"], 5)

    def test_patch_pq_pooling_uses_additive_statistics(self):
        stats = {
            "tp": np.asarray([[1] * 6, [2] * 6]),
            "fp": np.asarray([[1] * 6, [0] * 6]),
            "fn": np.asarray([[0] * 6, [1] * 6]),
            "sum_iou": np.asarray([[0.8] * 6, [1.4] * 6]),
        }
        self.assertAlmostEqual(pooled_mpq(stats, np.asarray([0, 1])), 2.2 / 4.0)

    def test_source_route_selects_whole_patches(self):
        metadata = pd.DataFrame({"patch_id": [0, 1, 2], "source": ["a", "b", "a"]})
        selected = routed_patch_ids(metadata, {"a": "x", "b": "y"}, "x")
        self.assertEqual(selected.tolist(), [0, 2])

    def test_source_pq_sufficient_statistics_pool_exactly(self):
        items = []
        for tp, fp, fn, iou in [(2, 1, 3, 1.5), (3, 2, 1, 2.4)]:
            items.append({"per_class_pq": {name: {"tp": tp, "fp": fp, "fn": fn, "sum_iou": iou} for name in CLASS_NAMES}})
        result = pooled_pq(items)
        expected = 3.9 / (5 + 0.5 * 3 + 0.5 * 4)
        self.assertAlmostEqual(result["mPQ+"], expected)

    def test_per_class_confidence_thresholds_reject_only_selected_class(self):
        probabilities = np.asarray([[0.6, 0.4], [0.2, 0.8], [0.45, 0.55]], dtype=np.float32)
        assignments = apply_class_thresholds(probabilities, np.asarray([0.7, 0.6]))
        self.assertEqual(assignments.tolist(), [0, 2, 0])

    def test_probability_level_counts_respect_center_and_dustbin(self):
        counts = counts_from_assignments(
            np.asarray([0, 0, 0, 1], dtype=np.int32),
            np.asarray([1, 0, 2, 6], dtype=np.int8),
            np.asarray([True, True, False, True]),
            2,
        )
        expected = np.zeros((2, 6), dtype=np.int32)
        expected[0, 0] = 1
        expected[1, 5] = 1
        np.testing.assert_array_equal(counts, expected)

    def test_signed_error_summary_preserves_bias_direction_and_outlier_rate(self):
        summary = signed_error_summary(np.asarray([10, 10, 10]), np.asarray([8, 10, 22]))
        self.assertAlmostEqual(summary["mean_signed_error"], 10 / 3)
        self.assertAlmostEqual(summary["under_fraction"], 1 / 3)
        self.assertAlmostEqual(summary["over_fraction"], 1 / 3)
        self.assertAlmostEqual(summary["absolute_error_gt_10_fraction"], 1 / 3)

    def test_fixed_mask_token_pooling_uses_touched_token_cells(self):
        tokens = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
        instances = np.zeros((64, 64), dtype=np.int32)
        instances[8:40, 20:48] = 1
        pooled = pool_instance_token(tokens, instances, 1)
        expected = tokens[:, 0:3, 1:3].mean(axis=(1, 2))
        np.testing.assert_allclose(pooled, expected)

    def test_curve_marker_uses_recorded_selection_score(self):
        rows = [
            {"val_R2": "0.8", "val_mPQ+": "0.2", "selection_score": "0.2"},
            {"val_R2": "0.7", "val_mPQ+": "0.3", "selection_score": "0.3"},
        ]
        self.assertIs(selected_row(rows, ["val_R2", "val_mPQ+"]), rows[1])

    def test_cache_signatures_detect_same_length_feature_changes(self):
        patches = np.asarray([1, 1, 2], dtype=np.int32)
        instances = np.asarray([1, 2, 1], dtype=np.int32)
        changed = np.asarray([1, 3, 1], dtype=np.int32)
        self.assertNotEqual(_feature_signature(patches, instances), _feature_signature(patches, changed))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "maps.npy"
            np.save(path, np.zeros((1, 2, 2), dtype=np.int32))
            before = _file_signature(path)
            np.save(path, np.zeros((2, 2, 2), dtype=np.int32))
            self.assertNotEqual(before, _file_signature(path))

    def test_perfect_pooled_pq(self):
        true = np.zeros((1, 24, 24, 2), dtype=np.int32)
        for cls in range(1, 7):
            y0 = 2 + ((cls - 1) // 3) * 10
            x0 = 2 + ((cls - 1) % 3) * 7
            true[0, y0:y0 + 4, x0:x0 + 4, 0] = cls
            true[0, y0:y0 + 4, x0:x0 + 4, 1] = cls
        result = multiclass_pq_plus(true, true)
        self.assertAlmostEqual(result["mPQ+"], 1.0, places=5)
        self.assertEqual(result["per_class"]["epithelial"]["tp"], 1)

    def test_r2_perfect(self):
        values = [1, 2, 4, 8]
        frame = pd.DataFrame({name: values for name in ["neutrophil", "epithelial", "lymphocyte", "plasma", "eosinophil", "connective"]})
        result = multiclass_r2(frame, frame)
        self.assertAlmostEqual(result["R2"], 1.0, places=5)

    def test_central_counts_include_border_crossing_instance(self):
        inst = np.zeros((256, 256), dtype=np.int32)
        cls = np.zeros((256, 256), dtype=np.uint8)
        inst[8:18, 20:30] = 1
        cls[8:18, 20:30] = 2
        inst[2:8, 40:50] = 2
        cls[2:8, 40:50] = 2
        counts = central_crop_counts(inst, cls)
        self.assertEqual(counts.tolist(), [0, 1, 0, 0, 0, 0])

    def test_fixed_mask_metrics_are_perfect_for_perfect_assignments(self):
        labels = np.concatenate([np.arange(1, 7), np.repeat(np.arange(1, 7), 2)]).astype(np.int8)
        patch_ids = np.concatenate([np.zeros(6), np.ones(12)]).astype(np.int32)
        metadata = pd.DataFrame({
            "patch_id": [0, 1],
            "split": ["test", "test"],
            **{f"count_{name}": [1, 2] for name in ["neutrophil", "epithelial", "lymphocyte", "plasma", "eosinophil", "connective"]},
        })
        result = evaluate_fixed_masks(
            assignments=labels,
            labels=labels,
            ious=np.full(len(labels), 0.8, dtype=np.float32),
            feature_patch_ids=patch_ids,
            central=np.ones(len(labels), dtype=bool),
            metadata=metadata,
            gt_full_counts=np.asarray([[1] * 6, [2] * 6]),
            split="test",
        )
        self.assertAlmostEqual(result["R2"], 1.0)
        self.assertAlmostEqual(result["mPQ+"], 0.8)

    def test_lora_is_identity_at_initialization(self):
        base = torch.nn.Linear(4, 3)
        wrapped = LoRALinear(base, rank=2, alpha=4)
        values = torch.randn(5, 4)
        self.assertTrue(torch.allclose(wrapped(values), base(values)))

    def test_instance_hv_map_points_away_from_center(self):
        instances = np.zeros((9, 9), dtype=np.int32)
        instances[2:7, 2:7] = 1
        hv = instance_hv_map(instances)
        self.assertLess(hv[0, 4, 2], 0)
        self.assertGreater(hv[0, 4, 6], 0)
        self.assertLess(hv[1, 2, 4], 0)
        self.assertGreater(hv[1, 6, 4], 0)

    def test_directional_maps_have_correct_diagonal_signs(self):
        instances = np.zeros((9, 9), dtype=np.int32)
        instances[1:8, 1:8] = 1
        maps = instance_directional_map(instances)
        self.assertEqual(maps.shape, (4, 9, 9))
        self.assertLess(maps[2, 2, 2], 0)
        self.assertGreater(maps[2, 6, 6], 0)
        self.assertLess(maps[3, 6, 2], 0)
        self.assertGreater(maps[3, 2, 6], 0)
        self.assertLessEqual(float(np.abs(maps).max()), 1.0)

    def test_expanding_hv_header_preserves_pretrained_hv_outputs(self):
        header = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity(), torch.nn.Conv2d(5, 2, 1))
        model = SimpleNamespace(
            hv_map_decoder=SimpleNamespace(decoder0_header=header),
            branches_output={"hv_map": 2},
        )
        features = torch.randn(2, 5, 7, 9)
        expected = header(features)
        expand_hv_head(model)
        actual = model.hv_map_decoder.decoder0_header(features)
        self.assertTrue(torch.equal(actual[:, :2], expected))
        self.assertEqual(actual.shape[1], 4)
        self.assertEqual(model.branches_output["hv_map"], 4)

    def test_fast_binary_pq_matches_reference(self):
        true = np.zeros((20, 20), dtype=np.int32)
        pred = np.zeros_like(true)
        true[2:8, 2:8] = 1
        true[10:17, 10:17] = 2
        pred[2:8, 2:8] = 1
        pred[11:18, 10:17] = 2
        pred[2:5, 13:16] = 3
        reference = pq_stats(true, pred)
        fast = fast_binary_pq_stats(true, pred)
        self.assertEqual(fast[:3], reference[:3])
        self.assertAlmostEqual(fast[3], reference[3])

    def test_vectorized_pq_stats_matches_legacy_greedy_reference(self):
        def legacy(true, pred, threshold):
            true_ids = [int(value) for value in np.unique(true) if value != 0]
            pred_ids = [int(value) for value in np.unique(pred) if value != 0]
            true_area = {value: int(np.sum(true == value)) for value in true_ids}
            pred_area = {value: int(np.sum(pred == value)) for value in pred_ids}
            candidates = []
            for gt in true_ids:
                for prediction in pred_ids:
                    intersection = int(np.sum((true == gt) & (pred == prediction)))
                    if not intersection:
                        continue
                    union = true_area[gt] + pred_area[prediction] - intersection
                    iou = intersection / union
                    if iou > threshold:
                        candidates.append((iou, gt, prediction))
            candidates.sort(reverse=True)
            matched_true, matched_pred = set(), set()
            sum_iou = 0.0
            for iou, gt, prediction in candidates:
                if gt not in matched_true and prediction not in matched_pred:
                    matched_true.add(gt)
                    matched_pred.add(prediction)
                    sum_iou += iou
            tp = len(matched_true)
            return tp, len(pred_ids) - tp, len(true_ids) - tp, sum_iou

        rng = np.random.default_rng(17)
        for _ in range(20):
            true = rng.choice([0, 2, 9, 31], size=(17, 19), p=[0.55, 0.15, 0.15, 0.15])
            pred = rng.choice([0, 4, 12, 77], size=(17, 19), p=[0.52, 0.16, 0.16, 0.16])
            for threshold in (0.3, 0.5, 0.7):
                expected = legacy(true, pred, threshold)
                actual = pq_stats(true, pred, threshold)
                self.assertEqual(actual[:3], expected[:3])
                self.assertAlmostEqual(actual[3], expected[3])

    def test_additive_patch_pq_statistics_reproduce_pooled_metric(self):
        truth = np.zeros((3, 12, 12, 2), dtype=np.int32)
        prediction = np.zeros_like(truth)
        truth[0, 1:5, 1:5] = (1, 1)
        prediction[0, 1:5, 1:5] = (9, 1)
        truth[1, 5:10, 5:10] = (3, 2)
        prediction[1, 6:11, 5:10] = (8, 2)
        truth[2, 2:7, 3:8] = (11, 6)
        prediction[2, 2:7, 3:8] = (4, 3)
        expected = multiclass_pq_plus(truth, prediction)
        actual = pq_summary_from_statistics(patch_pq_statistics(truth, prediction))
        for metric in ("mPQ+", "mDQ+", "mSQ+"):
            self.assertAlmostEqual(actual[metric], expected[metric])
        for name in CLASS_NAMES:
            for metric in ("pq", "dq", "sq", "sum_iou"):
                self.assertAlmostEqual(actual["per_class"][name][metric], expected["per_class"][name][metric])
            for metric in ("tp", "fp", "fn"):
                self.assertEqual(actual["per_class"][name][metric], expected["per_class"][name][metric])

    def test_fixed_geometry_pq_cache_reproduces_reference_typed_metric(self):
        truth = np.zeros((3, 12, 12, 2), dtype=np.int32)
        prediction = np.zeros_like(truth)
        truth[0, 1:5, 1:5] = (1, 1)
        prediction[0, 1:5, 1:5] = (9, 1)
        truth[1, 5:10, 5:10] = (3, 2)
        prediction[1, 6:11, 5:10] = (8, 2)
        truth[2, 2:7, 3:8] = (11, 6)
        prediction[2, 2:7, 3:8] = (4, 3)
        # Include a decoded-but-untyped dustbin instance.
        prediction[2, 8:10, 8:10, 0] = 7
        expected = multiclass_pq_plus(truth, prediction)
        cache = fixed_geometry_pq_cache(truth, prediction)
        statistics = fixed_geometry_patch_pq_statistics(cache, prediction)
        actual = pq_summary_from_statistics(statistics)
        for metric in ("mPQ+", "mDQ+", "mSQ+"):
            self.assertAlmostEqual(actual[metric], expected[metric])
        for name in CLASS_NAMES:
            for metric in ("pq", "dq", "sq", "sum_iou"):
                self.assertAlmostEqual(actual["per_class"][name][metric], expected["per_class"][name][metric])
            for metric in ("tp", "fp", "fn"):
                self.assertEqual(actual["per_class"][name][metric], expected["per_class"][name][metric])

        lookups = [decoded_class_lookup(patch) for patch in prediction]
        lookup_statistics = fixed_geometry_lookup_pq_statistics(cache, lookups)
        np.testing.assert_allclose(lookup_statistics, statistics)
        for patch, lookup in zip(prediction, lookups):
            np.testing.assert_array_equal(
                central_counts_from_lookup(patch[..., 0], lookup),
                central_crop_counts(patch[..., 0], patch[..., 1]),
            )

    def test_scaled_hv_decode_returns_original_resolution(self):
        foreground = np.zeros((32, 32), dtype=np.float32)
        foreground[8:24, 8:24] = 1.0
        instances = np.zeros((32, 32), dtype=np.int32)
        instances[8:24, 8:24] = 1
        hv = instance_hv_map(instances).transpose(1, 2, 0)
        decoded = decode_hv(
            foreground,
            hv,
            scale=2.0,
            binary_threshold=0.5,
            edge_threshold=0.5,
            object_size=1,
            ksize=3,
            opening_size=1,
            min_nucleus_size=1,
        )
        self.assertEqual(decoded.shape, foreground.shape)
        self.assertGreater(int(decoded.max()), 0)

    def test_four_direction_decode_accepts_directional_targets(self):
        foreground = np.zeros((32, 32), dtype=np.float32)
        foreground[7:25, 7:25] = 1.0
        instances = np.zeros((32, 32), dtype=np.int32)
        instances[7:25, 7:25] = 1
        maps = instance_directional_map(instances).transpose(1, 2, 0)
        decoded = decode_hv(
            foreground,
            maps,
            binary_threshold=0.5,
            edge_threshold=0.5,
            object_size=1,
            ksize=3,
            opening_size=1,
            min_nucleus_size=1,
        )
        self.assertEqual(decoded.shape, foreground.shape)
        self.assertGreater(int(decoded.max()), 0)

    def test_zero_directional_weight_matches_two_channel_decode(self):
        foreground = np.zeros((32, 32), dtype=np.float32)
        foreground[7:25, 7:25] = 1.0
        instances = np.zeros((32, 32), dtype=np.int32)
        instances[7:25, 7:25] = 1
        maps = instance_directional_map(instances).transpose(1, 2, 0)
        config = dict(binary_threshold=0.5, edge_threshold=0.5, object_size=1, ksize=3, opening_size=1, min_nucleus_size=1)
        expected = decode_hv(foreground, maps[..., :2], **config)
        actual = decode_hv(foreground, maps, directional_weight=0.0, **config)
        self.assertTrue(np.array_equal(actual, expected))

    def test_directional_weight_is_bounded(self):
        foreground = np.zeros((8, 8), dtype=np.float32)
        maps = np.zeros((8, 8, 4), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "directional_weight"):
            decode_hv(foreground, maps, directional_weight=1.1)

    def test_minority_patch_weights_upweight_rare_class_patch(self):
        counts = np.asarray([[100, 0], [100, 0], [0, 1]], dtype=np.int64)
        weights = minority_patch_weights(counts, blend=0.5)
        self.assertGreater(weights[2], weights[0])
        self.assertAlmostEqual(float(weights.mean()), 1.0)

    def test_color_blur_augmentation_preserves_image_contract(self):
        image = Image.fromarray(np.full((24, 24, 3), 128, dtype=np.uint8))
        torch.manual_seed(7)
        augmented = color_blur_augmentation(image, 0.1, 1.0, 0.3, 1.0)
        self.assertEqual(augmented.mode, "RGB")
        self.assertEqual(augmented.size, image.size)
        self.assertEqual(np.asarray(augmented).dtype, np.uint8)

    def test_hed_stain_augmentation_is_deterministic_and_preserves_contract(self):
        pixels = np.zeros((24, 24, 3), dtype=np.uint8)
        pixels[..., 0] = np.arange(24, dtype=np.uint8)[None] * 8
        pixels[..., 1] = 110
        pixels[..., 2] = 175
        image = Image.fromarray(pixels)
        torch.manual_seed(17)
        first = hed_stain_augmentation(image, probability=1.0)
        torch.manual_seed(17)
        second = hed_stain_augmentation(image, probability=1.0)
        self.assertEqual(first.mode, "RGB")
        self.assertEqual(first.size, image.size)
        self.assertTrue(np.array_equal(np.asarray(first), np.asarray(second)))
        self.assertFalse(np.array_equal(np.asarray(first), np.asarray(image)))
        values = np.asarray(first, dtype=np.int16)
        green_artifact = (values[..., 1] > values[..., 0] + 20) & (values[..., 1] > values[..., 2] + 20)
        self.assertLessEqual(float(green_artifact.mean()), 0.01)

    def test_source_patch_weights_equalize_aggregate_source_mass(self):
        sources = np.asarray(["large"] * 8 + ["small"] * 2)
        weights = source_patch_weights(sources, blend=1.0)
        self.assertAlmostEqual(float(weights[sources == "large"].sum()), float(weights[sources == "small"].sum()))
        self.assertAlmostEqual(float(weights.mean()), 1.0)

    def test_source_class_patch_weights_are_mean_normalized(self):
        sources = np.asarray(["a", "a", "a", "b"])
        counts = np.asarray([[10, 0], [8, 0], [9, 0], [0, 1]])
        weights = source_class_patch_weights(sources, counts)
        self.assertAlmostEqual(float(weights.mean()), 1.0)
        self.assertGreater(weights[-1], weights[0])

    def test_leave_source_out_filters_are_disjoint(self):
        metadata = pd.DataFrame(
            {
                "patch_id": [0, 1, 2, 3, 4],
                "split": ["train", "train", "train", "val", "val"],
                "source": ["glas", "crag", "glas", "glas", "crag"],
            }
        )
        train_ids, val_ids = select_train_validation_ids(metadata, {"glas"}, "train", {"glas"})
        self.assertEqual(train_ids.tolist(), [1])
        self.assertEqual(val_ids.tolist(), [0, 2])
        self.assertEqual(len(np.intersect1d(train_ids, val_ids)), 0)

    def test_paired_bpq_bootstrap_detects_identical_and_better_stats(self):
        reference = np.asarray([[5, 2, 3, 3.5], [4, 1, 2, 3.0]], dtype=np.float64)
        identical = paired_bootstrap_bpq(reference, reference)
        self.assertEqual(identical["candidate_minus_reference"], 0.0)
        self.assertEqual(identical["paired_patch_bootstrap_95_ci"], [0.0, 0.0])
        candidate = reference.copy()
        candidate[:, 0] += 2
        candidate[:, 2] -= 2
        candidate[:, 3] += 1.5
        better = paired_bootstrap_bpq(candidate, reference)
        self.assertGreater(better["candidate_minus_reference"], 0.0)
        self.assertGreater(better["paired_patch_bootstrap_95_ci"][0], 0.0)

    def test_equal_raw_map_ensemble_preserves_identical_member(self):
        truth = np.zeros((1, 32, 32), dtype=np.int32)
        truth[0, 7:25, 7:25] = 1
        foreground = (truth[0] > 0).astype(np.float32)
        hv = instance_hv_map(truth[0]).transpose(1, 2, 0)
        raw = np.concatenate((foreground[..., None], hv), axis=-1)[None]
        config = {
            "binary_threshold": 0.5,
            "edge_threshold": 0.5,
            "object_size": 1,
            "ksize": 3,
            "opening_size": 1,
            "min_nucleus_size": 1,
        }
        single, _ = evaluate_subset([raw, raw.copy()], (0,), truth, config, np.asarray(["source"]))
        ensemble, _ = evaluate_subset([raw, raw.copy()], (0, 1), truth, config, np.asarray(["source"]))
        self.assertEqual(single, ensemble)

    def test_raw_map_ensemble_preserves_unshared_directional_channels(self):
        truth = np.zeros((1, 32, 32), dtype=np.int32)
        truth[0, 7:25, 7:25] = 1
        foreground = (truth[0] > 0).astype(np.float32)
        directional = instance_directional_map(truth[0]).transpose(1, 2, 0)
        raw_four = np.concatenate((foreground[..., None], directional), axis=-1)[None]
        raw_two = raw_four[..., :3].copy()
        config = {
            "binary_threshold": 0.5,
            "edge_threshold": 0.5,
            "object_size": 1,
            "ksize": 3,
            "opening_size": 1,
            "min_nucleus_size": 1,
            "directional_weight": 0.5,
        }
        single, _ = evaluate_subset([raw_two, raw_four], (1,), truth, config, np.asarray(["source"]))
        ensemble, _ = evaluate_subset([raw_two, raw_four], (0, 1), truth, config, np.asarray(["source"]))
        self.assertEqual(single, ensemble)

    def test_hovernet_postprocessor_returns_empty_maps_for_empty_prediction(self):
        instances, classes = process_hovernet_prediction(
            np.zeros((256, 256), dtype=np.float32),
            np.zeros((256, 256, 2), dtype=np.float32),
            np.zeros((256, 256), dtype=np.int32),
        )
        self.assertEqual(instances.shape, (256, 256))
        self.assertEqual(classes.shape, (256, 256))
        self.assertEqual(int(instances.max()), 0)
        self.assertEqual(int(classes.max()), 0)

    def test_hovernet_instance_probabilities_are_conditional_and_aligned(self):
        instances = np.zeros((256, 256), dtype=np.int32)
        instances[24:96, 24:96] = 1
        instances[160:232, 160:232] = 2
        type_probs = np.zeros((256, 256, 7), dtype=np.float32)
        type_probs[..., 0] = 0.2
        type_probs[..., 1:] = 0.8 / 6.0
        type_probs[16:104, 16:104, 1] = 0.75
        type_probs[16:104, 16:104, 2:] = 0.01
        type_probs[152:240, 152:240, 4] = 0.75
        type_probs[152:240, 152:240, [1, 2, 3, 5, 6]] = 0.01
        instance_ids, probabilities = instance_class_probabilities(instances, type_probs)
        self.assertEqual(instance_ids.tolist(), [1, 2])
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6))
        self.assertEqual(probabilities.argmax(axis=1).tolist(), [0, 3])

    def test_official_hovernet_fold_is_group_disjoint(self):
        rows = []
        patch_id = 0
        for cohort in ("a", "b"):
            for group in range(10):
                for patch in range(2):
                    rows.append({"patch_id": patch_id, "patch_info": f"{cohort}_{group}-{patch:04d}"})
                    patch_id += 1
        metadata = pd.DataFrame(rows)
        train_ids, validation_ids = official_hovernet_fold(metadata)
        groups = metadata.set_index("patch_id").patch_info.str.split("-").str[0]
        self.assertFalse(set(groups.loc[train_ids]) & set(groups.loc[validation_ids]))
        self.assertEqual(len(train_ids) + len(validation_ids), len(metadata))

    def test_dashboard_refuses_to_silently_drop_subgroup_diagnostics(self):
        with self.assertRaisesRegex(RuntimeError, "silently omits"):
            require_complete_dashboard_summary({"subgroups": {}}, Path("missing-runs"))

    def test_dashboard_accepts_complete_subgroup_diagnostics(self):
        subgroup = {
            "by_class": [{"label": "epithelial"}],
            "by_institution": [{"label": "crag"}],
            "by_both": [{"label": "crag · epithelial"}],
            "confusions": [{"name": "baseline"}],
            "scatter_points": [{"gt": 1, "best_pred": 1}],
        }
        require_complete_dashboard_summary({"subgroups": subgroup}, Path("runs"))

    def test_dashboard_subgroup_default_names_the_recommended_single_model(self):
        name = build_subgroup_breakdown.__defaults__[-1]
        self.assertIn("rare-class-trained HoVer-Net", name)
        self.assertNotIn("debias", name.lower())

    def test_probability_ensemble_endpoint_round_trip(self):
        probabilities = np.asarray([[0.7, 0.3], [0.2, 0.8]], dtype=np.float32)
        recovered = softmax(calibrated_logits(probabilities, None))
        self.assertTrue(np.allclose(recovered, probabilities, atol=1e-7))

    def test_count_ensemble_endpoints_and_rounding(self):
        counts_a = np.asarray([[1, 4], [8, 2]], dtype=np.int32)
        counts_b = np.asarray([[3, 8], [2, 6]], dtype=np.int32)
        self.assertTrue(np.array_equal(blended_counts(counts_a, counts_b, np.ones(2)), counts_a))
        self.assertTrue(np.array_equal(blended_counts(counts_a, counts_b, np.zeros(2)), counts_b))
        expected_midpoint = np.asarray([[2, 6], [5, 4]], dtype=np.int32)
        self.assertTrue(np.array_equal(blended_counts(counts_a, counts_b, np.full(2, 0.5)), expected_midpoint))

    def test_robust_count_blend_preserves_endpoints_and_caps_additions_asymmetrically(self):
        anchor = np.asarray([0, 100], dtype=np.int32)
        auxiliary = np.asarray([100, 0], dtype=np.int32)
        self.assertTrue(
            np.array_equal(robust_blend(anchor, auxiliary, 1.0, 0.0, 0.0), anchor)
        )
        self.assertTrue(
            np.array_equal(
                robust_blend(anchor, auxiliary, 0.0, np.inf, np.inf), auxiliary
            )
        )
        prediction = robust_blend(anchor, auxiliary, 0.5, 1.0, np.inf)
        self.assertTrue(np.array_equal(prediction, np.asarray([1, 50], dtype=np.int32)))

    def test_robust_count_true_zero_gate_uses_two_patch_support_tolerance(self):
        truth = np.zeros(25, dtype=np.int32)
        anchor = np.zeros(25, dtype=np.int32)
        sources = np.asarray(["dpath"] * 25)
        two_outliers = np.zeros(25, dtype=np.int32)
        two_outliers[:2] = 11
        self.assertTrue(true_zero_tail_gate(truth, anchor, two_outliers, sources)["passes"])
        three_outliers = two_outliers.copy()
        three_outliers[2] = 11
        audit = true_zero_tail_gate(truth, anchor, three_outliers, sources)
        self.assertFalse(audit["passes"])
        self.assertEqual(audit["violations"][0]["threshold"], 10)

    def test_rotation_tta_exactly_inverts_hv_vector_axes(self):
        y, x = torch.meshgrid(torch.linspace(-1, 1, 11), torch.linspace(-1, 1, 15), indexing="ij")
        original_hv = torch.stack((x, y))[None]
        original_foreground = ((x.square() + y.square()) < 0.7)[None]
        for k in (1, 2, 3):
            spatial_hv = torch.rot90(original_hv, k, dims=(-2, -1))
            if k == 1:
                rotated_hv = torch.stack((spatial_hv[:, 1], -spatial_hv[:, 0]), dim=1)
            elif k == 2:
                rotated_hv = -spatial_hv
            else:
                rotated_hv = torch.stack((-spatial_hv[:, 1], spatial_hv[:, 0]), dim=1)
            rotated_foreground = torch.rot90(original_foreground, k, dims=(-2, -1))
            self.assertTrue(torch.allclose(invert_hv_rotation(rotated_hv, k), original_hv, atol=1e-6))
            self.assertTrue(torch.equal(invert_spatial_rotation(rotated_foreground, k), original_foreground))

    def test_tta_exactly_inverts_four_direction_maps(self):
        y, x = torch.meshgrid(torch.linspace(-1, 1, 11), torch.linspace(-1, 1, 15), indexing="ij")
        original = torch.stack((x, y, (x + y) / np.sqrt(2.0), (x - y) / np.sqrt(2.0)))[None]
        horizontal_input = torch.stack(
            (-original[:, 0], original[:, 1], -original[:, 3], -original[:, 2]), dim=1
        ).flip(-1)
        vertical_input = torch.stack(
            (original[:, 0], -original[:, 1], original[:, 3], original[:, 2]), dim=1
        ).flip(-2)
        self.assertTrue(torch.allclose(invert_hv_horizontal_flip(horizontal_input), original, atol=1e-6))
        self.assertTrue(torch.allclose(invert_hv_vertical_flip(vertical_input), original, atol=1e-6))
        for k in (1, 2, 3):
            spatial = torch.rot90(original, k, dims=(-2, -1))
            if k == 1:
                rotated = torch.stack((spatial[:, 1], -spatial[:, 0], -spatial[:, 3], spatial[:, 2]), dim=1)
            elif k == 2:
                rotated = -spatial
            else:
                rotated = torch.stack((-spatial[:, 1], spatial[:, 0], spatial[:, 3], -spatial[:, 2]), dim=1)
            self.assertTrue(torch.allclose(invert_hv_rotation(rotated, k), original, atol=1e-6))

    def test_visual_panel_places_count_chart_above_raw_and_overlaid_images(self):
        image = np.full((40, 40, 3), [17, 91, 203], dtype=np.uint8)
        empty_instance = np.zeros((40, 40), dtype=np.int32)
        empty_class = np.zeros((40, 40), dtype=np.uint8)
        panel = render_panel(
            image,
            empty_instance,
            empty_class,
            empty_instance,
            empty_class,
            np.zeros(6, dtype=np.int64),
            np.zeros(6, dtype=np.int64),
            "layout test",
        )
        header_height = 31
        raw_bottom_pixel = panel[header_height + image.shape[0] + 30, 20]
        chart_top_pixel = panel[header_height + 30, 20]
        self.assertTrue(np.array_equal(raw_bottom_pixel, image[30, 20]))
        self.assertFalse(np.array_equal(chart_top_pixel, image[30, 20]))

    def test_gt_cutouts_omit_empty_classes_and_explain_high_confidence_geometry_error(self):
        image = np.full((32, 32, 3), 180, dtype=np.uint8)
        true_instance = np.zeros((32, 32), dtype=np.int32)
        true_class = np.zeros((32, 32), dtype=np.uint8)
        true_instance[8:20, 8:20] = 3
        true_class[8:20, 8:20] = 2
        pred_instance = np.zeros((32, 32), dtype=np.int32)
        pred_class = np.zeros((32, 32), dtype=np.uint8)
        pred_instance[10:15, 10:15] = 7
        pred_class[10:15, 10:15] = 2
        probabilities = {7: np.asarray([0.01, 0.9, 0.02, 0.02, 0.02, 0.03], dtype=np.float32)}
        with tempfile.TemporaryDirectory() as directory:
            path, records = render_cutout_strip(
                image,
                true_instance,
                true_class,
                pred_instance,
                pred_class,
                probabilities,
                patch_id=4,
                outdir=Path(directory),
                max_per_class=2,
            )
            rendered = Image.open(path)
            self.assertEqual(rendered.height, 7 + 128 + 34 + 7)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["gt_class"], "epithelial")
        self.assertEqual(records[0]["status"], "low IoU")
        self.assertGreater(records[0]["probability_gt_class"], 0.5)
        self.assertFalse(records[0]["correct"])

    def test_final_stack_type_blend_gate_requires_broad_safe_gain(self):
        baseline_source = {metric: 0.4 for metric in ("mPQ+", "mDQ+", "mSQ+")}
        selected_source = {metric: 0.405 for metric in ("mPQ+", "mDQ+", "mSQ+")}
        report = {
            "selected_delta_vs_control": {"mPQ+": 0.01, "mDQ+": 0.012, "mSQ+": -0.002},
            "weight_stable_across_source_exclusions": True,
            "selected": {"by_source": {"dpath": selected_source}},
            "candidates": [
                {"candidate_type_weight": 0, "by_source": {"dpath": baseline_source}},
            ],
        }
        self.assertTrue(type_blend_eligible(report))
        report["selected"]["by_source"]["dpath"]["mDQ+"] = 0.38
        self.assertFalse(type_blend_eligible(report))

    def test_final_stack_sampler_prior_gate_requires_out_of_source_transfer(self):
        report = {
            "selected": {
                "mPQ+": {
                    "delta_vs_raw": {"mPQ+": 0.01},
                    "delta_vs_pooled_strength_0": {"mPQ+": 0.005},
                }
            },
            "leave_one_source_out": {
                "mPQ+": {"delta_vs_raw": {"mPQ+": 0.003}, "stable_within_one_grid_step": True}
            },
        }
        self.assertTrue(prior_eligible(report))
        report["leave_one_source_out"]["mPQ+"]["delta_vs_raw"]["mPQ+"] = -0.001
        self.assertFalse(prior_eligible(report))


if __name__ == "__main__":
    unittest.main()
