from __future__ import annotations

from pathlib import Path

from scripts import run_simsopt_batch


def test_policy_targets_override_default_engineering_targets() -> None:
    targets = run_simsopt_batch._policy_targets(
        {
            "targets": {
                "avg_BdotN_over_B": 0.03,
                "final_min_cc_separation": 0.16,
            },
            "common": {
                "targets": {
                    "avg_BdotN_over_B": 0.02,
                    "final_min_cc_separation": 0.18,
                    "final_min_cs_separation": 0.18,
                    "final_max_curvature": 5.0,
                }
            }
        }
    )

    assert targets["avg_BdotN_over_B"] == 0.02
    assert targets["final_min_cc_separation"] == 0.18
    assert targets["final_min_cs_separation"] == 0.18
    assert targets["final_max_curvature"] == 5.0
    assert targets["final_max_torsion"] == run_simsopt_batch.TARGETS["final_max_torsion"]


def test_policy_targets_accept_top_level_targets() -> None:
    targets = run_simsopt_batch._policy_targets(
        {
            "targets": {
                "avg_BdotN_over_B": 0.02,
                "final_min_cc_separation": 0.4,
                "final_min_cs_separation": 0.35,
            }
        }
    )

    assert targets["avg_BdotN_over_B"] == 0.02
    assert targets["final_min_cc_separation"] == 0.4
    assert targets["final_min_cs_separation"] == 0.35


def test_record_uses_job_targets_for_meets_targets() -> None:
    targets = {
        "avg_BdotN_over_B": 0.02,
        "final_min_cc_separation": 0.18,
        "final_min_cs_separation": 0.18,
        "final_max_curvature": 5.0,
        "final_max_torsion": 15.0,
    }
    record = run_simsopt_batch._record_from_metrics(
        {
            "run_id": "target_override_probe",
            "backend": "simsopt",
            "targets": targets,
            "case": {
                "optimizer_params": {"algorithm": "L-BFGS-B", "max_iterations": 1},
                "coils_params": {"order": 6},
                "experiment_metadata": {},
                "random_seed": 1,
            },
        },
        Path("unused"),
        {
            "avg_BdotN_over_B": 0.019,
            "final_min_cc_separation": 0.19,
            "final_min_cs_separation": 0.19,
            "final_max_curvature": 4.9,
            "final_max_torsion": 12.0,
        },
    )

    assert record["target_final_min_cc_separation"] == 0.18
    assert record["target_final_min_cs_separation"] == 0.18
    assert record["meets_targets"] is True


def test_target_b_normalized_metric_can_drive_meets_targets() -> None:
    targets = {
        "avg_BdotN_over_B": 0.02,
        "avg_BdotN_over_target_B": 0.015,
        "final_min_cc_separation": 0.18,
        "final_min_cs_separation": 0.18,
        "final_max_curvature": 5.0,
        "final_max_torsion": 15.0,
    }
    record = run_simsopt_batch._record_from_metrics(
        {
            "run_id": "target_b_normalized_probe",
            "backend": "simsopt",
            "targets": targets,
            "case": {
                "optimizer_params": {"algorithm": "L-BFGS-B", "max_iterations": 1},
                "coils_params": {"order": 6},
                "experiment_metadata": {},
                "random_seed": 1,
            },
        },
        Path("unused"),
        {
            "avg_BdotN_over_B": 0.019,
            "avg_BdotN_over_target_B": 0.016,
            "max_BdotN_over_target_B": 0.05,
            "final_min_cc_separation": 0.19,
            "final_min_cs_separation": 0.19,
            "final_max_curvature": 4.9,
            "final_max_torsion": 12.0,
        },
    )

    assert record["avg_BdotN_over_target_B"] == 0.016
    assert record["max_BdotN_over_target_B"] == 0.05
    assert record["target_avg_BdotN_over_target_B"] == 0.015
    assert record["meets_targets"] is False


def test_base_case_preserves_device_threshold_overrides() -> None:
    case = run_simsopt_batch._base_case(
        "plasma_surfaces/wout_eq_OP_ball3_033_optimized.nc",
        order=6,
        max_iterations=300,
        policy={
            "common": {
                "thresholds": {
                    "length_threshold_device": 6.0,
                    "cc_threshold_device": 0.25,
                    "cs_threshold_device": 0.25,
                    "curvature_threshold_device": 3.0,
                    "torsion_threshold_device": 15.0,
                    "msc_threshold_device": 8.0,
                    "arclength_variation_threshold_device": 0.5,
                    "length_variance_threshold_device": 0.5,
                }
            }
        },
    )

    terms = case["coil_objective_terms"]
    assert terms["length_threshold_device"] == 6.0
    assert terms["cc_threshold_device"] == 0.25
    assert terms["cs_threshold_device"] == 0.25
    assert terms["curvature_threshold_device"] == 3.0
    assert terms["length_variance_threshold_device"] == 0.5


def test_record_preserves_physics_context_from_case() -> None:
    targets = {
        "avg_BdotN_over_B": 0.02,
        "avg_BdotN_over_target_B": 0.05,
        "final_min_cc_separation": 0.18,
        "final_min_cs_separation": 0.18,
        "final_max_curvature": 5.0,
        "final_max_torsion": 15.0,
    }
    record = run_simsopt_batch._record_from_metrics(
        {
            "run_id": "physics_context_probe",
            "backend": "simsopt",
            "targets": targets,
            "case": {
                "surface_params": {"target_B": 1.25, "virtual_casing": True},
                "optimizer_params": {"algorithm": "L-BFGS-B", "max_iterations": 1},
                "coils_params": {"order": 6, "current_weights": [1.0, 1.0, 1.0]},
                "experiment_metadata": {},
                "random_seed": 1,
            },
        },
        Path("unused"),
        {
            "avg_BdotN_over_B": 0.019,
            "avg_BdotN_over_target_B": 0.016,
            "final_min_cc_separation": 0.19,
            "final_min_cs_separation": 0.19,
            "final_max_curvature": 4.9,
            "final_max_torsion": 12.0,
        },
    )

    assert record["target_B_field"] == 1.25
    assert record["virtual_casing"] is True
    assert record["requested_current_weights"] == [1.0, 1.0, 1.0]


def test_failure_record_preserves_job_targets() -> None:
    targets = {
        "avg_BdotN_over_B": 0.02,
        "final_min_cc_separation": 0.18,
        "final_min_cs_separation": 0.18,
        "final_max_curvature": 5.0,
        "final_max_torsion": 15.0,
    }
    record = run_simsopt_batch._failure_record(
        {
            "run_id": "target_override_failure",
            "backend": "simsopt",
            "targets": targets,
            "case": {
                "optimizer_params": {"algorithm": "L-BFGS-B", "max_iterations": 1},
                "coils_params": {"order": 6},
                "experiment_metadata": {},
                "random_seed": 1,
            },
        },
        Path("unused"),
        RuntimeError("probe"),
    )

    assert record["target_avg_BdotN_over_B"] == 0.02
    assert record["target_final_min_cc_separation"] == 0.18
    assert record["meets_targets"] is False
