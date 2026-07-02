"""Metric registry for MVP extraction and availability reports."""

PHYSICS_METRICS = ["final_squared_flux", "mean_abs_Bn", "max_abs_Bn", "rms_Bn", "final_objective"]
GEOMETRY_METRICS = [
    "total_coil_length",
    "mean_coil_length",
    "max_coil_length",
    "max_curvature",
    "p95_curvature",
    "mean_curvature",
    "curvature_integral",
    "max_torsion",
    "p95_torsion",
    "mean_abs_torsion",
    "min_coil_coil_distance",
    "min_coil_plasma_distance",
    "self_intersection_flag",
    "kink_flag",
]
NUMERICAL_METRICS = ["runtime_seconds", "iterations_used", "optimizer_success", "optimizer_message", "gradient_norm", "final_step_norm"]
DIAGNOSTIC_METRICS = ["missing_output_count", "artifact_count", "failure_reason", "warning_count"]

METRIC_TYPES = {
    **{name: "physics" for name in PHYSICS_METRICS},
    **{name: "geometry" for name in GEOMETRY_METRICS},
    **{name: "numerical" for name in NUMERICAL_METRICS},
    **{name: "diagnostic" for name in DIAGNOSTIC_METRICS},
}
ALL_METRICS = list(METRIC_TYPES)
