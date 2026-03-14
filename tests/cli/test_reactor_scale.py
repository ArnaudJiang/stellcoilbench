"""Tests for reactor scale metrics computation."""

import pytest

from stellcoilbench.cli import _compute_reactor_scale_metrics
from stellcoilbench.config_scheme import ARIES_CS_MINOR_RADIUS, REACTOR_REFERENCE

from tests.cli.conftest import _make_metrics


def test_reactor_scale_returns_error_when_params_missing():
    result = _compute_reactor_scale_metrics({})
    assert "error" in result


def test_reactor_scale_already_reactor_scaled_skips_scaling():
    """When already_reactor_scaled=True, raw metrics are used as reactor-scale (L_scale=1, B_scale=1)."""
    metrics = _make_metrics(
        minor_radius=0.2,
        target_B=1.0,
        final_min_cc_separation=0.5,
        final_min_cs_separation=1.5,
        final_total_length=200.0,
        final_max_curvature=0.5,
    )
    result = _compute_reactor_scale_metrics(metrics, already_reactor_scaled=True)
    assert result["scaling_factors"]["length_scale"] == 1.0
    assert result["scaling_factors"]["B_field_scale"] == 1.0
    assert result["scaling_factors"].get("already_reactor_scaled") is True
    assert result["reactor_scale_min_cc_separation"] == 0.5
    assert result["reactor_scale_min_cs_separation"] == 1.5
    assert result["reactor_scale_total_length"] == 200.0
    assert result["reactor_scale_max_curvature"] == 0.5


_REACTOR_SCALE_SCALING_PARAMS = [
    # (minor_radius, target_B, metric_overrides, expected_checks)
    (
        0.2,
        1.0,
        {
            "final_min_cc_separation": 0.1,
            "final_min_cs_separation": 0.2,
            "final_total_length": 5.0,
        },
        lambda L, B: {
            "reactor_scale_min_cc_separation": 0.1 * L,
            "reactor_scale_min_cs_separation": 0.2 * L,
            "reactor_scale_total_length": 5.0 * L,
        },
    ),
    (
        0.34,
        2.0,
        {
            "final_max_curvature": 3.0,
            "final_average_curvature": 1.5,
            "final_mean_squared_curvature": 4.0,
        },
        lambda L, B: {
            "reactor_scale_max_curvature": 3.0 / L,
            "reactor_scale_average_curvature": 1.5 / L,
            "reactor_scale_mean_squared_curvature": 4.0 / L**2,
        },
    ),
    (
        0.2,
        1.0,
        {"final_max_max_coil_force": 1e5, "final_avg_max_coil_force": 5e4},
        lambda L, B: {
            "reactor_scale_max_max_coil_force": 1e5 * (B**2 * L / 1e6),
            "reactor_scale_avg_max_coil_force": 5e4 * (B**2 * L / 1e6),
        },
    ),
    (
        0.2,
        1.0,
        {"final_max_max_coil_torque": 1e6, "final_avg_max_coil_torque": 5e5},
        lambda L, B: {
            "reactor_scale_max_max_coil_torque": 1e6 * (B**2 * L**2 / 1e6),
            "reactor_scale_avg_max_coil_torque": 5e5 * (B**2 * L**2 / 1e6),
        },
    ),
]


@pytest.mark.parametrize(
    "minor_radius,target_B,metric_overrides,expected_fn",
    _REACTOR_SCALE_SCALING_PARAMS,
    ids=["length", "curvature", "force", "torque"],
)
def test_reactor_scale_scaling(minor_radius, target_B, metric_overrides, expected_fn):
    """Lengths scale as L, curvature as 1/L, force as B²L/1e6, torque as B²L²/1e6."""
    L_scale = ARIES_CS_MINOR_RADIUS / minor_radius
    B_scale = REACTOR_REFERENCE["B_field"] / target_B
    metrics = _make_metrics(
        minor_radius=minor_radius, target_B=target_B, **metric_overrides
    )
    result = _compute_reactor_scale_metrics(metrics)
    expected = expected_fn(L_scale, B_scale)
    for key, exp_val in expected.items():
        assert result[key] == pytest.approx(exp_val), (
            f"{key}: got {result[key]}, expected {exp_val}"
        )


def test_reactor_scale_n_turns_per_coil():
    """N_turns_per_coil[i] = max(N_force_i, N_jc_i) per unique coil."""
    minor_radius = 0.2
    target_B = 1.0
    L_scale = ARIES_CS_MINOR_RADIUS / minor_radius
    B_scale = REACTOR_REFERENCE["B_field"] / target_B
    force_scale_raw = B_scale**2 * L_scale

    desired_MN = [2.3, 0.4, 0.7]
    device_forces = [d * 1e6 / force_scale_raw for d in desired_MN]

    metrics = _make_metrics(
        minor_radius=minor_radius,
        target_B=target_B,
        final_max_max_coil_force=max(device_forces),
        final_max_force_per_coil=device_forces,
    )
    result = _compute_reactor_scale_metrics(metrics)

    for i, d in enumerate(desired_MN):
        assert result["reactor_scale_force_per_coil_MN_per_m"][i] == pytest.approx(d)

    assert result["N_turns_force"] == [5, 1, 2]
    assert "N_turns_jc" in result
    assert len(result["N_turns_jc"]) == 3
    for i in range(3):
        assert result["N_turns_per_coil"][i] == max(
            result["N_turns_force"][i], result["N_turns_jc"][i]
        )
    assert result["force_limit_MN_per_m"] == 0.5
    assert "jc_model" in result
    assert "NI_reactor" in result["jc_model"]
    assert "I_turn" in result["jc_model"]
    assert "B_peak_estimate" in result["jc_model"]


def test_reactor_scale_n_turns_minimum_one():
    """N_turns_per_coil is at least 1 even when force is below the limit."""
    metrics = _make_metrics(
        minor_radius=0.2,
        target_B=1.0,
        final_max_max_coil_force=1e-3,
        final_max_force_per_coil=[1e-3, 1e-4],
    )
    result = _compute_reactor_scale_metrics(metrics)
    assert result["N_turns_per_coil"] == [1, 1]


def test_reactor_scale_n_turns_from_currents_only():
    """N_turns finite-build runs when only currents/lengths present (e.g. Zenodo coils)."""
    metrics = _make_metrics(
        minor_radius=0.2,
        target_B=1.0,
        final_total_length=40.0,
        final_current_per_coil=[1e5, 1e5, 1e5, 1e5],
        final_length_per_coil=[10.0, 10.0, 10.0, 10.0],
    )
    result = _compute_reactor_scale_metrics(metrics)
    assert "N_turns_per_coil" in result
    assert "N_turns_jc" in result
    assert "total_superconductor_length_km" in result
    assert result["N_turns_force"] == [1, 1, 1, 1]
    assert all(n >= 1 for n in result["N_turns_per_coil"])


def test_per_turn_max_force():
    """per_turn_max_force = max_i(reactor_force_i / N_turns_i)."""
    minor_radius = 0.2
    target_B = 1.0
    L_scale = ARIES_CS_MINOR_RADIUS / minor_radius
    B_scale = REACTOR_REFERENCE["B_field"] / target_B
    force_scale_raw = B_scale**2 * L_scale

    desired_MN = [2.5, 0.3]
    device_forces = [d * 1e6 / force_scale_raw for d in desired_MN]

    metrics = _make_metrics(
        minor_radius=minor_radius,
        target_B=target_B,
        final_max_max_coil_force=max(device_forces),
        final_max_force_per_coil=device_forces,
    )
    result = _compute_reactor_scale_metrics(metrics)

    n_turns = result["N_turns_per_coil"]
    rs_forces = result["reactor_scale_force_per_coil_MN_per_m"]
    expected_per_turn = [f / n for f, n in zip(rs_forces, n_turns)]
    assert result["per_turn_max_force"] == pytest.approx(max(expected_per_turn))


def test_per_turn_max_torque_with_per_coil():
    """per_turn_max_torque uses per-coil torque when available."""
    minor_radius = 0.2
    target_B = 1.0
    L_scale = ARIES_CS_MINOR_RADIUS / minor_radius
    B_scale = REACTOR_REFERENCE["B_field"] / target_B
    force_scale_raw = B_scale**2 * L_scale
    torque_scale_raw = B_scale**2 * L_scale**2

    desired_MN_force = [1.0, 1.0]
    device_forces = [d * 1e6 / force_scale_raw for d in desired_MN_force]
    device_torques = [1e4, 2e4]

    metrics = _make_metrics(
        minor_radius=minor_radius,
        target_B=target_B,
        final_max_max_coil_force=max(device_forces),
        final_max_force_per_coil=device_forces,
        final_max_max_coil_torque=max(device_torques),
        final_max_torque_per_coil=device_torques,
    )
    result = _compute_reactor_scale_metrics(metrics)

    n_turns = result["N_turns_per_coil"]
    reactor_torques = [t * torque_scale_raw / 1e6 for t in device_torques]
    expected_per_turn = [t / n for t, n in zip(reactor_torques, n_turns)]
    assert result["per_turn_max_torque"] == pytest.approx(max(expected_per_turn))


def test_per_turn_max_torque_fallback():
    """Without per-coil torque, falls back to max_torque / min(N_turns)."""
    minor_radius = 0.2
    target_B = 1.0
    L_scale = ARIES_CS_MINOR_RADIUS / minor_radius
    B_scale = REACTOR_REFERENCE["B_field"] / target_B
    force_scale_raw = B_scale**2 * L_scale

    desired_MN_force = [1.0, 2.0]
    device_forces = [d * 1e6 / force_scale_raw for d in desired_MN_force]

    metrics = _make_metrics(
        minor_radius=minor_radius,
        target_B=target_B,
        final_max_max_coil_force=max(device_forces),
        final_max_force_per_coil=device_forces,
        final_max_max_coil_torque=5e4,
    )
    result = _compute_reactor_scale_metrics(metrics)

    max_tau_rs = result["reactor_scale_max_max_coil_torque"]
    min_n = min(result["N_turns_per_coil"])
    assert result["per_turn_max_torque"] == pytest.approx(max_tau_rs / min_n)


def test_total_superconductor_length():
    """Total SC length = Σ N_turns_i * reactor_scale_length_i, in km."""
    minor_radius = 0.2
    target_B = 1.0
    L_scale = ARIES_CS_MINOR_RADIUS / minor_radius
    B_scale = REACTOR_REFERENCE["B_field"] / target_B
    force_scale_raw = B_scale**2 * L_scale

    desired_MN = [2.3, 0.4]
    device_forces = [d * 1e6 / force_scale_raw for d in desired_MN]
    device_lengths = [3.0, 4.0]

    metrics = _make_metrics(
        minor_radius=minor_radius,
        target_B=target_B,
        final_max_max_coil_force=max(device_forces),
        final_max_force_per_coil=device_forces,
        final_length_per_coil=device_lengths,
        final_total_length=sum(device_lengths),
    )
    result = _compute_reactor_scale_metrics(metrics)

    n_turns = result["N_turns_per_coil"]
    expected_km = sum(n * ln * L_scale for n, ln in zip(n_turns, device_lengths)) / 1e3
    assert result["total_superconductor_length_km"] == pytest.approx(expected_km)


def test_total_superconductor_length_fallback():
    """Without per-coil lengths, use uniform-length fallback."""
    minor_radius = 0.2
    target_B = 1.0
    L_scale = ARIES_CS_MINOR_RADIUS / minor_radius

    metrics = _make_metrics(
        minor_radius=minor_radius,
        target_B=target_B,
        final_max_max_coil_force=1e-3,
        final_max_force_per_coil=[1e-3, 1e-3],
        final_total_length=10.0,
    )
    result = _compute_reactor_scale_metrics(metrics)
    n_turns = result["N_turns_per_coil"]
    expected_km = sum(n * 5.0 * L_scale for n in n_turns) / 1e3
    assert result["total_superconductor_length_km"] == pytest.approx(expected_km)


def test_reactor_scale_squared_flux_scaling():
    """SquaredFlux [T²m²] = ½∫(B·n̂)²dS scales as B²·L² (NOT B²·L⁴)."""
    minor_radius = 0.453
    target_B = 3.0
    L_scale = ARIES_CS_MINOR_RADIUS / minor_radius
    B_scale = REACTOR_REFERENCE["B_field"] / target_B
    flux_scale = B_scale**2 * L_scale**2

    metrics = _make_metrics(
        minor_radius=minor_radius,
        target_B=target_B,
        final_squared_flux=1e-4,
    )
    result = _compute_reactor_scale_metrics(metrics)
    assert result["reactor_scale_squared_flux"] == pytest.approx(1e-4 * flux_scale)


def test_reactor_scale_arclength_variation_scaling():
    """ArclengthVariation [m²] (variance of arclengths) scales as L²."""
    minor_radius = 0.2
    target_B = 1.0
    L_scale = ARIES_CS_MINOR_RADIUS / minor_radius

    metrics = _make_metrics(
        minor_radius=minor_radius,
        target_B=target_B,
        final_arclength_variation=0.01,
    )
    result = _compute_reactor_scale_metrics(metrics)
    assert result["reactor_scale_arclength_variation"] == pytest.approx(
        0.01 * L_scale**2
    )


def test_n_turns_jc_with_currents():
    """N_turns_jc should increase for coils requiring more ampere-turns."""
    from stellcoilbench.reactor_scale import compute_N_turns_critical_current

    minor_radius = 0.2
    L_scale = ARIES_CS_MINOR_RADIUS / minor_radius
    result = compute_N_turns_critical_current(
        per_coil_forces=[1e5, 1e5],
        per_coil_currents=[1e4, 1e5],
        per_coil_lengths=[5.0, 5.0],
        L_scale=L_scale,
        B_scale=5.7,
        target_B=1.0,
    )
    n = result["N_turns_jc"]
    assert n[1] >= 5 * n[0]
    assert all(isinstance(x, int) and x >= 1 for x in n)
    assert all(0 < it <= 50e3 for it in result["I_turn"])


def test_n_turns_jc_no_currents_fallback():
    """Without per-coil currents, use force-based current estimate."""
    from stellcoilbench.reactor_scale import compute_N_turns_critical_current

    minor_radius = 0.2
    L_scale = ARIES_CS_MINOR_RADIUS / minor_radius
    result = compute_N_turns_critical_current(
        per_coil_forces=[1e5, 5e5],
        per_coil_currents=None,
        per_coil_lengths=None,
        L_scale=L_scale,
        B_scale=5.7,
        target_B=1.0,
    )
    n = result["N_turns_jc"]
    assert n[1] >= 3 * n[0]
    assert all(isinstance(x, int) and x >= 1 for x in n)


def test_n_turns_max_of_force_and_jc():
    """N_turns_per_coil should be element-wise max(N_force, N_jc)."""
    minor_radius = 0.2
    target_B = 1.0

    metrics = _make_metrics(
        minor_radius=minor_radius,
        target_B=target_B,
        final_max_max_coil_force=1e7,
        final_max_force_per_coil=[1e7, 1e3],
    )
    result = _compute_reactor_scale_metrics(metrics)
    for i in range(2):
        assert result["N_turns_per_coil"][i] >= result["N_turns_force"][i]
        assert result["N_turns_per_coil"][i] >= result["N_turns_jc"][i]
        assert result["N_turns_per_coil"][i] == max(
            result["N_turns_force"][i], result["N_turns_jc"][i]
        )


def test_winding_pack_width_formula():
    """Winding-pack side length: w = sqrt(N_turns) * 20 mm."""
    import numpy as np

    from stellcoilbench.reactor_scale import STELLARIS_A_TURN

    turn_side = np.sqrt(STELLARIS_A_TURN)
    assert turn_side == pytest.approx(0.020)
    assert np.sqrt(324) * turn_side == pytest.approx(0.360)
    assert np.sqrt(225) * turn_side == pytest.approx(0.300)


def test_winding_pack_in_reactor_metrics():
    """_compute_reactor_scale_metrics stores winding-pack width per coil."""
    import numpy as np

    from stellcoilbench.reactor_scale import STELLARIS_A_TURN

    minor_radius = 0.2
    target_B = 1.0
    L_scale = ARIES_CS_MINOR_RADIUS / minor_radius
    B_scale = REACTOR_REFERENCE["B_field"] / target_B
    force_scale_raw = B_scale**2 * L_scale

    desired_MN = [2.5, 0.3]
    device_forces = [d * 1e6 / force_scale_raw for d in desired_MN]

    metrics = _make_metrics(
        minor_radius=minor_radius,
        target_B=target_B,
        final_max_max_coil_force=max(device_forces),
        final_max_force_per_coil=device_forces,
    )
    result = _compute_reactor_scale_metrics(metrics)

    n_turns = result["N_turns_per_coil"]
    wp_widths = result["winding_pack_width_per_coil"]
    assert len(wp_widths) == len(n_turns)

    turn_side = np.sqrt(STELLARIS_A_TURN)
    for i, (n, w) in enumerate(zip(n_turns, wp_widths)):
        expected = float(np.sqrt(n) * turn_side)
        assert w == pytest.approx(expected), f"Coil {i}: expected {expected}, got {w}"
    assert result["max_winding_pack_width"] == pytest.approx(max(wp_widths))


def test_winding_pack_single_turn():
    """A coil with N_turns=1 has winding-pack width = 20 mm."""
    import numpy as np

    from stellcoilbench.reactor_scale import STELLARIS_A_TURN

    metrics = _make_metrics(
        minor_radius=0.2,
        target_B=1.0,
        final_max_max_coil_force=1e-3,
        final_max_force_per_coil=[1e-3],
    )
    result = _compute_reactor_scale_metrics(metrics)
    assert all(n >= 1 for n in result["N_turns_per_coil"])
    n = result["N_turns_per_coil"][0]
    expected_w = float(np.sqrt(n) * np.sqrt(STELLARIS_A_TURN))
    assert result["winding_pack_width_per_coil"][0] == pytest.approx(expected_w)
    assert result["max_winding_pack_width"] == pytest.approx(expected_w)


def test_finite_build_cc_clearance_positive():
    """Clearance = d_cc_min - w_max; positive when coils don't overlap."""
    minor_radius = 0.2
    target_B = 1.0
    L_scale = ARIES_CS_MINOR_RADIUS / minor_radius

    metrics = _make_metrics(
        minor_radius=minor_radius,
        target_B=target_B,
        final_min_cc_separation=0.2,
        final_max_max_coil_force=1e-3,
        final_max_force_per_coil=[1e-3, 1e-3],
    )
    result = _compute_reactor_scale_metrics(metrics)

    d_cc = result["reactor_scale_min_cc_separation"]
    w_max = result["max_winding_pack_width"]
    assert d_cc == pytest.approx(0.2 * L_scale)
    assert w_max < d_cc
    assert result["finite_build_cc_clearance"] == pytest.approx(d_cc - w_max)
    assert result["finite_build_cc_clearance"] > 0


def test_finite_build_cc_clearance_negative():
    """Clearance is negative when the winding pack exceeds the gap."""
    minor_radius = 0.2
    target_B = 1.0
    L_scale = ARIES_CS_MINOR_RADIUS / minor_radius
    B_scale = REACTOR_REFERENCE["B_field"] / target_B
    force_scale_raw = B_scale**2 * L_scale

    small_gap = 0.001
    big_force = 50.0 * 1e6 / force_scale_raw

    metrics = _make_metrics(
        minor_radius=minor_radius,
        target_B=target_B,
        final_min_cc_separation=small_gap,
        final_max_max_coil_force=big_force,
        final_max_force_per_coil=[big_force, big_force],
    )
    result = _compute_reactor_scale_metrics(metrics)

    d_cc = result["reactor_scale_min_cc_separation"]
    w_max = result["max_winding_pack_width"]
    assert w_max > d_cc
    assert result["finite_build_cc_clearance"] < 0


def test_finite_build_cc_clearance_absent_without_cc_sep():
    """If min_cc_separation is not in metrics, no clearance is computed."""
    metrics = _make_metrics(
        minor_radius=0.2,
        target_B=1.0,
        final_max_max_coil_force=1e-3,
        final_max_force_per_coil=[1e-3],
    )
    result = _compute_reactor_scale_metrics(metrics)
    assert "finite_build_cc_clearance" not in result
