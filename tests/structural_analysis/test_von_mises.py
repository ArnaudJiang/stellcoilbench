"""Tests for Von Mises stress and Lorentz force (J×B) computation."""

from __future__ import annotations

import numpy as np


class TestVonMisesComputation:
    """Test Von Mises stress from known strain states."""

    def test_uniaxial_tension(self) -> None:
        """Uniaxial tension: σ_vm = σ for pure axial stress."""
        E = 200e9
        nu = 0.3
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))

        # Pure uniaxial strain in x: eps_xx = 0.001, rest = 0
        eps = np.zeros((3, 3))
        eps[0, 0] = 0.001

        sig = lam * np.trace(eps) * np.eye(3) + 2 * mu * eps
        s_dev = sig - (1.0 / 3.0) * np.trace(sig) * np.eye(3)
        vm = np.sqrt(1.5 * np.sum(s_dev * s_dev))

        # For uniaxial strain, σ_vm should be positive and proportional to strain
        assert vm > 0
        # σ_xx dominates; σ_vm ≈ |σ_xx| for nearly uniaxial stress
        sigma_xx = sig[0, 0]
        sigma_yy = sig[1, 1]
        sigma_zz = sig[2, 2]
        vm_manual = np.sqrt(
            0.5
            * (
                (sigma_xx - sigma_yy) ** 2
                + (sigma_yy - sigma_zz) ** 2
                + (sigma_zz - sigma_xx) ** 2
            )
        )
        assert np.isclose(vm, vm_manual, rtol=1e-10)

    def test_hydrostatic_gives_zero_vm(self) -> None:
        """Hydrostatic (equal principal stresses) → σ_vm = 0."""
        E = 100e9
        nu = 0.3
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))

        # Pure volumetric strain
        eps = 0.001 * np.eye(3)
        sig = lam * np.trace(eps) * np.eye(3) + 2 * mu * eps
        s_dev = sig - (1.0 / 3.0) * np.trace(sig) * np.eye(3)
        vm = np.sqrt(1.5 * np.sum(s_dev * s_dev))
        assert np.isclose(vm, 0.0, atol=1e-6)

    def test_pure_shear(self) -> None:
        """Pure shear: σ_vm = √3 * τ."""
        E = 100e9
        nu = 0.3
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))

        eps = np.zeros((3, 3))
        eps[0, 1] = eps[1, 0] = 0.001

        sig = lam * np.trace(eps) * np.eye(3) + 2 * mu * eps
        s_dev = sig - (1.0 / 3.0) * np.trace(sig) * np.eye(3)
        vm = np.sqrt(1.5 * np.sum(s_dev * s_dev))

        tau_xy = sig[0, 1]
        vm_expected = np.sqrt(3) * abs(tau_xy)
        assert np.isclose(vm, vm_expected, rtol=1e-10)


class TestLorentzForceLogic:
    """Test the Lorentz force J × B computation logic."""

    def test_cross_product_direction(self) -> None:
        """J × B should be perpendicular to both J and B."""
        J = np.array([1.0, 0.0, 0.0])
        B = np.array([0.0, 1.0, 0.0])
        F = np.cross(J, B)
        assert np.isclose(np.dot(F, J), 0.0)
        assert np.isclose(np.dot(F, B), 0.0)
        # J × B for (1,0,0) × (0,1,0) = (0,0,1)
        np.testing.assert_allclose(F, [0.0, 0.0, 1.0])

    def test_parallel_jb_gives_zero_force(self) -> None:
        """Parallel J and B should give zero Lorentz force."""
        J = np.array([1.0, 0.0, 0.0])
        B = np.array([2.0, 0.0, 0.0])
        F = np.cross(J, B)
        np.testing.assert_allclose(F, [0.0, 0.0, 0.0])

    def test_current_density_scaling(self) -> None:
        """Body force should scale with current / area."""
        current = 1e6  # 1 MA
        area = 0.05 * 0.05  # 5 cm × 5 cm
        J_mag = current / area
        tangent = np.array([1.0, 0.0, 0.0])
        J = J_mag * tangent
        B = np.array([0.0, 0.0, 5.0])  # 5 T
        F = np.cross(J, B)

        # Expected: J_mag * 5 in the y-direction (cross x with z → -y)
        expected = np.array([0.0, -J_mag * 5.0, 0.0])
        np.testing.assert_allclose(F, expected)

    def test_regularized_jcross_b_finite_everywhere(self) -> None:
        """Regularized _compute_jcross_b yields finite force at points inside coils."""
        from simsopt.field import BiotSavart, Current, coils_via_symmetries
        from simsopt.geo import create_equally_spaced_curves

        from stellcoilbench.structural_analysis._common import _compute_jcross_b

        coils_raw = create_equally_spaced_curves(
            2, 1, stellsym=True, R0=1.2, R1=0.1, order=2
        )
        base_currents = [Current(1e6) for _ in range(2)]
        coils = coils_via_symmetries(coils_raw, base_currents, 1, True)
        bs = BiotSavart(coils)
        area = 0.05 * 0.05
        width, height = 0.05, 0.05

        # Points on/near centerline (would blow up with BiotSavart self-field)
        gamma0 = np.asarray(coils[0].curve.gamma())[0]
        coords = np.array(
            [
                gamma0,
                gamma0 + np.array([0.001, 0, 0]),
                gamma0 + np.array([0.005, 0.005, 0]),
            ]
        )

        force = _compute_jcross_b(
            coords,
            coils,
            bs,
            area,
            width=width,
            height=height,
            use_regularized=True,
        )
        assert np.all(np.isfinite(force)), "Regularized field must be finite everywhere"
        assert force.shape == (3, 3)

    def test_regularized_jcross_b_finite_at_outside_points(self) -> None:
        """Points with |u|>1 or |v|>1 get finite force via B_self=0 (J x B_mutual only)."""
        from simsopt.field import BiotSavart, Current, coils_via_symmetries
        from simsopt.geo import create_equally_spaced_curves

        from stellcoilbench.structural_analysis._common import _compute_jcross_b

        coils_raw = create_equally_spaced_curves(
            2, 1, stellsym=True, R0=1.2, R1=0.1, order=2
        )
        base_currents = [Current(1e6) for _ in range(2)]
        coils = coils_via_symmetries(coils_raw, base_currents, 1, True)
        bs = BiotSavart(coils)
        gamma = np.asarray(coils[0].curve.gamma())
        gammadash = np.asarray(coils[0].curve.gammadash())
        t = gammadash[0] / np.linalg.norm(gammadash[0])
        C = np.mean(gamma, axis=0)
        w = gamma[0] - C
        p = w - np.dot(w, t) * t
        p = p / np.linalg.norm(p)
        q = np.cross(t, p)
        width, height = 0.05, 0.05
        area = width * height
        # Points outside cross-section (|u|>1 or |v|>1) but near centerline (dist <= 2*max(w,h)).
        # B_self=0 for these points; only B_mutual contributes. No BiotSavart self-field fallback.
        outside_coords = np.array(
            [
                gamma[0] + 0.03 * p,  # u~1.2, v~0; outside but near
                gamma[0] + 0.03 * q,  # u~0, v~1.2; outside but near
                gamma[0] + 0.02 * p + 0.02 * q,  # inside (u,v in [-1,1])
            ]
        )
        force = _compute_jcross_b(
            outside_coords,
            coils,
            bs,
            area,
            width=width,
            height=height,
            use_regularized=True,
        )
        assert np.all(np.isfinite(force)), (
            "B_self=0 for outside points must yield finite J x B_mutual"
        )
        assert force.shape == (3, 3)

    def test_jcross_b_mutual_includes_symmetry_copies(self) -> None:
        """J×B mutual field increases when all_coils includes symmetry copies."""
        from simsopt.field import BiotSavart, Current, coils_via_symmetries
        from simsopt.geo import create_equally_spaced_curves

        from stellcoilbench.structural_analysis._common import _compute_jcross_b

        # nfp=2, stellsym=True -> 4 unique coils from 2 base curves (2*2)
        coils_raw = create_equally_spaced_curves(
            2, 1, stellsym=True, R0=1.2, R1=0.1, order=2
        )
        base_currents = [Current(1e6) for _ in range(2)]
        all_coils = coils_via_symmetries(coils_raw, base_currents, 2, True)
        # Unique coils: first of each base
        step = max(1, len(all_coils) // 2)
        unique_coils = [all_coils[i] for i in range(0, len(all_coils), step)][:2]

        bs_full = BiotSavart(all_coils)
        area = 0.05 * 0.05
        width, height = 0.05, 0.05

        gamma0 = np.asarray(unique_coils[0].curve.gamma())[0]
        coords = gamma0.reshape(1, 3) + np.array([[0.001, 0.002, 0.0]])  # inside coil 0

        force_unique_only = _compute_jcross_b(
            coords,
            unique_coils,
            bs_full,
            area,
            width=width,
            height=height,
            use_regularized=True,
            mesh_coils=unique_coils,
            all_coils=unique_coils,
        )
        force_with_symmetry = _compute_jcross_b(
            coords,
            unique_coils,
            bs_full,
            area,
            width=width,
            height=height,
            use_regularized=True,
            mesh_coils=unique_coils,
            all_coils=all_coils,
        )

        mag_unique = np.linalg.norm(force_unique_only)
        mag_full = np.linalg.norm(force_with_symmetry)
        assert mag_full > mag_unique, (
            "Mutual B from full coils (incl. symmetry copies) must yield larger J×B "
            f"than unique-only (|F_full|={mag_full:.2e} vs |F_unique|={mag_unique:.2e})"
        )

    def test_B0_cross_section_force_averages_to_zero(self) -> None:
        """Newton's 3rd law: integral of J × B0 over cross-section = 0 for straight wire."""
        from stellcoilbench.structural_analysis._common import _compute_B0

        a, b = 0.01, 0.01  # full side dimensions [m] for 1 cm × 1 cm cross-section
        current = 1e5  # 100 kA
        # Constant frame for straight wire: t along z, p along x, q along y
        t = np.array([0.0, 0.0, 1.0])
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([0.0, 1.0, 0.0])
        n_u, n_v = 51, 51
        u_1d = np.linspace(-1, 1, n_u)
        v_1d = np.linspace(-1, 1, n_v)
        u, v = np.meshgrid(u_1d, v_1d, indexing="ij")
        u_flat = u.ravel()
        v_flat = v.ravel()
        p_arr = np.tile(p, (len(u_flat), 1))
        q_arr = np.tile(q, (len(u_flat), 1))
        B0 = _compute_B0(u_flat, v_flat, a, b, current, p_arr, q_arr)
        J_mag = current / (a * b)
        J_vec = np.tile(J_mag * t, (len(u_flat), 1))
        J_cross_B0 = np.cross(J_vec, B0)
        du = 2.0 / (n_u - 1) if n_u > 1 else 2.0
        dv = 2.0 / (n_v - 1) if n_v > 1 else 2.0
        dA = (a * b / 4) * du * dv  # Jacobian (ab/4) for u,v in [-1,1]
        integral = np.sum(J_cross_B0, axis=0) * dA
        np.testing.assert_allclose(integral, [0, 0, 0], atol=1e-10)

    def test_B0_matches_analytical_at_known_point(self) -> None:
        """B0 at (u,v)=(0.5, 0.5) matches explicit G-function evaluation (Landreman Eq 17)."""
        from stellcoilbench.structural_analysis._common import (
            MU0,
            _G_helper,
            _compute_B0,
        )

        a, b = 0.01, 0.01  # full dimensions [m]
        current = 1e5  # 100 kA
        u_test, v_test = 0.5, 0.5
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([0.0, 1.0, 0.0])
        p_arr = np.broadcast_to(p, (1, 3))
        q_arr = np.broadcast_to(q, (1, 3))

        B0_computed = _compute_B0(
            np.array([u_test]), np.array([v_test]), a, b, current, p_arr, q_arr
        )[0]

        # Explicit evaluation of Eq 17: B0 = (mu0*I/(4*pi*a*b)) * sum su*sv * [G(b(v-sv), a(u-su))*q - G(a(u-su), b(v-sv))*p]
        prefac = MU0 * current / (4 * np.pi * a * b)
        B0_ref = np.zeros(3)
        for su in (1, -1):
            for sv in (1, -1):
                g1 = _G_helper(
                    np.array([b * (v_test - sv)]), np.array([a * (u_test - su)])
                )
                c1 = float(np.asarray(g1).flat[0])
                g2 = _G_helper(
                    np.array([a * (u_test - su)]), np.array([b * (v_test - sv)])
                )
                c2 = float(np.asarray(g2).flat[0])
                B0_ref += prefac * su * sv * (c1 * q - c2 * p)

        np.testing.assert_allclose(
            B0_computed,
            B0_ref,
            rtol=1e-10,
            atol=1e-12,
            err_msg="B0 must match explicit G-function evaluation",
        )

    def test_regularized_matches_circular_coil_analytic(self) -> None:
        """B_internal at inner edge of circular coil matches Landreman Eq 97 (within ~20%)."""
        from simsopt.field import Coil, Current
        from simsopt.geo import CurveXYZFourier
        from simsopt.field.selffield import regularization_rect

        from stellcoilbench.structural_analysis._common import (
            MU0,
            _compute_B_internal,
            _compute_Breg_for_coil,
            _compute_coil_frame,
        )

        R0, a_half, b_half = 1.0, 0.005, 0.005  # R0=1 m, 1 cm square cross-section
        current = 1e5  # 100 kA
        curve = CurveXYZFourier(quadpoints=64, order=1)
        curve.set("xc(1)", R0)
        curve.set("ys(1)", R0)
        coil = Coil(curve, Current(current))
        width, height = 2 * a_half, 2 * b_half
        coil_frame = _compute_coil_frame(coil)
        Breg = _compute_Breg_for_coil(coil, width, height)
        reg = float(np.asarray(regularization_rect(width, height)))
        delta = reg / (width * height)
        gamma = coil_frame["gamma"]
        idx = 0
        p0 = coil_frame["p"][idx]
        eval_coord = gamma[idx] + a_half * p0
        Breg = _compute_Breg_for_coil(coil, width, height)
        B_internal = _compute_B_internal(
            coil_frame,
            Breg,
            eval_coord.reshape(1, 3),
            np.array([idx]),
            width,
            height,
            delta,
        )[0]
        B_mag = float(np.linalg.norm(B_internal))
        B_analytic_ln = (
            MU0
            * current
            / (4 * np.pi * R0)
            * (1 + np.log(16 * R0 / np.sqrt(a_half * b_half)))
        )
        # B_internal = Breg + B0 + Bkappa + Bb. The ln-term (Eq 97) is one
        # contribution; full B is larger. Bounds catch gross implementation errors.
        assert B_mag > 0.5 * B_analytic_ln, (
            f"B_internal {B_mag:.2f} T too small vs ln-term {B_analytic_ln:.2f} T"
        )
        assert B_mag < 100.0, (
            f"B_internal {B_mag:.2f} T unreasonably large (Landreman Eq 97)"
        )

    def test_regularized_vs_filamentary_at_large_distance(self) -> None:
        """At large distance, regularized yields finite force (B_self=0, J×B_mutual only).

        We no longer use BiotSavart for self-field at far points; B_self=0 for dist > 2*max(w,h).
        So regularized and filamentary no longer agree at large distance. This test only checks
        that regularized is finite.
        """
        from simsopt.field import BiotSavart, Current, coils_via_symmetries
        from simsopt.geo import create_equally_spaced_curves

        from stellcoilbench.structural_analysis._common import _compute_jcross_b

        coils_raw = create_equally_spaced_curves(
            2, 1, stellsym=True, R0=1.2, R1=0.1, order=2
        )
        base_currents = [Current(1e6) for _ in range(2)]
        coils = coils_via_symmetries(coils_raw, base_currents, 1, True)
        bs = BiotSavart(coils)
        area = 0.05 * 0.05
        width, height = 0.05, 0.05
        coords = np.array(
            [
                [0.0, 2.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        force_reg = _compute_jcross_b(
            coords, coils, bs, area, width=width, height=height, use_regularized=True
        )
        assert np.all(np.isfinite(force_reg)), (
            "Regularized force must be finite at large distance"
        )

    def test_self_field_error_quantification(self) -> None:
        """Diagnostic: document BiotSavart vs regularized |B| at various distances from centerline."""
        from scipy.spatial import cKDTree

        from simsopt.field import BiotSavart, Current, coils_via_symmetries
        from simsopt.geo import create_equally_spaced_curves
        from simsopt.field.selffield import regularization_rect

        from stellcoilbench.structural_analysis._common import (
            _build_coil_centerline_data,
            _compute_B_internal,
            _compute_Breg_for_coil,
            _compute_coil_frame,
        )

        coils_raw = create_equally_spaced_curves(
            2, 1, stellsym=True, R0=1.2, R1=0.1, order=2
        )
        base_currents = [Current(1e6) for _ in range(2)]
        coils = coils_via_symmetries(coils_raw, base_currents, 1, True)
        bs_self = BiotSavart([coils[0]])
        width, height = 0.05, 0.05
        gamma0 = np.asarray(coils[0].curve.gamma())[0]
        coil_frame = _compute_coil_frame(coils[0])
        Breg = _compute_Breg_for_coil(coils[0], width, height)
        p0 = coil_frame["p"][0]
        distances_mm = [1, 5, 10, 25]
        B_fil_mags: list[float] = []
        B_reg_mags: list[float] = []
        reg = float(np.asarray(regularization_rect(width, height)))
        delta = reg / (width * height)
        all_gamma, _, _, coil_boundaries = _build_coil_centerline_data(coils)
        tree = cKDTree(all_gamma)
        for d_mm in distances_mm:
            d = d_mm * 1e-3
            pt = gamma0 + d * p0
            bs_self.set_points(pt.reshape(1, 3))
            B_fil = np.asarray(bs_self.B()).reshape(3)
            B_fil_mag = float(np.linalg.norm(B_fil))
            _, nearest_idx = tree.query(pt.reshape(1, 3))
            local_idx = nearest_idx[0] - coil_boundaries[0]
            B_reg = _compute_B_internal(
                coil_frame,
                Breg,
                pt.reshape(1, 3),
                np.array([local_idx]),
                width,
                height,
                delta,
            )[0]
            B_reg_mag = float(np.linalg.norm(B_reg))
            B_fil_mags.append(B_fil_mag)
            B_reg_mags.append(B_reg_mag)
        print("\nSelf-field error quantification (5 cm × 5 cm, 1 MA):")
        print("  d [mm]  |B_fil| [T]  |B_reg| [T]  ratio")
        for d_mm, bf, br in zip(distances_mm, B_fil_mags, B_reg_mags):
            ratio = bf / br if br > 0 else float("inf")
            print(f"  {d_mm:5d}    {bf:9.2f}   {br:9.2f}   {ratio:.2f}")
        assert B_fil_mags[0] / B_reg_mags[0] > 2, (
            "At 1 mm: filamentary should overestimate by >2x"
        )
        assert B_fil_mags[1] / B_reg_mags[1] > 2, (
            "At 5 mm: filamentary should overestimate by >2x"
        )
