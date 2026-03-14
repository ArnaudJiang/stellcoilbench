"""Cross-validation tests: compare DOLFINx and scikit-fem on identical meshes."""

from __future__ import annotations

import numpy as np

from conftest import (
    _build_box_msh,
    _require_both_backends,
)
from tests.structural_analysis._integration_helpers import (
    _solve_with_dolfinx,
    _solve_with_skfem,
)


class TestCrossValidation:
    """Compare DOLFINx and scikit-fem on identical meshes and loads.

    Both solvers use the same Gmsh mesh, the same material constants, and the
    same boundary-condition strategy (bottom-15%-z clamped).  The results should
    agree to within a few percent for aggregate metrics (max/mean displacement
    and Von Mises stress).  Per-node agreement is tighter because both use
    linear tets (CG-1) and a direct solver.
    """

    def test_uniform_gravity_on_unit_cube(self, tmp_path):
        """Uniform downward body force on a [0,1]³ cube."""
        _require_both_backends()

        msh_path = _build_box_msh(tmp_path, nx=3)
        E, nu = 100e9, 0.3

        def body_force(coords):
            f = np.zeros_like(coords)
            f[:, 2] = -1e6
            return f

        u_dfx, vm_dfx, _ = _solve_with_dolfinx(msh_path, body_force, E, nu)
        u_skf, vm_skf, _ = _solve_with_skfem(msh_path, body_force, E, nu)

        max_disp_dfx = np.max(np.linalg.norm(u_dfx, axis=1))
        max_disp_skf = np.max(np.linalg.norm(u_skf, axis=1))
        assert max_disp_dfx > 0
        assert max_disp_skf > 0
        np.testing.assert_allclose(
            max_disp_skf,
            max_disp_dfx,
            rtol=0.10,
            err_msg="Max displacement mismatch between backends",
        )

        max_vm_dfx = np.max(vm_dfx)
        max_vm_skf = np.max(vm_skf)
        mean_vm_dfx = np.mean(vm_dfx)
        mean_vm_skf = np.mean(vm_skf)
        np.testing.assert_allclose(
            max_vm_skf,
            max_vm_dfx,
            rtol=0.15,
            err_msg="Max Von Mises mismatch between backends",
        )
        np.testing.assert_allclose(
            mean_vm_skf,
            mean_vm_dfx,
            rtol=0.10,
            err_msg="Mean Von Mises mismatch between backends",
        )

    def test_lateral_shear_on_unit_cube(self, tmp_path):
        """Uniform lateral (x-direction) body force on a [0,1]³ cube."""
        _require_both_backends()

        msh_path = _build_box_msh(tmp_path, nx=3)
        E, nu = 200e9, 0.25

        def body_force(coords):
            f = np.zeros_like(coords)
            f[:, 0] = 5e5
            return f

        u_dfx, vm_dfx, _ = _solve_with_dolfinx(msh_path, body_force, E, nu)
        u_skf, vm_skf, _ = _solve_with_skfem(msh_path, body_force, E, nu)

        max_disp_dfx = np.max(np.linalg.norm(u_dfx, axis=1))
        max_disp_skf = np.max(np.linalg.norm(u_skf, axis=1))
        assert max_disp_dfx > 0
        assert max_disp_skf > 0
        np.testing.assert_allclose(
            max_disp_skf,
            max_disp_dfx,
            rtol=0.10,
            err_msg="Max displacement mismatch (lateral shear)",
        )

        np.testing.assert_allclose(
            np.mean(vm_skf),
            np.mean(vm_dfx),
            rtol=0.10,
            err_msg="Mean Von Mises mismatch (lateral shear)",
        )

    def test_position_dependent_force_on_unit_cube(self, tmp_path):
        """Body force that varies linearly with z: f_z = -1e6 * (1 + z)."""
        _require_both_backends()

        msh_path = _build_box_msh(tmp_path, nx=4)
        E, nu = 150e9, 0.3

        def body_force(coords):
            f = np.zeros_like(coords)
            f[:, 2] = -1e6 * (1.0 + coords[:, 2])
            return f

        u_dfx, vm_dfx, _ = _solve_with_dolfinx(msh_path, body_force, E, nu)
        u_skf, vm_skf, _ = _solve_with_skfem(msh_path, body_force, E, nu)

        max_disp_dfx = np.max(np.linalg.norm(u_dfx, axis=1))
        max_disp_skf = np.max(np.linalg.norm(u_skf, axis=1))
        assert max_disp_dfx > 0
        assert max_disp_skf > 0
        np.testing.assert_allclose(
            max_disp_skf,
            max_disp_dfx,
            rtol=0.10,
            err_msg="Max displacement mismatch (z-varying force)",
        )

        np.testing.assert_allclose(
            np.max(vm_skf),
            np.max(vm_dfx),
            rtol=0.15,
            err_msg="Max Von Mises mismatch (z-varying force)",
        )
        np.testing.assert_allclose(
            np.mean(vm_skf),
            np.mean(vm_dfx),
            rtol=0.10,
            err_msg="Mean Von Mises mismatch (z-varying force)",
        )
