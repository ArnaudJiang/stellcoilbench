"""MMS integration tests."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import (
    _build_box_msh,
    _mms_derive,
    _mms_full_dirichlet,
    _mms_smooth_neumann,
    _save_fem_artifact_fig,
)
from tests.structural_analysis._integration_helpers import (
    _import_ok,
    _solve_with_dolfinx,
    _solve_with_skfem,
    _solve_with_skfem_allbc,
)


class TestManufacturedSolution:
    """Method-of-Manufactured-Solutions (MMS) convergence tests.

    Three variants test different manufactured solutions, all with full-Dirichlet
    BCs on the unit cube so the natural BC is consistent with the PDE:
    1. sin(πx)sin(πy)sin(πz) -- smooth trig, equal coefficients.
    2. x(1-x)y(1-y)z(1-z) -- polynomial, mixed coefficients.
    3. sin-based with anisotropic coefficients [1, 0.8, 0.6].
    """

    def _run_h_convergence(
        self,
        tmp_path,
        body_force_fn,
        u_exact_fn,
        E,
        nu,
        nx_list,
        bc_mode="all",
    ) -> dict[str, list[float]]:
        """Run h-convergence for available backends; return {backend: [errors]}."""
        has_dolfinx = _import_ok("dolfinx")
        has_skfem = _import_ok("skfem")
        errors_all: dict[str, list[float]] = {}
        skfem_fn = _solve_with_skfem_allbc if bc_mode == "all" else _solve_with_skfem
        for backend, solve_fn in [
            ("dolfinx", _solve_with_dolfinx),
            ("skfem", skfem_fn),
        ]:
            if backend == "dolfinx" and not has_dolfinx:
                continue
            if backend == "skfem" and not has_skfem:
                continue
            errors_all[backend] = []
            for nx in nx_list:
                msh_path = _build_box_msh(tmp_path, nx=nx)
                if backend == "dolfinx":
                    qdeg = 4 if bc_mode == "all" else 2
                    u_nodes, _, coords = solve_fn(
                        msh_path, body_force_fn, E, nu, bc_mode=bc_mode, q_degree=qdeg
                    )
                else:
                    u_nodes, _, coords = solve_fn(msh_path, body_force_fn, E, nu)
                u_exact_at_nodes = u_exact_fn(coords)
                diff_sq = np.sum((u_nodes - u_exact_at_nodes) ** 2, axis=1)
                errors_all[backend].append(float(np.sqrt(np.mean(diff_sq))))
        return errors_all

    def test_mms_full_dirichlet(self, tmp_path):
        """MMS: sin(πx)sin(πy)sin(πz), full Dirichlet, h- and p-convergence."""
        pytest.importorskip("sympy", reason="sympy required")
        pytest.importorskip("gmsh", reason="gmsh required")
        has_dolfinx = _import_ok("dolfinx")
        has_skfem = _import_ok("skfem")
        if not has_dolfinx and not has_skfem:
            pytest.skip("No FEM backend available")

        E, nu = 100e9, 0.3
        body_force_fn, u_exact_fn = _mms_full_dirichlet(E, nu)

        nx_list = [4, 6, 8, 12]
        h_vals = [1.0 / nx for nx in nx_list]
        errors_all = self._run_h_convergence(
            tmp_path, body_force_fn, u_exact_fn, E, nu, nx_list
        )

        for bk, errs in errors_all.items():
            assert errs[-1] < errs[0], (
                f"{bk}: finest error should be < coarsest: {errs}"
            )
            log_h = np.log(h_vals)
            log_e = np.log(errs)
            rate = np.polyfit(log_h, log_e, 1)[0]
            assert rate > 1.0, f"{bk}: convergence rate {rate:.2f} < 1.0: {errs}"

        # P-convergence (DOLFINx only)
        p_errors: list[float] = []
        if has_dolfinx:
            msh_p = _build_box_msh(tmp_path, nx=6)
            for deg in [1, 2]:
                u_nodes, _, coords = _solve_with_dolfinx(
                    msh_p, body_force_fn, E, nu, degree=deg, bc_mode="all", q_degree=4
                )
                diff_sq = np.sum((u_nodes - u_exact_fn(coords)) ** 2, axis=1)
                p_errors.append(float(np.sqrt(np.mean(diff_sq))))
            assert p_errors[1] < p_errors[0], f"P2 should beat P1: {p_errors}"

        import matplotlib.pyplot as plt

        fig, (ax_h, ax_p) = plt.subplots(1, 2, figsize=(10, 4))
        for bk, errs in errors_all.items():
            ax_h.loglog(h_vals, errs, "o-", label=bk)
        ref_errs = errors_all[list(errors_all)[0]]
        ax_h.loglog(
            h_vals,
            [ref_errs[0] * (h / h_vals[0]) ** 2 for h in h_vals],
            "k--",
            alpha=0.4,
            label="O(h²)",
        )
        ax_h.set_xlabel("h")
        ax_h.set_ylabel("L2 error")
        ax_h.legend()
        ax_h.set_title("Full Dirichlet: h-convergence")
        if p_errors:
            ax_p.semilogy([1, 2], p_errors, "o-")
            ax_p.set_xlabel("Degree p")
            ax_p.set_ylabel("L2 error")
            ax_p.set_title("Full Dirichlet: p-convergence (nx=6)")
        fig.suptitle("MMS: sin(πx)sin(πy)sin(πz)")
        fig.tight_layout()
        _save_fem_artifact_fig(fig, "mms_full_dirichlet.png")

    def test_mms_polynomial(self, tmp_path):
        """MMS: x(1-x)y(1-y)z(1-z) polynomial, full Dirichlet, h- and p-convergence."""
        pytest.importorskip("sympy", reason="sympy required")
        pytest.importorskip("gmsh", reason="gmsh required")
        has_dolfinx = _import_ok("dolfinx")
        has_skfem = _import_ok("skfem")
        if not has_dolfinx and not has_skfem:
            pytest.skip("No FEM backend available")

        E, nu = 100e9, 0.3
        import sympy as sp

        x_, y_, z_ = sp.symbols("x y z", real=True)
        A = sp.Float(1e-4)
        pxyz = x_ * (1 - x_) * y_ * (1 - y_) * z_ * (1 - z_)
        u_syms = [A * pxyz, A / 2 * pxyz, A / 2 * pxyz]
        body_force_fn, u_exact_fn = _mms_derive(u_syms, E, nu)

        nx_list = [4, 6, 8, 12]
        h_vals = [1.0 / nx for nx in nx_list]
        errors_all = self._run_h_convergence(
            tmp_path, body_force_fn, u_exact_fn, E, nu, nx_list
        )
        for bk, errs in errors_all.items():
            assert errs[-1] < errs[0], f"{bk}: finest should beat coarsest: {errs}"

        # P-convergence (DOLFINx only)
        p_errors: list[float] = []
        if has_dolfinx:
            msh_p = _build_box_msh(tmp_path, nx=6)
            for deg in [1, 2]:
                u_nodes, _, coords = _solve_with_dolfinx(
                    msh_p, body_force_fn, E, nu, degree=deg, bc_mode="all", q_degree=4
                )
                diff_sq = np.sum((u_nodes - u_exact_fn(coords)) ** 2, axis=1)
                p_errors.append(float(np.sqrt(np.mean(diff_sq))))
            assert p_errors[1] < p_errors[0], f"P2 should beat P1: {p_errors}"

        import matplotlib.pyplot as plt

        fig, (ax_h, ax_p) = plt.subplots(1, 2, figsize=(10, 4))
        for bk, errs in errors_all.items():
            ax_h.loglog(h_vals, errs, "o-", label=bk)
        ax_h.set_xlabel("h")
        ax_h.set_ylabel("L2 error")
        ax_h.legend()
        ax_h.set_title("MMS polynomial: h-convergence")
        if p_errors:
            ax_p.semilogy([1, 2], p_errors, "o-")
            ax_p.set_xlabel("Degree p")
            ax_p.set_ylabel("L2 error")
            ax_p.set_title("MMS polynomial: p-convergence (nx=6)")
        fig.suptitle("MMS: x(1-x)y(1-y)z(1-z)")
        fig.tight_layout()
        _save_fem_artifact_fig(fig, "mms_polynomial.png")

    def test_mms_anisotropic(self, tmp_path):
        """MMS: sin-trig with anisotropic coefficients [1, 0.8, 0.6], h- and p-convergence."""
        pytest.importorskip("sympy", reason="sympy required")
        pytest.importorskip("gmsh", reason="gmsh required")
        has_dolfinx = _import_ok("dolfinx")
        has_skfem = _import_ok("skfem")
        if not has_dolfinx and not has_skfem:
            pytest.skip("No FEM backend available")

        E, nu = 100e9, 0.3
        body_force_fn, u_exact_fn = _mms_smooth_neumann(E, nu)

        nx_list = [4, 6, 8, 12]
        h_vals = [1.0 / nx for nx in nx_list]
        errors_all = self._run_h_convergence(
            tmp_path, body_force_fn, u_exact_fn, E, nu, nx_list
        )
        for bk, errs in errors_all.items():
            assert errs[-1] < errs[0], f"{bk}: finest should beat coarsest: {errs}"
            log_h = np.log(h_vals)
            log_e = np.log(errs)
            rate = np.polyfit(log_h, log_e, 1)[0]
            assert rate > 0.8, f"{bk}: convergence rate {rate:.2f} < 0.8: {errs}"

        # P-convergence (DOLFINx only)
        p_errors: list[float] = []
        if has_dolfinx:
            msh_p = _build_box_msh(tmp_path, nx=6)
            for deg in [1, 2]:
                u_nodes, _, coords = _solve_with_dolfinx(
                    msh_p, body_force_fn, E, nu, degree=deg, bc_mode="all", q_degree=4
                )
                diff_sq = np.sum((u_nodes - u_exact_fn(coords)) ** 2, axis=1)
                p_errors.append(float(np.sqrt(np.mean(diff_sq))))
            assert p_errors[1] < p_errors[0], f"P2 should beat P1: {p_errors}"

        import matplotlib.pyplot as plt

        fig, (ax_h, ax_p) = plt.subplots(1, 2, figsize=(10, 4))
        for bk, errs in errors_all.items():
            ax_h.loglog(h_vals, errs, "o-", label=bk)
        ax_h.set_xlabel("h")
        ax_h.set_ylabel("L2 error")
        ax_h.legend()
        ax_h.set_title("MMS anisotropic: h-convergence")
        if p_errors:
            ax_p.semilogy([1, 2], p_errors, "o-")
            ax_p.set_xlabel("Degree p")
            ax_p.set_ylabel("L2 error")
            ax_p.set_title("MMS anisotropic: p-convergence (nx=6)")
        fig.suptitle("MMS trig (anisotropic): A=[1, 0.8, 0.6]")
        fig.tight_layout()
        _save_fem_artifact_fig(fig, "mms_anisotropic.png")
