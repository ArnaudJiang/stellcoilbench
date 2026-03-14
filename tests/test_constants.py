"""Tests for stellcoilbench.constants module.

Verifies that all public constants have the expected types, are not None,
and have physically reasonable values.
"""

from __future__ import annotations

import stellcoilbench.constants as C


class TestConstantTypes:
    """Ensure each constant has the correct Python type."""

    def test_int_constants(self) -> None:
        """Integer constants should be int."""
        int_constants = [
            C.CI_MAX_ITER_CAP,
            C.LBFGSB_DEFAULT_MAXLS,
            C.LBFGSB_MAXFUN_MULTIPLIER,
            C.MAX_ADAPTIVE_ITERATIONS,
            C.DEFAULT_COIL_QUADPOINTS,
            C.MAX_OSCILLATION_COUNT,
            C.MAX_CURRENT_ADJUSTMENT_ITERS,
            C.LBFGSB_DEFAULT_MAXCOR,
            C.TAYLOR_TEST_SEED,
            C.VERBOSE_ITERATION_INTERVAL,
            C.MIN_POINTS_ALONG_CURVE,
            C.VC_SRC_NPHI,
            C.VC_SRC_NTHETA,
            C.DEFAULT_PLOT_DPI,
            C.BN_ERROR_PLOT_DPI,
            C.BN_ERROR_PDF_DPI,
            C.DEFAULT_NFIELDLINES,
            C.DEFAULT_VMEC_NS,
            C.DEFAULT_SURFACE_NPHI,
            C.DEFAULT_SURFACE_NTHETA,
            C.DEFAULT_SIMPLE_NTESTPART,
            C.SIMPLE_SUBPROCESS_TIMEOUT,
            C.N_TURNS_MODEL,
        ]
        for val in int_constants:
            assert isinstance(val, int), f"{val!r} is not int"

    def test_float_constants(self) -> None:
        """Float constants should be float."""
        float_constants = [
            C.INITIAL_TOTAL_CURRENT,
            C.ADAPTIVE_TOLERANCE,
            C.MIN_DISTANCE_FRACTION,
            C.ADAPTIVE_CONVERGENCE_TOL,
            C.MAX_LINKING_NUMBER,
            C.R0_SCALE_INIT,
            C.R1_SCALE_INIT,
            C.R0_SCALE_MAX,
            C.R1_SCALE_MAX,
            C.PLASMA_BOUNDARY_INNER_FACTOR,
            C.PLASMA_BOUNDARY_OUTER_FACTOR,
            C.R1_GROWTH_FACTOR_LARGE,
            C.R1_GROWTH_FACTOR_SMALL,
            C.R0_SHRINK_FACTOR_MILD,
            C.R0_SHRINK_FACTOR_GENTLE,
            C.CS_DISTANCE_MARGIN_TIGHT,
            C.CS_DISTANCE_MARGIN_LOOSE,
            C.R0_GROWTH_FACTOR,
            C.CURRENT_SCALING_FACTOR,
            C.TNC_DEFAULT_FTOL,
            C.DEFAULT_DISTANCE_CONSTRAINT_WEIGHT,
            C.DEFAULT_FLUX_WEIGHT,
            C.TAYLOR_TEST_ERROR_RATIO_THRESHOLD,
            C.TANGENT_NORM_FLOOR,
            C.WEIGHT_CALCULATION_TOL,
            C.POINCARE_Z_TOLERANCE,
            C.DEFAULT_FIELDLINE_TMAX,
            C.DEFAULT_FIELDLINE_TOL,
            C.ARIES_CS_MINOR_RADIUS,
            C.ARIES_CS_B0,
            C.WP_YOUNGS_MODULUS_PA,
            C.WP_POISSON_RATIO,
        ]
        for val in float_constants:
            assert isinstance(val, float), f"{val!r} is not float"

    def test_string_constants(self) -> None:
        """String constants should be str."""
        str_constants = [
            C.SUBMISSION_DATETIME_FMT,
            C.COILS_FILENAME,
            C.ZIP_FILENAME,
            C.DEFAULT_USERNAME,
        ]
        for val in str_constants:
            assert isinstance(val, str), f"{val!r} is not str"

    def test_tuple_constants(self) -> None:
        """Tuple constants should be tuple."""
        assert isinstance(C.TAYLOR_TEST_EPSILONS, tuple)
        assert all(isinstance(e, float) for e in C.TAYLOR_TEST_EPSILONS)


class TestConstantValues:
    """Verify constants have physically reasonable values."""

    def test_no_constants_are_none(self) -> None:
        """No public constant should be None."""
        public_names = [
            name for name in dir(C) if name.isupper() and not name.startswith("_")
        ]
        for name in public_names:
            val = getattr(C, name)
            assert val is not None, f"{name} is None"

    def test_positive_iteration_caps(self) -> None:
        """Iteration caps should be positive integers."""
        assert C.CI_MAX_ITER_CAP > 0
        assert C.MAX_ADAPTIVE_ITERATIONS > 0
        assert C.MAX_CURRENT_ADJUSTMENT_ITERS > 0
        assert C.LBFGSB_DEFAULT_MAXLS > 0

    def test_positive_dpi_values(self) -> None:
        """DPI values should be positive."""
        assert C.DEFAULT_PLOT_DPI > 0
        assert C.BN_ERROR_PLOT_DPI > 0
        assert C.BN_ERROR_PDF_DPI > 0

    def test_physics_constants_reasonable(self) -> None:
        """ARIES-CS and winding-pack constants should be in expected ranges."""
        assert 1.0 < C.ARIES_CS_MINOR_RADIUS < 5.0
        assert 3.0 < C.ARIES_CS_B0 < 10.0
        assert C.WP_YOUNGS_MODULUS_PA > 1e9
        assert 0.0 < C.WP_POISSON_RATIO < 0.5

    def test_scale_factors_positive(self) -> None:
        """Scale init/max factors should be positive."""
        assert C.R0_SCALE_INIT > 0
        assert C.R1_SCALE_INIT > 0
        assert C.R0_SCALE_MAX > C.R0_SCALE_INIT
        assert C.R1_SCALE_MAX > C.R1_SCALE_INIT

    def test_tolerances_positive(self) -> None:
        """Numerical tolerances should be small positive numbers."""
        assert 0 < C.ADAPTIVE_TOLERANCE < 1
        assert 0 < C.ADAPTIVE_CONVERGENCE_TOL < 1
        assert C.TANGENT_NORM_FLOOR > 0
        assert C.WEIGHT_CALCULATION_TOL > 0
        assert C.DEFAULT_FIELDLINE_TOL > 0

    def test_taylor_test_epsilons_decreasing(self) -> None:
        """Taylor test epsilons should be in decreasing order."""
        eps = C.TAYLOR_TEST_EPSILONS
        assert len(eps) >= 2
        for i in range(len(eps) - 1):
            assert eps[i] > eps[i + 1]

    def test_filenames_nonempty(self) -> None:
        """Filename constants should be non-empty strings."""
        assert len(C.COILS_FILENAME) > 0
        assert len(C.ZIP_FILENAME) > 0
        assert len(C.SUBMISSION_DATETIME_FMT) > 0

    def test_constraint_violations_key(self) -> None:
        """CONSTRAINT_VIOLATIONS_KEY used by update_db for leaderboard entries."""
        assert C.CONSTRAINT_VIOLATIONS_KEY == "constraint_violations"
