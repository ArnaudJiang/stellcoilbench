"""Named constants for StellCoilBench.

Centralizes magic numbers used across the optimization, post-processing,
and evaluation modules for easier tuning and documentation.

Units convention
---------------
All lengths are in meters [m] internally. Conversion to centimetres [cm]
happens only at Gmsh boundaries when required.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Optimization iteration caps
# ---------------------------------------------------------------------------
CI_MAX_ITER_CAP: int = 10_000
"""Maximum iterations allowed in CI runs to prevent runaway jobs."""

DEFAULT_CI_TIMEOUT_MINUTES: int = 120
"""Default timeout [minutes] for CI autopilot case runs."""

# ---------------------------------------------------------------------------
# L-BFGS-B defaults
# ---------------------------------------------------------------------------
LBFGSB_DEFAULT_MAXLS: int = 40
"""Max line search steps for L-BFGS-B (reduces ABNORMAL_TERMINATION_IN_LNSRCH)."""

LBFGSB_MAXFUN_MULTIPLIER: int = 15_000
"""Multiplier for maxfun = max_iterations * this value."""

# ---------------------------------------------------------------------------
# Coil initialization
# ---------------------------------------------------------------------------
INITIAL_TOTAL_CURRENT: float = 5e7
"""Initial current guess [A] for reactor-scale coil initialization (50 MA)."""

MAX_ADAPTIVE_ITERATIONS: int = 50
"""Maximum iterations for adaptive R0/R1 search in coil initialization."""

ADAPTIVE_TOLERANCE: float = 0.1
"""Fractional tolerance (10%) for distance constraint checks in adaptive search."""

DEFAULT_COIL_QUADPOINTS: int = 256
"""Default number of quadrature points along each coil curve."""

# ---------------------------------------------------------------------------
# Adaptive R0/R1 search scales
# ---------------------------------------------------------------------------
MIN_DISTANCE_FRACTION: float = 0.1
"""Minimum coil-surface and coil-coil distance as fraction of major radius."""

ADAPTIVE_CONVERGENCE_TOL: float = 0.01
"""Convergence tolerance for R0/R1 scale oscillation detection."""

MAX_OSCILLATION_COUNT: int = 3
"""Number of oscillation cycles before terminating adaptive search."""

MAX_LINKING_NUMBER: float = 0.1
"""Linking number threshold below which coils are considered non-interlinked."""

# Scale factors for initial R0/R1 guess (standard modular coils)
R0_SCALE_INIT: float = 1.0
"""Initial R0 scale factor (modular coils)."""
R1_SCALE_INIT: float = 2.5
"""Initial R1 scale factor (modular coils)."""
R0_SCALE_MAX: float = 3.0
"""Maximum R0 scale factor (modular coils)."""
R1_SCALE_MAX: float = 5.0
"""Maximum R1 scale factor (modular coils)."""

PLASMA_BOUNDARY_INNER_FACTOR: float = 0.98
"""Factor for checking if coil points are inside the plasma hole."""
PLASMA_BOUNDARY_OUTER_FACTOR: float = 1.02
"""Factor for checking if coil points extend beyond the plasma."""

R1_GROWTH_FACTOR_LARGE: float = 1.2
"""R1 scale increase when coils need to expand significantly."""
R1_GROWTH_FACTOR_SMALL: float = 1.15
"""R1 scale increase for mild adjustments."""
R0_SHRINK_FACTOR_MILD: float = 0.95
"""R0 scale decrease when coil-surface distance has margin."""
R0_SHRINK_FACTOR_GENTLE: float = 0.98
"""R0 scale decrease when coil-surface distance is generous."""
CS_DISTANCE_MARGIN_TIGHT: float = 1.1
"""Threshold multiplier for tight coil-surface distance margin."""
CS_DISTANCE_MARGIN_LOOSE: float = 1.5
"""Threshold multiplier for loose coil-surface distance margin."""
R0_GROWTH_FACTOR: float = 1.1
"""R0 scale increase for constraint violations."""

# ---------------------------------------------------------------------------
# Current initialisation
# ---------------------------------------------------------------------------
CURRENT_SCALING_FACTOR: float = 1e-7
"""Numerical conditioning factor for simsopt Current objects (I * factor * 1/factor)."""

MAX_CURRENT_ADJUSTMENT_ITERS: int = 30
"""Maximum iterations for current adjustment to match target B-field."""

CURRENT_SCALING_CONVERGENCE_TOL: float = 1e-3
"""Tolerance for current adjustment convergence (fractional)."""

# ---------------------------------------------------------------------------
# Optimizer-specific defaults
# ---------------------------------------------------------------------------
LBFGSB_DEFAULT_MAXCOR: int = 300
"""Default L-BFGS-B history size (maxcor)."""

TNC_DEFAULT_FTOL: float = 1e-6
"""Default function tolerance for TNC optimiser."""

FTOL_DEFAULT: float = 1e-12
"""Default function tolerance for L-BFGS-B and similar optimizers."""

GTOL_DEFAULT_LBFGSB: float = 1e-12
"""Default gradient tolerance for L-BFGS-B."""

GTOL_DEFAULT_BFGS: float = 1e-5
"""Default gradient tolerance for BFGS."""

TOL_DEFAULT: float = 1e-12
"""General optimization tolerance (COBYLA, etc.)."""

NUMERICAL_FLOOR: float = 1e-12
"""Small number to avoid division by zero in Taylor test and similar."""

# ---------------------------------------------------------------------------
# Constraint weights
# ---------------------------------------------------------------------------
DEFAULT_DISTANCE_CONSTRAINT_WEIGHT: float = 1e3
"""Default weight for distance constraints in augmented Lagrangian and weighted sum."""

DEFAULT_FLUX_WEIGHT: float = 1.0
"""Default weight for squared-flux objective."""

# ---------------------------------------------------------------------------
# Taylor test
# ---------------------------------------------------------------------------
TAYLOR_TEST_SEED: int = 42
"""Random seed for Taylor test perturbation direction."""

TAYLOR_TEST_EPSILONS: tuple[float, ...] = (1e-4, 1e-5)
"""Step sizes for finite-difference Taylor test. Coarser than (1e-6,1e-7,1e-8) to avoid machine precision; structural diagnostic showed failure at 1e-6 and below."""

TAYLOR_TEST_ERROR_RATIO_THRESHOLD: float = 0.6
"""Maximum acceptable error ratio between successive Taylor test steps."""

# ---------------------------------------------------------------------------
# Verbose output
# ---------------------------------------------------------------------------
VERBOSE_ITERATION_INTERVAL: int = 100
"""Print verbose output every this many optimizer iterations."""

# ---------------------------------------------------------------------------
# Miscellaneous numerical
# ---------------------------------------------------------------------------
TANGENT_NORM_FLOOR: float = 1e-14
"""Floor for tangent vector norms to avoid division by zero."""

WEIGHT_CALCULATION_TOL: float = 1e-10
"""Tolerance for weight calculation in LinearPenalty scaling."""

# ---------------------------------------------------------------------------
# Finite build
# ---------------------------------------------------------------------------
MIN_POINTS_ALONG_CURVE: int = 128
"""Minimum quadrature points for finite-build coil curves."""

# Tetrahedral mesh size (Gmsh) [m]. Reactor-scale defaults; scaled by 1/a0 for
# device size. After scaling, min/max are floored to avoid memory explosion in
# DOLFINx for small devices (high a0).
DEFAULT_MIN_TETRAHEDRAL_MESH_SIZE_M: float = 0.02
"""Reactor-scale default min element size [m] for Gmsh tetrahedral mesh."""
DEFAULT_MAX_TETRAHEDRAL_MESH_SIZE_M: float = 0.04
"""Reactor-scale default max element size [m] for Gmsh tetrahedral mesh."""
MIN_TETRAHEDRAL_MESH_SIZE_M: float = 0.005
"""Lower bound on min_mesh_size [m] after a0 scaling (prevents memory OOM)."""
MAX_TETRAHEDRAL_MESH_SIZE_M: float = 0.04
"""Lower bound on max_mesh_size [m] after a0 scaling (prevents memory OOM)."""

# ---------------------------------------------------------------------------
# Virtual casing
# ---------------------------------------------------------------------------
VC_SRC_NPHI: int = 80
"""Source-grid toroidal resolution for virtual casing."""

VC_SRC_NTHETA: int = 80
"""Source-grid poloidal resolution for virtual casing."""

# ---------------------------------------------------------------------------
# Post-processing: plotting
# ---------------------------------------------------------------------------
DEFAULT_PLOT_DPI: int = 300
"""Default DPI for publication-quality PNG/PDF plots."""

BN_ERROR_PLOT_DPI: int = 200
"""DPI for inline B·n error 3D figures (lower for speed)."""

BN_ERROR_PDF_DPI: int = 150
"""DPI for B·n error PDF export (vector + raster mix)."""

# ---------------------------------------------------------------------------
# Post-processing: Poincaré / fieldline tracing
# ---------------------------------------------------------------------------
POINCARE_Z_TOLERANCE: float = 0.01
"""Z-plane tolerance [m] for Poincaré section classification."""

DEFAULT_FIELDLINE_TMAX: float = 40000.0
"""Default maximum field-line integration time."""

DEFAULT_FIELDLINE_TOL: float = 1e-12
"""Default ODE tolerance for fieldline integration."""

DEFAULT_NFIELDLINES: int = 20
"""Default number of fieldlines for Poincaré plots."""

# ---------------------------------------------------------------------------
# Post-processing: VMEC
# ---------------------------------------------------------------------------
DEFAULT_VMEC_NS: int = 50
"""Default number of radial surfaces for VMEC equilibrium."""

DEFAULT_SURFACE_NPHI: int = 256
"""Default toroidal grid resolution for surface loading."""

DEFAULT_SURFACE_NTHETA: int = 256
"""Default poloidal grid resolution for surface loading."""

# ---------------------------------------------------------------------------
# Post-processing: SIMPLE particle tracing
# ---------------------------------------------------------------------------
DEFAULT_SIMPLE_NTESTPART: int = 1024
"""Default number of test particles for SIMPLE."""

SIMPLE_SUBPROCESS_TIMEOUT: int = 3600
"""Timeout [s] for the simple.x subprocess (1 hour)."""

# ---------------------------------------------------------------------------
# CLI / Submission
# ---------------------------------------------------------------------------
SUBMISSION_DATETIME_FMT: str = "%m-%d-%Y_%H-%M"
"""Datetime format string for submission directory names."""

COILS_FILENAME: str = "coils.json"
"""Standard filename for serialised coil sets."""

ZIP_FILENAME: str = "all_files.zip"
"""Standard filename for the submission archive."""

DEFAULT_USERNAME: str = "unknown_user"
"""Fallback username when GitHub detection fails."""

# ---------------------------------------------------------------------------
# Update DB / constraints
# ---------------------------------------------------------------------------
CONSTRAINT_VIOLATIONS_KEY: str = "constraint_violations"
"""Key in leaderboard entry dict for list of violated constraint descriptions."""

N_TURNS_MODEL: int = 500
"""Default number of superconductor turns for reactor-scale force model."""

# ---------------------------------------------------------------------------
# Physics constants (ARIES-CS reactor reference)
# ---------------------------------------------------------------------------
ARIES_CS_MINOR_RADIUS: float = 1.7
"""ARIES-CS reference minor radius [m]."""

ARIES_CS_B0: float = 5.7
"""ARIES-CS reference on-axis magnetic field [T]."""

# ---------------------------------------------------------------------------
# Structural analysis: winding-pack material properties (homogenised)
# ---------------------------------------------------------------------------
WP_YOUNGS_MODULUS_PA: float = 100.0e9
"""Homogenised Young's modulus [Pa] for a REBCO/copper/steel winding pack."""

WP_POISSON_RATIO: float = 0.3
"""Homogenised Poisson ratio for a REBCO/copper/steel winding pack."""
