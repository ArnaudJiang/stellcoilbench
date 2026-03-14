"""Post-processing runner registry.

Centralizes the optional post-processing steps (finite-build VTK, structural
analysis, shape gradients) so adding a new runner requires only one tuple.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

# (flag_key, adapter_fn) - adapter takes (bfield, surface, output_dir, opts, results)
RUNNERS: List[Tuple[str, Callable[..., None]]] = []


def _register_runners() -> None:
    """Populate RUNNERS with lazy imports to avoid circular deps."""
    from ._finite_build_runner import _run_finite_build_vtk
    from ._structural_runner import _run_structural, _run_shape_gradient_analysis

    def _run_finite_build(
        bfield: Any,
        surface: Any,
        output_dir: Path,
        opts: Dict[str, Any],
        results: Dict[str, Any],
    ) -> None:
        _run_finite_build_vtk(
            bfield,
            surface,
            output_dir,
            opts.get("finite_build_width"),
            opts.get("finite_build_height"),
            results,
        )

    def _run_structural_adapter(
        bfield: Any,
        surface: Any,
        output_dir: Path,
        opts: Dict[str, Any],
        results: Dict[str, Any],
    ) -> None:
        _run_structural(
            bfield,
            surface,
            output_dir,
            results,
            opts.get("finite_build_width"),
            opts.get("finite_build_height"),
            opts.get("structural_E"),
            opts.get("structural_nu"),
            export_full_coil_set=opts.get("export_structural_full_coil_set", False),
        )

    def _run_shape_gradient_adapter(
        bfield: Any,
        surface: Any,
        output_dir: Path,
        opts: Dict[str, Any],
        results: Dict[str, Any],
    ) -> None:
        _run_shape_gradient_analysis(bfield, surface, output_dir, results)

    global RUNNERS
    RUNNERS = [
        ("plot_finite_build", _run_finite_build),
        ("run_structural", _run_structural_adapter),
        ("compute_shape_gradient", _run_shape_gradient_adapter),
    ]


def run_optional_steps(
    bfield: Any,
    surface: Any,
    output_dir: Path,
    opts: Dict[str, Any],
    results: Dict[str, Any],
    *,
    is_proc0: bool = True,
) -> None:
    """Run all optional post-processing steps whose flags are True in opts.

    Iterates over RUNNERS and invokes each when opts[flag_key] is truthy.
    """
    if not RUNNERS:
        _register_runners()
    for flag_key, adapter in RUNNERS:
        if opts.get(flag_key) and is_proc0:
            adapter(bfield, surface, output_dir, opts, results)
