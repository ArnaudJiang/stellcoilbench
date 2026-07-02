"""FOCUS optimization backend integration.

This module treats FOCUS as an external coil generator and converts its
outputs into the existing simsopt coil objects used by StellCoilBench metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

import numpy as np

from ..constants import COILS_FILENAME, DEFAULT_COIL_QUADPOINTS
from ..mpi_utils import proc0_print
from ._adaptive_search import _coils_via_symmetries_compat


@dataclass
class FocusHarmonicCoil:
    """One FOCUS Fourier coil block."""

    name: str
    coil_type: int
    coil_symm: int
    nseg: int
    current: float
    length: float
    target_length: float
    xc: list[float]
    xs: list[float]
    yc: list[float]
    ys: list[float]
    zc: list[float]
    zs: list[float]

    @property
    def order(self) -> int:
        return len(self.xc) - 1


@dataclass
class FocusHarmonicData:
    """Parsed FOCUS harmonic coil file."""

    coils: list[FocusHarmonicCoil]
    path: Path

    @property
    def ncoils(self) -> int:
        return len(self.coils)

    @property
    def order(self) -> int:
        return max((coil.order for coil in self.coils), default=0)


@dataclass
class FocusFilamentCoil:
    """One coil represented as discrete FOCUS filament points."""

    name: str
    coil_id: int
    current: float
    points: np.ndarray


@dataclass
class FocusFilamentData:
    """Parsed FOCUS filament file."""

    coils: list[FocusFilamentCoil]
    path: Path
    nfp: int | None = None


def _next_data_line(lines: list[str], idx: int) -> tuple[str, int]:
    """Return next non-comment, non-empty line and its following index."""
    while idx < len(lines):
        line = lines[idx].strip()
        idx += 1
        if line and not line.startswith("#"):
            return line, idx
    raise ValueError("Unexpected end of FOCUS file")


def parse_focus_harmonics(path: Path | str) -> FocusHarmonicData:
    """Parse a FOCUS ``*.focus`` harmonic coil output file."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    line, idx = _next_data_line(lines, 0)
    ncoils = int(line.split()[0])
    coils: list[FocusHarmonicCoil] = []

    for _ in range(ncoils):
        line, idx = _next_data_line(lines, idx)
        coil_parts = line.split()
        if len(coil_parts) < 3:
            raise ValueError(f"Invalid FOCUS coil header in {path}: {line}")
        coil_type = int(coil_parts[0])
        coil_symm = int(coil_parts[1])
        name = coil_parts[2]

        line, idx = _next_data_line(lines, idx)
        meta_parts = line.split()
        if len(meta_parts) < 6:
            raise ValueError(f"Invalid FOCUS coil metadata in {path}: {line}")
        nseg = int(meta_parts[0])
        current = float(meta_parts[1])
        length = float(meta_parts[3])
        target_length = float(meta_parts[5])

        line, idx = _next_data_line(lines, idx)
        nfcoil = int(line.split()[0])

        rows: list[list[float]] = []
        for _row in range(6):
            line, idx = _next_data_line(lines, idx)
            values = [float(v) for v in line.split()]
            if len(values) != nfcoil + 1:
                raise ValueError(
                    f"Expected {nfcoil + 1} Fourier coefficients for {name}, "
                    f"got {len(values)}"
                )
            rows.append(values)

        coils.append(
            FocusHarmonicCoil(
                name=name,
                coil_type=coil_type,
                coil_symm=coil_symm,
                nseg=nseg,
                current=current,
                length=length,
                target_length=target_length,
                xc=rows[0],
                xs=rows[1],
                yc=rows[2],
                ys=rows[3],
                zc=rows[4],
                zs=rows[5],
            )
        )

    return FocusHarmonicData(coils=coils, path=path)


def parse_focus_filaments(path: Path | str) -> FocusFilamentData:
    """Parse a FOCUS ``*.coils`` filament output file.

    The parser keeps the first occurrence of each named coil. FOCUS filament
    files may include symmetry copies separated by zero-current name markers.
    """
    path = Path(path)
    nfp: int | None = None
    seen_names: set[str] = set()
    coils: list[FocusFilamentCoil] = []
    current_points: list[list[float]] = []
    current_value: float | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.lower().startswith("periods"):
            parts = line.split()
            if len(parts) >= 2:
                nfp = int(parts[1])
            continue
        if line.lower().startswith(("begin", "mirror", "end")):
            continue

        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            x, y, z, current = (float(parts[i]) for i in range(4))
        except ValueError:
            continue

        current_points.append([x, y, z])
        if current_value is None and abs(current) > 0.0:
            current_value = current

        if abs(current) == 0.0 and len(parts) >= 6:
            try:
                coil_id = int(parts[4])
            except ValueError:
                coil_id = len(coils) + 1
            name = parts[5]
            if name not in seen_names and len(current_points) > 2:
                coils.append(
                    FocusFilamentCoil(
                        name=name,
                        coil_id=coil_id,
                        current=float(abs(current_value or 0.0)),
                        points=np.asarray(current_points, dtype=float),
                    )
                )
                seen_names.add(name)
            current_points = []
            current_value = None

    return FocusFilamentData(coils=coils, path=path, nfp=nfp)


def read_focus_h5_metadata(path: Path | str | None) -> dict[str, Any]:
    """Read lightweight FOCUS HDF5 metadata when h5py is available."""
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    try:
        import h5py  # type: ignore[import-not-found]
    except ImportError:
        return {}

    metadata: dict[str, Any] = {}
    with h5py.File(path, "r") as handle:
        for key in ("Ncoils", "Nfp", "NFcoil", "Nseg", "time_optimize"):
            if key in handle:
                value = handle[key][()]
                metadata[key] = np.asarray(value).reshape(-1)[0].item()
    return metadata


def _focus_component_dofs(cos_coeffs: list[float], sin_coeffs: list[float]) -> np.ndarray:
    """Map FOCUS coefficient rows to simsopt CurveXYZFourier DOF order."""
    order = len(cos_coeffs) - 1
    dofs = np.zeros(2 * order + 1)
    dofs[0] = cos_coeffs[0]
    for n in range(1, order + 1):
        dofs[2 * n - 1] = sin_coeffs[n]
        dofs[2 * n] = cos_coeffs[n]
    return dofs


def _curve_from_focus_harmonic(coil: FocusHarmonicCoil, numquadpoints: int):
    """Create a simsopt CurveXYZFourier from one parsed FOCUS harmonic coil."""
    from simsopt.geo import CurveXYZFourier

    curve = CurveXYZFourier(numquadpoints, coil.order)
    dofs = np.concatenate(
        [
            _focus_component_dofs(coil.xc, coil.xs),
            _focus_component_dofs(coil.yc, coil.ys),
            _focus_component_dofs(coil.zc, coil.zs),
        ]
    )
    curve.set_dofs(dofs)
    return curve


def _fit_periodic_fourier(points: np.ndarray, order: int) -> tuple[np.ndarray, ...]:
    """Fit FOCUS filament points to FOCUS-style Fourier coefficient rows."""
    pts = np.asarray(points, dtype=float)
    if np.linalg.norm(pts[0] - pts[-1]) < 1e-10:
        pts = pts[:-1]
    npts = len(pts)
    if npts < 2 * order + 1:
        raise ValueError(
            f"Need at least {2 * order + 1} filament points to fit order {order}"
        )

    theta = np.linspace(0.0, 2.0 * np.pi, npts, endpoint=False)
    cols = [np.ones(npts)]
    for n in range(1, order + 1):
        cols.append(np.sin(n * theta))
        cols.append(np.cos(n * theta))
    basis = np.column_stack(cols)

    rows = []
    for component in range(3):
        coeffs, *_ = np.linalg.lstsq(basis, pts[:, component], rcond=None)
        cos_coeffs = [float(coeffs[0])]
        sin_coeffs = [0.0]
        for n in range(1, order + 1):
            sin_coeffs.append(float(coeffs[2 * n - 1]))
            cos_coeffs.append(float(coeffs[2 * n]))
        rows.extend([np.asarray(cos_coeffs), np.asarray(sin_coeffs)])
    return tuple(rows)


def _harmonics_from_filaments(
    filament_data: FocusFilamentData,
    order: int,
) -> FocusHarmonicData:
    """Convert filament coils to fitted harmonic coil data."""
    coils: list[FocusHarmonicCoil] = []
    for filament in filament_data.coils:
        xc, xs, yc, ys, zc, zs = _fit_periodic_fourier(filament.points, order)
        coils.append(
            FocusHarmonicCoil(
                name=filament.name,
                coil_type=1,
                coil_symm=1,
                nseg=len(filament.points),
                current=filament.current,
                length=0.0,
                target_length=0.0,
                xc=xc.tolist(),
                xs=xs.tolist(),
                yc=yc.tolist(),
                ys=ys.tolist(),
                zc=zc.tolist(),
                zs=zs.tolist(),
            )
        )
    return FocusHarmonicData(coils=coils, path=filament_data.path)


def focus_harmonics_to_simsopt_coils(
    harmonic_data: FocusHarmonicData,
    *,
    nfp: int,
    stellsym: bool,
    numquadpoints: int = DEFAULT_COIL_QUADPOINTS,
) -> list:
    """Convert parsed FOCUS harmonic coils to a full simsopt coil set."""
    from simsopt.field import Current

    base_curves = [
        _curve_from_focus_harmonic(coil, numquadpoints) for coil in harmonic_data.coils
    ]
    base_currents = [Current(float(coil.current)) for coil in harmonic_data.coils]
    return _coils_via_symmetries_compat(base_curves, base_currents, nfp, stellsym)


def _resolve_existing_path(path_value: str | None, base_dirs: list[Path]) -> Path | None:
    """Resolve a configured path if present."""
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    for base in base_dirs:
        candidate = base / path
        if candidate.exists():
            return candidate.resolve()
    return (base_dirs[0] / path).resolve()


def _format_focus_arg(arg: str, values: dict[str, Any]) -> str:
    """Format a FOCUS command argument with known runtime placeholders."""
    return arg.format(**{k: str(v) for k, v in values.items()})


def _stage_focus_input_files(
    input_files: list[str],
    *,
    base_dirs: list[Path],
    output_dir: Path,
) -> list[Path]:
    """Copy configured FOCUS input files into the backend run directory."""
    staged: list[Path] = []
    for item in input_files:
        src = _resolve_existing_path(item, base_dirs)
        if src is None or not src.exists():
            raise FileNotFoundError(f"FOCUS input file not found: {item}")
        dst = output_dir / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        staged.append(dst)
    return staged


def write_focus_boundary(surface: Any, path: Path, *, atol: float = 1e-14) -> None:
    """Write a simsopt ``SurfaceRZFourier`` in FOCUS boundary format."""
    def _safe_coeff(method_name: str, m: int, n: int) -> float:
        try:
            return float(getattr(surface, method_name)(m, n))
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return 0.0

    rows: list[tuple[int, int, float, float, float, float]] = []
    mpol = int(getattr(surface, "mpol"))
    ntor = int(getattr(surface, "ntor"))
    for m in range(mpol + 1):
        for n in range(-ntor, ntor + 1):
            rc = _safe_coeff("get_rc", m, n)
            rs = _safe_coeff("get_rs", m, n)
            zc = _safe_coeff("get_zc", m, n)
            zs = _safe_coeff("get_zs", m, n)
            if any(abs(v) > atol for v in (rc, rs, zc, zs)):
                rows.append((n, m, rc, rs, zc, zs))

    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Nbmn   Nfp  Nbnf\n")
        handle.write(f"{len(rows)} {int(getattr(surface, 'nfp', 1))} 0\n")
        handle.write("# plasma boundary\n")
        handle.write("# n m Rbc Rbs Zbc Zbs\n")
        for n, m, rc, rs, zc, zs in rows:
            handle.write(
                f"{n:d} {m:d} {rc: .16E} {rs: .16E} {zc: .16E} {zs: .16E}\n"
            )
        handle.write("# Bn harmonics\n")
        handle.write("# n m bnc bns\n")


def _write_focus_input_file(
    path: Path,
    *,
    boundary_name: str,
    ncoils: int,
    order: int,
    max_iterations: int,
    focus_params: dict[str, Any],
    surface: Any,
    coil_params: dict[str, Any],
) -> None:
    """Write a minimal FOCUS namelist driven by StellCoilBench case parameters."""
    init_current = float(focus_params.get("init_current", 1.0e6))
    init_radius = float(
        focus_params.get(
            "init_radius",
            max(0.25, 0.5 * float(getattr(surface, "minor_radius", lambda: 1.0)())),
        )
    )
    nseg = int(focus_params.get("nseg", coil_params.get("numquadpoints", 128)))
    nteta = int(focus_params.get("nteta", 64))
    nzeta = int(focus_params.get("nzeta", 64))
    weight_bnorm = float(focus_params.get("weight_bnorm", 100.0))
    weight_ttlen = float(focus_params.get("weight_ttlen", 1.0))
    weight_cssep = float(focus_params.get("weight_cssep", 10.0))
    weight_ccsep = float(focus_params.get("weight_ccsep", 10.0))
    weight_curv = float(focus_params.get("weight_curv", 1.0))
    target_length = float(focus_params.get("target_length", 0.0))
    curv_k0 = float(focus_params.get("curv_k0", 5.0))
    ccsep_alpha = float(focus_params.get("ccsep_alpha", 10.0))
    ccsep_beta = float(focus_params.get("ccsep_beta", 2.0))
    cssep_factor = float(focus_params.get("cssep_factor", 4.0))
    is_symmetric = int(focus_params.get("is_symmetric", 2))
    case_postproc = int(focus_params.get("case_postproc", 0))

    contents = f"""&focusin
 IsQuiet        =       -1
 IsSymmetric    =        {is_symmetric}
 input_surf     =       '{boundary_name}'
 input_harm     =       'target.harmonics'
 input_coils    =       'none'
 case_surface   =        0
 Nteta          =        {nteta}
 Nzeta          =        {nzeta}
 case_init      =        1
 case_coils     =        1
 Ncoils         =        {ncoils}
 init_current   =        {init_current:.16E}
 init_radius    =        {init_radius:.16E}
 IsVaryCurrent  =        {int(focus_params.get("is_vary_current", 0))}
 IsVaryGeometry =        1
 NFcoil         =        {order}
 Nseg           =        {nseg}
 IsNormalize    =        1
 IsNormWeight   =        1
 case_bnormal   =        1
 case_length    =        1
 case_curv      =        3
 curv_alpha     =        2.0
 curv_k0        =        {curv_k0:.16E}
 weight_bnorm   =        {weight_bnorm:.16E}
 weight_bharm   =        0.0
 weight_tflux   =        0.0
 target_tflux   =        0.0
 weight_ttlen   =        {weight_ttlen:.16E}
 target_length  =        {target_length:.16E}
 weight_specw   =        0.0
 weight_cssep   =        {weight_cssep:.16E}
 cssep_factor   =        {cssep_factor:.16E}
 weight_ccsep   =        {weight_ccsep:.16E}
 ccsep_alpha    =        {ccsep_alpha:.16E}
 ccsep_beta     =        {ccsep_beta:.16E}
 weight_curv    =        {weight_curv:.16E}
 weight_inorm   =        1.0
 weight_gnorm   =        1.0
 weight_mnorm   =        1.0
 case_optimize  =        1
 exit_tol       =        {float(focus_params.get("exit_tol", 1.0e-4)):.16E}
 DF_maxiter     =        {int(focus_params.get("df_maxiter", 0))}
 DF_xtol        =        1.0E-8
 DF_tausta      =        0.0
 DF_tauend      =        1.0
 CG_maxiter     =        {max_iterations}
 CG_xtol        =        1.0E-8
 CG_wolfe_c1    =        0.1
 CG_wolfe_c2    =        0.9
 LM_maxiter     =        0
 HN_maxiter     =        0
 TN_maxiter     =        0
 case_postproc  =        {case_postproc}
 save_freq      =        {int(focus_params.get("save_freq", max(1, max_iterations)))}
 save_coils     =        1
 save_harmonics =        0
 save_filaments =        0
 update_plasma  =        0
/
"""
    path.write_text(contents, encoding="utf-8")


def _ensure_default_focus_inputs(
    focus_params: dict[str, Any],
    *,
    output_dir: Path,
    surface: Any,
    coil_params: dict[str, Any],
    optimizer_params: dict[str, Any],
) -> dict[str, Any]:
    """Generate default FOCUS boundary/input files when no command is supplied."""
    params = dict(focus_params)
    if params.get("arguments") or params.get("input_files"):
        return params

    stem = str(params.get("run_stem", "focus_run"))
    boundary_path = output_dir / f"{stem}.boundary"
    input_path = output_dir / f"{stem}.input"
    write_focus_boundary(surface, boundary_path)
    _write_focus_input_file(
        input_path,
        boundary_name=boundary_path.name,
        ncoils=int(coil_params.get("ncoils", params.get("ncoils", 4))),
        order=int(coil_params.get("order", params.get("order", 8))),
        max_iterations=int(optimizer_params.get("max_iterations", 50)),
        focus_params=params,
        surface=surface,
        coil_params=coil_params,
    )
    params["arguments"] = [input_path.name]
    params.setdefault("output_harmonics_file", f"{stem}.focus")
    params.setdefault("output_filaments_file", f"coils.{stem}")
    params.setdefault("output_h5_file", f"{stem}.h5")
    return params


def _run_focus_executable(
    focus_params: dict[str, Any],
    *,
    output_dir: Path,
    case_yaml_path_abs: Path | None,
    case_path: Path,
    surface_file: str | None,
) -> None:
    """Run the configured external FOCUS command unless ``skip_run`` is set."""
    if focus_params.get("skip_run", False):
        proc0_print("Skipping FOCUS executable run (focus_params.skip_run=true)")
        return

    executable = focus_params.get("executable")
    if not executable:
        raise ValueError("focus_params.executable is required for FOCUS backend runs")

    base_dirs = [
        output_dir,
        case_yaml_path_abs.parent if case_yaml_path_abs else Path.cwd(),
        case_path.parent if case_path.is_file() else case_path,
        Path.cwd(),
    ]
    staged_inputs = _stage_focus_input_files(
        list(focus_params.get("input_files", [])),
        base_dirs=base_dirs,
        output_dir=output_dir,
    )
    values = {
        "run_dir": output_dir,
        "case_yaml": case_yaml_path_abs or case_path,
        "surface": surface_file or "",
        "input_file": staged_inputs[0] if staged_inputs else "",
    }
    args = [
        _format_focus_arg(str(arg), values)
        for arg in focus_params.get("arguments", [])
    ]
    command = [str(executable), *args]
    stdout_path = output_dir / "focus_stdout.log"
    stderr_path = output_dir / "focus_stderr.log"
    proc0_print(f"Running FOCUS backend command: {' '.join(command)}")
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        subprocess.run(
            command,
            cwd=output_dir,
            stdout=stdout,
            stderr=stderr,
            timeout=int(focus_params.get("timeout_seconds", 3600)),
            check=True,
        )


def _load_focus_output(
    focus_params: dict[str, Any],
    *,
    output_dir: Path,
    case_yaml_path_abs: Path | None,
    case_path: Path,
    surface_nfp: int,
    coil_order: int,
) -> tuple[FocusHarmonicData, dict[str, Any]]:
    """Load configured FOCUS output into harmonic data."""
    base_dirs = [
        output_dir,
        case_yaml_path_abs.parent if case_yaml_path_abs else Path.cwd(),
        case_path.parent if case_path.is_file() else case_path,
        Path.cwd(),
    ]
    harmonics_path = _resolve_existing_path(
        focus_params.get("output_harmonics_file"),
        base_dirs,
    )
    filaments_path = _resolve_existing_path(
        focus_params.get("output_filaments_file"),
        base_dirs,
    )
    h5_path = _resolve_existing_path(focus_params.get("output_h5_file"), base_dirs)
    metadata = read_focus_h5_metadata(h5_path)
    parser = focus_params.get("parser", "auto")

    if parser in {"auto", "focus_fourier"} and harmonics_path and harmonics_path.exists():
        return parse_focus_harmonics(harmonics_path), metadata
    if parser == "focus_fourier":
        raise FileNotFoundError(f"FOCUS harmonic output not found: {harmonics_path}")

    if parser in {"auto", "focus_filament"} and filaments_path and filaments_path.exists():
        filament_data = parse_focus_filaments(filaments_path)
        order = int(focus_params.get("order", coil_order))
        if order < 1:
            order = max(1, int(metadata.get("NFcoil", coil_order or 1)))
        if "Nfp" not in metadata and filament_data.nfp is not None:
            metadata["Nfp"] = filament_data.nfp
        return _harmonics_from_filaments(filament_data, order), metadata

    raise FileNotFoundError(
        "No readable FOCUS output found. Configure output_harmonics_file "
        "or output_filaments_file."
    )


def run_focus_backend(
    surface: Any,
    case_cfg: Any,
    coil_params: dict[str, Any],
    optimizer_params: dict[str, Any],
    output_dir: Path,
    surface_resolution: int,
    case_yaml_path_abs: Path | None,
    case_path: Path,
) -> tuple[list, dict[str, Any]]:
    """Run FOCUS and return simsopt coils plus unified benchmark metrics."""
    from simsopt import save
    from simsopt.field import BiotSavart

    from ._external_eval import evaluate_external_coils

    start = time.perf_counter()
    focus_params = _ensure_default_focus_inputs(
        dict(case_cfg.focus_params or {}),
        output_dir=output_dir,
        surface=surface,
        coil_params=coil_params,
        optimizer_params=optimizer_params,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    surface_file = getattr(surface, "filename", None)
    surface_range = getattr(surface, "range", "half period")

    _run_focus_executable(
        focus_params,
        output_dir=output_dir,
        case_yaml_path_abs=case_yaml_path_abs,
        case_path=case_path,
        surface_file=surface_file,
    )
    harmonic_data, metadata = _load_focus_output(
        focus_params,
        output_dir=output_dir,
        case_yaml_path_abs=case_yaml_path_abs,
        case_path=case_path,
        surface_nfp=int(getattr(surface, "nfp", 1)),
        coil_order=int(coil_params.get("order", 1)),
    )

    nfp = int(focus_params.get("nfp", metadata.get("Nfp", getattr(surface, "nfp", 1))))
    stellsym = bool(focus_params.get("stellsym", getattr(surface, "stellsym", False)))
    numquadpoints = int(
        focus_params.get("numquadpoints", coil_params.get("numquadpoints", DEFAULT_COIL_QUADPOINTS))
    )
    coils = focus_harmonics_to_simsopt_coils(
        harmonic_data,
        nfp=nfp,
        stellsym=stellsym,
        numquadpoints=numquadpoints,
    )

    coils_json_path = output_dir / COILS_FILENAME
    save(coils, coils_json_path)
    BiotSavart(coils).save(output_dir / "biot_savart_optimized.json")

    metrics = evaluate_external_coils(
        coils_json_path,
        surface_file=str(surface_file),
        surface_range=surface_range,
        surface_resolution=surface_resolution,
    )
    elapsed = float(metadata.get("time_optimize", time.perf_counter() - start))
    metrics.update(
        {
            "optimization_backend": "focus",
            "optimization_time": elapsed,
            "walltime_sec": time.perf_counter() - start,
            "iterations_used": int(metadata.get("iout", 0) or 0),
            "focus_output_harmonics": str(harmonic_data.path),
            "focus_ncoils_base": harmonic_data.ncoils,
            "focus_coil_order": harmonic_data.order,
        }
    )
    return coils, metrics
