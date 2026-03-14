"""Surface file path resolution utilities."""

from __future__ import annotations

from pathlib import Path

from ._path_search import find_plasma_surfaces_dir
from ._surface_extraction import get_surface_filename
from ._yaml import load_yaml


def surface_stem_from_filename(filename: str) -> str:
    """
    Extract a normalized surface stem from a surface filename.

    Strips common prefixes (input., wout.) and suffixes (.focus) so that
    "input.LandremanPaul2021_QA", "wout.LandremanPaul2021_QA", and
    "LandremanPaul2021_QA.focus" all yield "LandremanPaul2021_QA".

    Parameters
    ----------
    filename : str
        Surface filename (e.g., "input.LandremanPaul2021_QA", "foo.focus").

    Returns
    -------
    str
        Normalized stem for use as a surface identifier.
    """
    name = Path(filename).name
    for prefix in ("input.", "wout."):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name.replace(".focus", "")


def normalize_surface_id(name: str, for_filename: bool = False) -> str:
    """Canonical surface identifier for leaderboards and filenames.

    Strips input./wout. prefixes and .focus suffix, then optionally
    replaces dots and spaces for filesystem-safe identifiers.

    Parameters
    ----------
    name : str
        Surface name or filename (e.g., "input.LandremanPaul2021_QA").
    for_filename : bool, default=False
        If True, replace "." and " " with "_" for safe filenames.

    Returns
    -------
    str
        Normalized identifier (e.g., "LandremanPaul2021_QA" or
        "LandremanPaul2021_QA" with underscores when for_filename=True).
    """
    base = surface_stem_from_filename(name)
    if for_filename:
        return base.replace(".", "_").replace(" ", "_")
    return base


def get_target_B_from_surface(surface_file: str) -> float:
    """
    Get target on-axis B-field [T] from surface filename.

    Maps known surface names to their design target B-field for coil
    optimization and reactor-scale scaling.

    Parameters
    ----------
    surface_file : str
        Surface filename (e.g., "input.LandremanPaul2021_QA", "muse.focus").

    Returns
    -------
    float
        Target B-field in Tesla.
    """
    sf = surface_file.lower()
    if "muse" in sf:
        return 0.15
    if "landremanpaul2021_qa" in sf:
        return 1.0
    if "landremanpaul2021_qh" in sf or "landremanpaul2021_qh_reactorscale" in sf:
        return 5.7
    if "circular_tokamak" in sf or "rotating_ellipse" in sf or "cfqs_2b40" in sf:
        return 1.0
    if "c09r00" in sf:
        return 0.5  # NCSX from PM4Stell design
    if "w7-x" in sf or "w7x" in sf:
        return 2.5
    if "hsx" in sf:
        return 2.0
    if "schuetthenneberg" in sf:
        return 5.7
    return 5.7  # Default ARIES-CS


def get_surface_search_base_dirs(
    case_path: Path | None = None,
    plasma_surfaces_dir: Path | None = None,
    coils_json_path: Path | None = None,
) -> list[Path]:
    """
    Build list of directories to search for surface files.

    Collects plasma_surfaces directories and case/output directories that may
    contain surface files. Used with resolve_surface_path for consistent
    surface resolution across the codebase.

    Parameters
    ----------
    case_path : Path | None
        Case directory or case.yaml path.
    plasma_surfaces_dir : Path | None
        Explicit plasma_surfaces directory (e.g., from config).
    coils_json_path : Path | None
        Path to coils JSON (used to find plasma_surfaces via find_plasma_surfaces_dir).

    Returns
    -------
    list[Path]
        Directories to search, in priority order.
    """
    dirs: list[Path] = []
    seen: set[str] = set()

    def add(d: Path | None) -> None:
        if d is None or not d.exists():
            return
        key = str(d.resolve())
        if key not in seen:
            seen.add(key)
            dirs.append(d)

    if case_path:
        p = Path(case_path)
        if p.is_file():
            add(p.parent)
            add(find_plasma_surfaces_dir(p.parent))
        elif p.is_dir():
            add(p)
            ps_sub = p / "plasma_surfaces"
            if ps_sub.exists():
                add(ps_sub)
            add(find_plasma_surfaces_dir(p))
    if plasma_surfaces_dir:
        add(Path(plasma_surfaces_dir))
    if coils_json_path:
        add(Path(coils_json_path).parent)
        add(find_plasma_surfaces_dir(Path(coils_json_path).parent))
    add(Path("plasma_surfaces"))
    add(Path.cwd() / "plasma_surfaces")

    return dirs


def resolve_surface_path(
    surface_name: str,
    base_dirs: list[Path],
    case_insensitive: bool = True,
) -> Path | None:
    """
    Resolve a surface filename to an existing file path.

    Searches in base_dirs (e.g., plasma_surfaces, case dirs). Optionally
    falls back to case-insensitive match within those directories.

    Parameters
    ----------
    surface_name : str
        Surface filename (e.g., "input.LandremanPaul2021_QA").
    base_dirs : list[Path]
        Directories to search (e.g., [Path("plasma_surfaces"), Path.cwd() / "plasma_surfaces"]).
    case_insensitive : bool, default=True
        If exact match fails, try case-insensitive directory search.

    Returns
    -------
    Path | None
        Path to surface file if found; None otherwise.
    """
    # Exact match
    for base in base_dirs:
        if not base.exists():
            continue
        candidate = base / surface_name
        if candidate.exists():
            return candidate

    # Case-insensitive fallback
    if case_insensitive:
        surface_lower = surface_name.lower()
        for base in base_dirs:
            if not base.exists() or not base.is_dir():
                continue
            for f in base.iterdir():
                if f.is_file() and f.name.lower() == surface_lower:
                    return f

    return None


def resolve_surface_file_path(
    case_yaml_path: Path | None = None,
    surface_filename: str | None = None,
    plasma_surfaces_dir: Path | None = None,
    coils_json_path: Path | None = None,
) -> Path | None:
    """Resolve a plasma surface file path from case YAML or explicit filename.

    Centralises the surface-path resolution logic shared by coil loading,
    Poincaré plotting, and VMEC input resolution. When case_yaml_path or
    coils_json_path is provided, delegates to resolve_case_and_surface for
    unified resolution; otherwise uses get_surface_search_base_dirs +
    resolve_surface_path for explicit surface_filename lookup.

    Parameters
    ----------
    case_yaml_path : Path, optional
        Path to a ``case.yaml`` file containing ``surface_params.surface``.
    surface_filename : str, optional
        Explicit surface filename to resolve (overrides case YAML lookup).
    plasma_surfaces_dir : Path, optional
        Directory containing plasma surface files.
    coils_json_path : Path, optional
        Coils JSON path used as an additional base directory for search.

    Returns
    -------
    Path or None
        Resolved absolute path to the surface file, or ``None`` if not found.
    """
    from ._case_resolution import resolve_case_and_surface

    # When case/coils context exists, use unified resolver
    if case_yaml_path is not None or coils_json_path is not None:
        result = resolve_case_and_surface(
            case_hint=case_yaml_path,
            coils_path=Path(coils_json_path) if coils_json_path else None,
            plasma_dir=Path(plasma_surfaces_dir) if plasma_surfaces_dir else None,
        )
        if result.surface_path is not None and result.surface_path.exists():
            return result.surface_path
        if surface_filename is None:
            return None

    # Fallback: explicit surface_filename with base dirs
    surf_file = surface_filename
    if surf_file is None and case_yaml_path is not None:
        if not isinstance(case_yaml_path, Path):
            case_yaml_path = Path(case_yaml_path)
        if case_yaml_path.exists():
            try:
                case_data = load_yaml(path=case_yaml_path)
                surf_file = get_surface_filename(case_data)
            except (OSError, ValueError, KeyError, TypeError):
                pass
    if not surf_file:
        return None

    p = Path(surf_file)
    if p.is_absolute() and p.exists():
        return p

    search_kwargs: dict = {}
    if case_yaml_path is not None:
        search_kwargs["case_path"] = (
            case_yaml_path.parent if case_yaml_path.is_file() else case_yaml_path
        )
    if plasma_surfaces_dir is not None:
        search_kwargs["plasma_surfaces_dir"] = Path(plasma_surfaces_dir)
    if coils_json_path is not None:
        search_kwargs["coils_json_path"] = coils_json_path

    base_dirs = get_surface_search_base_dirs(**search_kwargs)
    return resolve_surface_path(surf_file, base_dirs)
