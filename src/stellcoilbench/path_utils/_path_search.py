"""Path search utilities: walk up directory tree to find files/dirs."""

from __future__ import annotations

from pathlib import Path


def _find_path_up(
    start_path: Path,
    name: str,
    max_levels: int,
    is_dir: bool,
) -> Path | None:
    """
    Walk up from start_path to find a file or directory with the given name.

    Parameters
    ----------
    start_path : Path
        Directory to start searching from.
    name : str
        Name of the file or directory to find.
    max_levels : int
        Maximum number of parent levels to traverse.
    is_dir : bool
        If True, look for a directory; if False, look for a file.

    Returns
    -------
    Path | None
        Path if found; None otherwise.
    """
    current = Path(start_path).resolve()
    for _ in range(max_levels):
        candidate = current / name
        if candidate.exists() and (
            candidate.is_dir() if is_dir else candidate.is_file()
        ):
            return candidate
        if current.parent == current:
            break
        current = current.parent
    return None


def find_file_up(
    start_path: Path,
    file_name: str,
    max_levels: int = 5,
) -> Path | None:
    """
    Walk up from start_path to find a file with the given name.

    Parameters
    ----------
    start_path : Path
        Directory to start searching from.
    file_name : str
        Name of the file to find (e.g., "case.yaml").
    max_levels : int, default=5
        Maximum number of parent levels to traverse.

    Returns
    -------
    Path | None
        Path to the file if found; None otherwise.
    """
    return _find_path_up(start_path, file_name, max_levels, is_dir=False)


def find_dir_up(
    start_path: Path,
    dir_name: str,
    max_levels: int = 5,
) -> Path | None:
    """
    Walk up from start_path to find a directory with the given name.

    Parameters
    ----------
    start_path : Path
        Directory to start searching from.
    dir_name : str
        Name of the directory to find (e.g., "plasma_surfaces", "cases").
    max_levels : int, default=5
        Maximum number of parent levels to traverse.

    Returns
    -------
    Path | None
        Path to the directory if found; None otherwise.
    """
    return _find_path_up(start_path, dir_name, max_levels, is_dir=True)


def find_repo_root(start_path: Path | None = None, max_levels: int = 15) -> Path | None:
    """Walk up from start_path to find repository root (.git or pyproject.toml).

    Parameters
    ----------
    start_path : Path | None
        Directory to start searching from. Defaults to Path.cwd().
    max_levels : int, default=15
        Maximum number of parent levels to traverse.

    Returns
    -------
    Path | None
        Repository root directory if found; None otherwise.
    """
    current = Path(start_path or Path.cwd()).resolve()
    if not current.is_dir():
        current = current.parent
    for _ in range(max_levels):
        if (current / ".git").exists():
            return current
        if (current / "pyproject.toml").exists():
            return current
        if current.parent == current:
            break
        current = current.parent
    return None


def find_plasma_surfaces_dir(start_path: Path, max_levels: int = 5) -> Path | None:
    """
    Walk up from start_path to find a plasma_surfaces directory.

    Parameters
    ----------
    start_path : Path
        Directory to start searching from (e.g., output dir, case dir).
    max_levels : int, default=5
        Maximum number of parent levels to traverse.

    Returns
    -------
    Path | None
        Path to plasma_surfaces directory if found; None otherwise.
    """
    return find_dir_up(start_path, "plasma_surfaces", max_levels)
