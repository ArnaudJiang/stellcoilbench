"""Resolve visualization plot links for leaderboard entries."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, Tuple

from ._path_parsing import parse_submission_path

GITHUB_BASE_URL = "https://cdn.jsdelivr.net/gh/akaptano/stellcoilbench@main"

# Relative path from docs/_build/html/leaderboard/*.html to repo root (4 levels up).
LOCAL_LINK_PREFIX = "../../../../"


def _resolve_plot_path(repo_root: Path, base: Path) -> Path | None:
    """Return path to plot file if it exists. Prefer PNG, fall back to PDF."""
    for ext in (".png", ".pdf"):
        cand = base.with_suffix(ext)
        if (repo_root / cand).exists():
            return cand
    return None


def resolve_visualization_links(
    entry: Dict[str, Any],
    repo_root: Path,
    *,
    github_base_url: str = GITHUB_BASE_URL,
    use_local_links: bool = False,
) -> Tuple[str, str, str, str]:
    """Resolve initial (i) and final (f) coil visualization plot links for a leaderboard entry.

    Handles both standard submissions and Fourier continuation submissions (plots in
    order_X subdirs). Prefers PNG, falls back to PDF for backward compatibility.

    Parameters
    ----------
    entry : dict
        Leaderboard entry with ``path``, ``rank``, ``metrics``.
    repo_root : Path
        Repository root for resolving paths.
    github_base_url : str, optional
        Base URL for CDN links (default: jsdelivr stellcoilbench).
    use_local_links : bool, optional
        If True, emit relative file paths (e.g. ../../../../submissions/.../file.pdf)
        instead of CDN URLs. Use when building docs for local viewing so plots open
        from disk and avoid jsDelivr's 50 MB package limit.

    Returns
    -------
    i_link_html : str
        HTML for initial coil plot link (or "—" if not found).
    i_link_sort : str
        Sort value for initial column.
    f_link_html : str
        HTML for final coil plot link (or "—" if not found).
    f_link_sort : str
        Sort value for final column.
    """
    rank_num = str(entry.get("rank", "-"))
    entry_path = entry.get("path", "")
    if entry_path.startswith("/"):
        entry_path = entry_path[1:]
    i_link_html, i_link_sort = "—", ""
    f_link_html, f_link_sort = "—", ""

    if not entry_path:
        return i_link_html, i_link_sort, f_link_html, f_link_sort

    path_obj = Path(entry_path)
    submission_dir = _resolve_submission_dir(path_obj, repo_root)
    if not submission_dir:
        return i_link_html, i_link_sort, f_link_html, f_link_sort

    full_submission_dir = (repo_root / submission_dir).resolve()
    metrics = entry.get("metrics") or {}
    fourier_orders_str = metrics.get("fourier_continuation_orders")
    is_fourier_continuation = fourier_orders_str and fourier_orders_str != "—"
    orders: list[int] = []

    if is_fourier_continuation and isinstance(fourier_orders_str, str):
        try:
            orders = [int(o.strip()) for o in fourier_orders_str.split(",")]
        except (ValueError, AttributeError):
            orders = []

    def _url(rel_path: Path) -> str:
        path_str = str(rel_path).replace("\\", "/")
        return (
            f"{LOCAL_LINK_PREFIX}{path_str}"
            if use_local_links
            else f"{github_base_url}/{path_str}"
        )

    def _link(url: str, label: str, *, target_blank: bool = True) -> str:
        target = ' target="_blank"' if target_blank else ""
        return f'<a href="{html.escape(url)}"{target}>{html.escape(label)}</a>'

    if is_fourier_continuation and orders:
        order_dirs = []
        for order in orders:
            od = full_submission_dir / f"order_{order}"
            if od.exists() and od.is_dir():
                order_dirs.append((order, f"order_{order}"))
        if order_dirs:
            first_order, first_od = order_dirs[0]
            init_base = submission_dir / first_od / "bn_error_3d_plot_initial"
            init_plot = _resolve_plot_path(repo_root, init_base)
            if init_plot is not None:
                i_link_html = _link(_url(init_plot), rank_num)
                i_link_sort = rank_num
            f_links = []
            for order, od_name in order_dirs:
                fp_base = submission_dir / od_name / "bn_error_3d_plot"
                fp = _resolve_plot_path(repo_root, fp_base)
                if fp is not None:
                    f_links.append(_link(_url(fp), str(order)))
            if f_links:
                f_link_html = " ".join(f_links)
                f_link_sort = rank_num
    else:
        final_plot = _resolve_plot_path(
            repo_root, submission_dir / "bn_error_3d_plot"
        )
        init_plot = _resolve_plot_path(
            repo_root, submission_dir / "bn_error_3d_plot_initial"
        )
        if final_plot is not None:
            f_link_html = _link(_url(final_plot), rank_num)
            f_link_sort = rank_num
        if init_plot is not None:
            i_link_html = _link(_url(init_plot), rank_num)
            i_link_sort = rank_num

    return i_link_html, i_link_sort, f_link_html, f_link_sort


def _resolve_submission_dir(path_obj: Path, repo_root: Path) -> Path | None:
    """Resolve submission directory from entry path for plot lookup.

    Handles three path types:
    - **all_files.zip**: uses parent directory as submission dir.
    - **Other .zip**: parses surface/user/timestamp via :func:`parse_submission_path`,
      builds ``submissions/surface/user/timestamp`` and verifies it exists;
      falls back to zip parent if not.
    - **results.json or dir**: uses ``path_obj.parent``.

    Normalizes to a path relative to repo_root for CDN URL construction.
    Returns None if resolution fails (e.g. absolute path outside repo).

    Parameters
    ----------
    path_obj : Path
        Entry path from leaderboard (e.g. ``submissions/QA/user1/run1/results.json``).
    repo_root : Path
        Repository root.

    Returns
    -------
    Path | None
        Relative submission directory, or None.
    """
    submissions_root = repo_root / "submissions"
    submission_dir: Path | None = None

    if path_obj.name == "all_files.zip":
        submission_dir = path_obj.parent
    elif path_obj.suffix == ".zip":
        parsed = parse_submission_path(path_obj, submissions_root)
        surface = parsed.get("surface", "unknown")
        user = parsed.get("user", "unknown")
        timestamp = parsed.get("timestamp", "") or path_obj.stem
        if timestamp.endswith(".zip"):
            timestamp = Path(timestamp).stem
        if surface != "unknown" and user != "unknown":
            cand = Path("submissions") / surface / user / timestamp
            if (repo_root / cand).exists():
                submission_dir = cand
            else:
                submission_dir = path_obj.parent
        else:
            cand = Path("submissions") / user / timestamp
            if (repo_root / cand).exists():
                submission_dir = cand
            else:
                submission_dir = path_obj.parent
    else:
        submission_dir = path_obj.parent

    if not submission_dir:
        return None
    # Normalize to relative path
    if submission_dir.is_absolute():
        try:
            submission_dir = submission_dir.relative_to(repo_root.resolve())
        except ValueError:
            sd_str = str(submission_dir)
            if "submissions" in sd_str:
                idx = sd_str.find("submissions")
                submission_dir = Path(sd_str[idx:])
            else:
                return path_obj.parent if not path_obj.is_absolute() else None
    sd_str = str(submission_dir).replace("\\", "/").lstrip("/")
    if sd_str.startswith("./"):
        sd_str = sd_str[2:].lstrip("/")
    return Path(sd_str) if sd_str else submission_dir
