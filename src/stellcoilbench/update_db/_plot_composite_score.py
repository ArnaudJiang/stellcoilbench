"""Composite score vs. date plot for reactor-scale leaderboard.

Generates a time-series plot of composite score by submission date, with
one color per plasma surface. Used in the reactor-scale leaderboard docs.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from ._writers_common import _surface_display_name

logger = logging.getLogger(__name__)

# Color palette: distinct, colorblind-friendly colors for up to ~12 surfaces
_SURFACE_COLORS = [
    "#0173b2",  # blue
    "#de8f05",  # orange
    "#029e73",  # teal
    "#cc78bc",  # magenta
    "#ca9161",  # brown
    "#fbafe4",  # pink
    "#949494",  # gray
    "#ece133",  # yellow
    "#56b4e9",  # light blue
    "#d55e00",  # red-orange
    "#009e73",  # green
    "#f0e442",  # pale yellow
]


def _parse_run_date(run_date: str) -> datetime | None:
    """Parse ISO-format run_date string to datetime."""
    if not run_date:
        return None
    try:
        # Handle both "2026-02-27T01:39:12.423195" and "2026-02-27"
        if "T" in run_date:
            return datetime.fromisoformat(run_date.replace("Z", "+00:00"))
        return datetime.strptime(run_date[:10], "%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def plot_composite_score_vs_date(
    surface_leaderboards: Dict[str, Dict[str, Any]],
    out_path: Path,
    figsize: tuple[float, float] = (10, 6),
    dpi: int = 150,
) -> bool:
    """Plot composite score vs. submission date, one series per plasma surface.

    Parameters
    ----------
    surface_leaderboards : dict[str, dict]
        Per-surface data from build_surface_leaderboards. Each surface has
        ``entries`` with ``composite_score`` and ``run_date``.
    out_path : Path
        Output file path (PNG or SVG).
    figsize : tuple[float, float], optional
        Figure size in inches (width, height).
    dpi : int, optional
        DPI for raster output (PNG).

    Returns
    -------
    bool
        True if the plot was written successfully; False if no plottable data.
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import numpy as np

    # Collect (date, composite_score) per surface
    surface_data: Dict[str, list[tuple[datetime, float]]] = {}
    for surface_name, surf_data in surface_leaderboards.items():
        entries = surf_data.get("entries", [])
        points: list[tuple[datetime, float]] = []
        for e in entries:
            run_date = e.get("run_date", "")
            composite = e.get("composite_score")
            if composite is None:
                continue
            dt = _parse_run_date(run_date)
            if dt is not None:
                points.append((dt, float(composite)))
        if points:
            points.sort(key=lambda p: p[0])
            surface_data[surface_name] = points

    if not surface_data:
        logger.debug(
            "No plottable composite_score/run_date data for score-vs-date plot"
        )
        return False

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#fafafa")
    fig.patch.set_facecolor("white")

    surfaces_sorted = sorted(surface_data.keys())
    for i, surface_name in enumerate(surfaces_sorted):
        points = surface_data[surface_name]
        color = _SURFACE_COLORS[i % len(_SURFACE_COLORS)]
        dates = [p[0] for p in points]
        scores = np.array([p[1] for p in points])
        label = _surface_display_name(surface_name)
        ax.scatter(
            dates,
            scores,
            c=color,
            s=36,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
            label=f"{label} (n={len(points)})",
        )
        # Optional: connect points with a light line for trend
        ax.plot(
            dates,
            scores,
            color=color,
            alpha=0.4,
            linewidth=1.5,
            zorder=2,
        )

    ax.set_xlabel("Submission date", fontsize=11, fontweight="medium")
    ax.set_ylabel("Composite score", fontsize=11, fontweight="medium")
    ax.set_title(
        "Reactor-scale composite score over time",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))
    plt.xticks(rotation=35, ha="right")
    # Start x-axis at February 20, 2026
    ax.set_xlim(left=datetime(2026, 2, 20))
    ax.legend(
        loc="upper left",
        fontsize=9,
        framealpha=0.95,
        edgecolor="#cccccc",
        fancybox=True,
    )
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()
    if suffix == ".svg":
        fig.savefig(out_path, format="svg", bbox_inches="tight")
    else:
        fig.savefig(out_path, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote composite score vs. date plot to %s", out_path)
    return True
