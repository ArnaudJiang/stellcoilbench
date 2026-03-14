"""
Centralized matplotlib setup for StellCoilBench.

Provides a single import path for matplotlib availability, backend setup (Agg),
and lazy access to pyplot. Used by optimization, plotting, post-processing,
and sensitivity modules to avoid duplicated try/except blocks.
"""

from __future__ import annotations

from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    MATPLOTLIB_AVAILABLE: bool = True
except ImportError:
    _plt = None  # type: ignore[assignment]
    MATPLOTLIB_AVAILABLE: bool = False


def ensure_mpl_agg() -> None:
    """Set matplotlib backend to Agg if matplotlib is available.

    No-op if matplotlib is not installed. Call explicitly when you need
    to ensure the non-interactive backend is active before any plotting.
    """
    if MATPLOTLIB_AVAILABLE:
        import matplotlib

        matplotlib.use("Agg")


def get_plt() -> Any:
    """Return matplotlib.pyplot if available, otherwise None.

    Returns
    -------
    matplotlib.pyplot | None
        The pyplot module when matplotlib is installed, else None.
    """
    return _plt
