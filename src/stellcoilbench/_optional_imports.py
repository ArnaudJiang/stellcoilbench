"""Optional imports with consistent error handling for StellCoilBench.

Centralizes try/except patterns for modules that may be missing (e.g. reactor_scale
in stale editable installs). Reduces duplication of editable-install error messages.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

T = TypeVar("T")


def optional_import(module: str, name: str, fallback: T | None = None) -> Any | T:
    """Import a name from a module, returning fallback on ImportError or AttributeError.

    Tries ``__import__(module, fromlist=[name])`` and ``getattr`` to obtain
    the requested name. When ``name`` is empty, returns the module itself
    (useful for top-level modules like ``booz_xform``).

    Parameters
    ----------
    module : str
        Module path (e.g. ``"simsopt.field.magneticfield"``).
    name : str
        Attribute name to retrieve. If empty, returns the module object.
    fallback : object, optional
        Value to return on ImportError or AttributeError. Default is None.

    Returns
    -------
    object
        The imported name (or module when name is empty), or fallback.
    """
    try:
        if name:
            mod = __import__(module, fromlist=[name])
            return getattr(mod, name)
        return __import__(module)
    except (ImportError, AttributeError):
        return fallback


_REACTOR_SCALE_EDITABLE_MSG = (
    "stellcoilbench.reactor_scale not found. This usually means the editable "
    "install is stale or pointing to the wrong directory. Run 'pip install -e .' "
    "from the stellcoilbench repo root, and ensure you're using the correct "
    "conda environment (stellcoilbench_vmec)."
)


def get_reactor_scale_compute() -> Callable[..., dict[str, Any]] | None:
    """Return compute_reactor_scale_metrics or None if unavailable.

    Use when reactor-scale metrics are optional (e.g. generate-submission
    may run without reactor_scale).

    Returns
    -------
    callable or None
        compute_reactor_scale_metrics from reactor_scale module, or None.
    """
    try:
        from .reactor_scale import compute_reactor_scale_metrics

        return compute_reactor_scale_metrics
    except ModuleNotFoundError:
        return None


def require_reactor_scale_compute() -> Callable[..., dict[str, Any]]:
    """Return compute_reactor_scale_metrics; raise with helpful message if unavailable.

    Use when reactor-scale metrics are required (e.g. submit-case, submission
    packaging, ci_autopilot).

    Returns
    -------
    callable
        compute_reactor_scale_metrics from reactor_scale module.

    Raises
    ------
    ModuleNotFoundError
        When reactor_scale cannot be imported, with editable-install hints.
    """
    try:
        from .reactor_scale import compute_reactor_scale_metrics

        return compute_reactor_scale_metrics
    except ModuleNotFoundError as e:
        if "reactor_scale" in str(e):
            raise ModuleNotFoundError(_REACTOR_SCALE_EDITABLE_MSG) from e
        raise
