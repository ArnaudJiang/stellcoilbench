"""
Shared MPI utilities for StellCoilBench.

Provides comm_world, proc0_print, and is_proc0 for rank-aware operations.
Handles systems without MPI (e.g., ReadTheDocs) via fallbacks.
"""

from __future__ import annotations

from contextlib import contextmanager
import inspect
from typing import Any, Callable

try:
    from simsopt.util import comm_world, proc0_print
except (ImportError, OSError, RuntimeError):
    # ImportError: simsopt not installed
    # OSError: cannot open shared object file (e.g. libmpi.so on ReadTheDocs)
    # RuntimeError: mpi4py installed but MPI library not available
    comm_world = None  # type: ignore

    def proc0_print(*args: Any, **kwargs: Any) -> None:
        """Fallback when MPI unavailable: always print."""
        print(*args, **kwargs)


def proc0_warning(msg: str) -> None:
    """Print a warning message on rank 0 only.

    Standardises the ``Warning: …`` prefix across the codebase so that
    warning messages are easy to grep for and consistently formatted.

    Parameters
    ----------
    msg : str
        Warning text (the ``Warning: `` prefix is added automatically).
    """
    proc0_print(f"Warning: {msg}")


_DEFAULT_EXC_TYPES: tuple[type[BaseException], ...] = (
    OSError,
    RuntimeError,
    ValueError,
)


@contextmanager
def proc0_try(
    msg: str,
    *exc_types: type[BaseException],
    default: Any = None,
    reraise: bool = False,
    on_catch: Callable[[], None] | None = None,
):
    """Context manager that catches specified exceptions, logs via proc0_warning, and either suppresses or re-raises.

    Use this for the common pattern of catching OSError/RuntimeError/ValueError,
    logging a warning, and continuing with a default or re-raising.

    Parameters
    ----------
    msg : str
        Warning message template; use ``{e}`` for the exception (e.g. ``"X failed: {e}"``).
    *exc_types : type
        Exception types to catch. Defaults to ``(OSError, RuntimeError, ValueError)`` if empty.
    default : Any, optional
        Not used by the context manager; the caller should initialize their result
        variable to this value before the ``with`` block so it is used when an exception is caught.
    reraise : bool, optional
        If True, call proc0_warning and then re-raise. If False, suppress the exception.
    on_catch : callable, optional
        If provided, called after proc0_warning when an exception is caught (e.g. for proc0_print).
        May accept zero arguments or one (the caught exception instance).

    Yields
    ------
    None
        The context manager does not yield a value; the caller manages their result variable.

    Examples
    --------
    >>> result = None
    >>> with proc0_try("Error running tool: {e}", default=None):
    ...     result = do_something()
    >>> return result

    >>> with proc0_try("Failed: {e}", reraise=True):
    ...     risky_operation()
    """
    types = exc_types if exc_types else _DEFAULT_EXC_TYPES
    try:
        yield
    except types as e:
        proc0_warning(msg.format(e=e))
        if on_catch is not None:
            sig = inspect.signature(on_catch)
            if len(sig.parameters) >= 1:
                on_catch(e)
            else:
                on_catch()
        if reraise:
            raise


def is_proc0() -> bool:
    """
    Check if this process is rank 0 (or non-MPI environment).

    Returns
    -------
    bool
        True if rank 0 or MPI unavailable; False otherwise.
    """
    return comm_world is None or not hasattr(comm_world, "rank") or comm_world.rank == 0


def is_mpi_enabled() -> bool:
    """Check whether MPI is available and running with more than one rank.

    Returns
    -------
    bool
        True if MPI is initialized with size > 1; False otherwise.
    """
    return (
        comm_world is not None and hasattr(comm_world, "size") and comm_world.size > 1
    )
