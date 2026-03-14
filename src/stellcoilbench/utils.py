"""Shared utility functions for StellCoilBench.

Provides output suppression, timing helpers, and other small utilities
used across optimization, post-processing, and finite-build modules.

Extracted from ``post_processing.py`` to break circular import chains
(``finite_build`` and ``reactor_scale`` previously imported from
``post_processing``).
"""

from __future__ import annotations

import io
import os
import sys
import time
from contextlib import contextmanager
from typing import Dict, Generator

from .mpi_utils import proc0_print

# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------

_timing_results: Dict[str, float] = {}
"""Global dictionary storing section-level elapsed times."""


@contextmanager
def timed_section(name: str, print_time: bool = False) -> Generator[None, None, None]:
    """Context manager for timing code sections.

    Stores elapsed time in :data:`_timing_results` (accessible via
    :func:`get_timing_results`) and optionally prints it on exit.
    By default, timing is recorded silently; use :func:`print_timing_summary`
    to display all results at the end of a pipeline.

    Parameters
    ----------
    name : str
        Name of the section being timed.
    print_time : bool, default False
        Whether to print the elapsed time immediately on exit.

    Yields
    ------
    None
    """
    global _timing_results
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        _timing_results[name] = elapsed
        if print_time:
            proc0_print(f"[TIMING] {name}: {elapsed:.2f}s")


def get_timing_results() -> Dict[str, float]:
    """Return a copy of the timing results dictionary."""
    return _timing_results.copy()


def clear_timing_results() -> None:
    """Clear all timing results."""
    _timing_results.clear()


def print_timing_summary() -> None:
    """Print a formatted summary of all timing results."""
    global _timing_results
    if not _timing_results:
        proc0_print("No timing data recorded.")
        return

    proc0_print("\n" + "=" * 60)
    proc0_print("TIMING SUMMARY")
    proc0_print("=" * 60)

    sorted_times = sorted(_timing_results.items(), key=lambda x: x[1], reverse=True)
    total_time = sum(_timing_results.values())

    for name, elapsed in sorted_times:
        pct = (elapsed / total_time * 100) if total_time > 0 else 0
        proc0_print(f"  {name:40s} {elapsed:8.2f}s ({pct:5.1f}%)")

    proc0_print("-" * 60)
    proc0_print(f"  {'TOTAL':40s} {total_time:8.2f}s")
    proc0_print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Output suppression
# ---------------------------------------------------------------------------


@contextmanager
def suppress_output() -> Generator[None, None, None]:
    """Context manager to suppress stdout and stderr.

    Used when running VMEC, booz_xform, and other tools that print to
    console.  Redirects file descriptors to ``/dev/null`` for the duration
    of the block.  When ``fileno()`` is unavailable (e.g. pytest capsys),
    yields without redirecting.
    """
    try:
        stdout_fd = sys.stdout.fileno()
        stderr_fd = sys.stderr.fileno()
    except (io.UnsupportedOperation, OSError):
        yield
        return

    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)

    try:
        os.dup2(devnull, stdout_fd)
        os.dup2(devnull, stderr_fd)
        yield
    finally:
        os.dup2(saved_stdout_fd, stdout_fd)
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        os.close(devnull)
