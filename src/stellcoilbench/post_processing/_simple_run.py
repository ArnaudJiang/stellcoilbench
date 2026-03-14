"""SIMPLE subprocess execution.

Runs the ``simple.x`` executable in a subprocess with appropriate environment
setup (OpenMP, NetCDF library path, MPI variable cleanup).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

from ..mpi_utils import (
    comm_world,
    is_mpi_enabled,
    proc0_print,
    proc0_try,
    proc0_warning,
)


def _run_simple_subprocess(
    simple_executable_path: Path,
    output_dir: Path,
    timeout: int,
) -> Optional[subprocess.CompletedProcess]:
    """Execute ``simple.x`` in *output_dir* and return the result.

    Parameters
    ----------
    simple_executable_path : Path
        Absolute path to the ``simple.x`` executable.
    output_dir : Path
        Working directory (must contain ``simple.in``).
    timeout : int
        Subprocess timeout in seconds.

    Returns
    -------
    subprocess.CompletedProcess or None
        The completed process, or ``None`` on timeout / OS error.
    """
    env = os.environ.copy()

    # Ensure simple.x finds libnetcdff at runtime (it was built linked against NetCDF Fortran).
    # Prepend the current Python env's lib dir so the dynamic linker can locate the dylib.
    try:
        import sys

        lib_dir = Path(sys.prefix) / "lib"
        if lib_dir.exists():
            key = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
            existing = env.get(key, "")
            env[key] = f"{lib_dir}:{existing}" if existing else str(lib_dir)
    except Exception:
        pass  # Best-effort; avoid breaking the run if sys.prefix unavailable

    mpi_vars_to_remove = [
        k
        for k in env.keys()
        if k.startswith(("OMPI_", "PMIX_", "MPI_", "MPICH_", "I_MPI_", "SLURM_"))
    ]
    for var in mpi_vars_to_remove:
        del env[var]

    if "SIMPLE_NUM_THREADS" in os.environ:
        nthreads = int(os.environ["SIMPLE_NUM_THREADS"])
        thread_source = "SIMPLE_NUM_THREADS env var"
    elif is_mpi_enabled():
        nthreads = comm_world.size
        thread_source = "MPI world size"
    else:
        available = os.cpu_count() or 4
        nthreads = max(1, available // 2)
        thread_source = "available cores // 2"

    env["OMP_NUM_THREADS"] = str(nthreads)
    env["OMP_STACKSIZE"] = "64M"
    env["OMP_PROC_BIND"] = "false"
    env["OMP_PLACES"] = "threads"
    env["MKL_NUM_THREADS"] = str(nthreads)
    env["OPENBLAS_NUM_THREADS"] = str(nthreads)

    proc0_print(f"  OpenMP threads: {nthreads} ({thread_source})")

    result = None
    try:
        with proc0_try("Error running simple.x: {e}", default=None):
            result = subprocess.run(
                [str(simple_executable_path)],
                cwd=str(output_dir),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode != 0:
                proc0_warning(f"simple.x exited with code {result.returncode}")
                stdout_str = result.stdout or ""
                stderr_str = result.stderr or ""
                if "Phi period of 1" in stdout_str or "Phi period of 1" in stderr_str:
                    proc0_print(
                        "  Error: SIMPLE does not support configurations with nfp=1 (phi period of 1)."
                    )
                    proc0_print(
                        "  This limitation affects both tokamaks and nfp=1 stellarators."
                    )
                    proc0_print("  SIMPLE requires configurations with nfp > 1.")
                else:
                    proc0_print(
                        f"  stdout: {stdout_str[-500:] if stdout_str else '(empty)'}"
                    )
                    proc0_print(
                        f"  stderr: {stderr_str[-500:] if stderr_str else '(empty)'}"
                    )
                return None

            proc0_print("simple.x completed successfully")
            if result.stdout:
                proc0_print(f"  stdout: {result.stdout[-500:]}")
            return result

    except subprocess.TimeoutExpired:
        proc0_warning(f"simple.x timed out after {timeout}s")
        return None
    return result
