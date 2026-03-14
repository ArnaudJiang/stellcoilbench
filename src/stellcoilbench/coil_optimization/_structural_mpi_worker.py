"""MPI worker loop for parallel structural dJ() computation.

When running with MPI and a structural stress objective, ranks 1..P-1 run this
loop and block on Bcast(control) from rank 0. When rank 0 computes the structural
gradient, it broadcasts control=1; workers then participate in the collective
FD gradient computation. When optimization finishes, rank 0 broadcasts control=0.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..mpi_utils import comm_world


def _structural_dj_worker_loop(structural_obj_raw: Any) -> None:
    """Run on ranks 1..P-1; block on control Bcast, participate in dJ when requested.

    Parameters
    ----------
    structural_obj_raw : StructuralStressObjective
        The unwrapped structural objective (has _collective_dj_body).
    """
    comm = comm_world
    if comm.rank == 0:
        return
    control = np.empty(2, dtype=np.int64)
    use_cached_K = structural_obj_raw._use_cached_K
    while True:
        comm.Bcast(control, root=0)
        tag = int(control[0])
        if tag == 0:
            break
        if tag == 2:
            continue
        if tag == 1:
            n = int(control[1])
            structural_obj_raw._collective_dj_body(use_cached_K=use_cached_K, n=n)
