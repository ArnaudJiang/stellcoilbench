"""Tests for structural dJ MPI worker loop."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from stellcoilbench.coil_optimization._structural_mpi_worker import (
    _structural_dj_worker_loop,
)


class TestStructuralDjWorkerLoop:
    """Tests for _structural_dj_worker_loop."""

    def test_worker_rank_zero_returns_immediately(self) -> None:
        """Rank 0 returns without calling Bcast."""
        mock_comm = MagicMock()
        mock_comm.rank = 0
        mock_comm.size = 4
        mock_obj = MagicMock()

        with patch(
            "stellcoilbench.coil_optimization._structural_mpi_worker.comm_world",
            mock_comm,
        ):
            _structural_dj_worker_loop(mock_obj)

        mock_comm.Bcast.assert_not_called()
        mock_obj._collective_dj_body.assert_not_called()

    def test_worker_loop_receives_exit_tag(self) -> None:
        """Worker exits when Bcast receives tag=0."""
        mock_comm = MagicMock()
        mock_comm.rank = 1
        mock_comm.size = 4

        def bcast_side_effect(buf, root=0):
            buf[0] = 0
            buf[1] = 0

        mock_comm.Bcast = MagicMock(side_effect=bcast_side_effect)
        mock_obj = MagicMock()
        mock_obj._use_cached_K = False

        with patch(
            "stellcoilbench.coil_optimization._structural_mpi_worker.comm_world",
            mock_comm,
        ):
            _structural_dj_worker_loop(mock_obj)

        mock_comm.Bcast.assert_called_once()
        mock_obj._collective_dj_body.assert_not_called()
