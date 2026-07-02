from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")

from stellcoilbench.coil_optimization._scipy_optimizer import OptimizationHistoryRecorder


class _Objective:
    def __init__(self, value: float) -> None:
        self.value = value

    def J(self) -> float:
        return self.value


def test_history_recorder_writes_objective_and_constraint_csv(tmp_path: Path) -> None:
    recorder = OptimizationHistoryRecorder(
        tmp_path,
        2,
        constraint_names_and_thresholds=[("flux", 1e-8), ("length", 6.0)],
        weights=[1.0, 5.0],
    )

    assert recorder.should_record(1)
    assert not recorder.should_record(3)
    assert recorder.should_record(4)

    recorder.record(1, 3.5, [_Objective(2.0), _Objective(0.4)])

    with (tmp_path / "objective_history.csv").open(newline="", encoding="utf-8") as f:
        objective_rows = list(csv.DictReader(f))
    with (tmp_path / "constraint_history.csv").open(newline="", encoding="utf-8") as f:
        constraint_rows = list(csv.DictReader(f))

    assert objective_rows[0]["iteration"] == "1"
    assert objective_rows[0]["objective"] == "3.5"
    assert constraint_rows[1]["constraint_name"] == "length"
    assert constraint_rows[1]["weighted_value"] == "2.0"
