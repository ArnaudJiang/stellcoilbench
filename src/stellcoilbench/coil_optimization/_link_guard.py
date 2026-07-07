"""Lightweight coil-coil topology guard for optimization."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _closed_points(curve: Any, *, sample_stride: int = 1) -> np.ndarray:
    pts = np.asarray(curve.gamma(), dtype=float).reshape(-1, 3)
    stride = max(1, int(sample_stride))
    if stride > 1 and len(pts) > 0:
        pts = pts[::stride]
    if len(pts) == 0:
        return pts
    if np.linalg.norm(pts[0] - pts[-1]) > 1e-10:
        pts = np.vstack([pts, pts[0]])
    return pts


def _pairwise_gauss_link(
    curve_a: Any,
    curve_b: Any,
    *,
    sample_stride: int = 1,
) -> float:
    p = _closed_points(curve_a, sample_stride=sample_stride)
    q = _closed_points(curve_b, sample_stride=sample_stride)
    if len(p) < 2 or len(q) < 2:
        return 0.0

    p0, p1 = p[:-1], p[1:]
    q0, q1 = q[:-1], q[1:]
    dp = p1 - p0
    dq = q1 - q0
    pm = 0.5 * (p0 + p1)
    qm = 0.5 * (q0 + q1)

    total = 0.0
    for point, tangent in zip(pm, dp):
        rel = point - qm
        cross = np.cross(tangent, dq)
        num = np.einsum("ij,ij->i", rel, cross)
        den = np.linalg.norm(rel, axis=1) ** 3
        mask = den > 1e-14
        total += float(np.sum(num[mask] / den[mask]))
    return float(total / (4.0 * np.pi))


class PairwiseLinkGuard:
    """Detect and penalize coil-coil topology changes during optimization.

    The guard compares rounded pairwise Gauss linking numbers against their
    initial values. It is intended as a coarse hard guard: when a pair changes
    topology, the objective receives a large penalty and the event is logged.
    """

    def __init__(
        self,
        curves: list[Any],
        *,
        output_dir: str | Path | None = None,
        interval: int = 1,
        penalty: float = 1e12,
        tolerance: float = 0.5,
        rollback: bool = True,
        sample_stride: int = 1,
        record_interval: int | None = None,
    ) -> None:
        self.curves = curves
        self.interval = max(1, int(interval))
        self.penalty = float(penalty)
        self.tolerance = float(tolerance)
        self.rollback = bool(rollback)
        self.sample_stride = max(1, int(sample_stride))
        self.record_interval = (
            self.interval if record_interval is None else max(1, int(record_interval))
        )
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.history_path = (
            self.output_dir / "link_guard_history.csv" if self.output_dir else None
        )
        self.audit_path = (
            self.output_dir / "link_guard_initial.json" if self.output_dir else None
        )
        self.final_path = (
            self.output_dir / "link_guard_final.json" if self.output_dir else None
        )
        self._initialized_history = False
        self.initial_matrix = self._compute_matrix()
        self.initial_rounded = np.rint(self.initial_matrix).astype(int)
        self.last_safe_x: np.ndarray | None = None
        self.last_safe_iteration: int | None = None
        self.last_safe_objective: float | None = None
        self.violation_count = 0
        self.triggered = False
        self._write_initial_audit()

    def _compute_matrix(self) -> np.ndarray:
        n = len(self.curves)
        matrix = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                val = _pairwise_gauss_link(
                    self.curves[i],
                    self.curves[j],
                    sample_stride=self.sample_stride,
                )
                matrix[i, j] = val
                matrix[j, i] = val
        return matrix

    def _write_initial_audit(self) -> None:
        if self.audit_path is None:
            return
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "pairwise_link_guard_initial_v1",
            "num_curves": len(self.curves),
            "initial_pairwise_gauss_matrix": self.initial_matrix.tolist(),
            "initial_pairwise_rounded_matrix": self.initial_rounded.tolist(),
            "tolerance": self.tolerance,
            "penalty": self.penalty,
            "interval": self.interval,
            "rollback": self.rollback,
            "sample_stride": self.sample_stride,
            "record_interval": self.record_interval,
        }
        self.audit_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _ensure_history(self) -> None:
        if self.history_path is None or self._initialized_history:
            return
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "iteration",
                    "penalty",
                    "max_abs_delta",
                    "changed_pair_count",
                    "changed_pairs_json",
                ]
            )
        self._initialized_history = True

    def evaluate(
        self,
        iteration: int,
        *,
        x: np.ndarray | None = None,
        objective: float | None = None,
    ) -> float:
        """Return topology-change penalty for the current curve state."""
        if iteration != 1 and iteration % self.interval != 0:
            return 0.0

        matrix = self._compute_matrix()
        changed, max_abs_delta = self._changed_pairs(matrix)
        penalty = self.penalty * float(len(changed)) if changed else 0.0
        if changed:
            self.violation_count += 1
            self.triggered = True
        else:
            if x is not None:
                self.last_safe_x = np.array(x, dtype=float, copy=True)
            self.last_safe_iteration = int(iteration)
            self.last_safe_objective = None if objective is None else float(objective)
        if penalty or iteration == 1 or iteration % self.record_interval == 0:
            self._record(iteration, penalty, max_abs_delta, changed)
        return penalty

    def _changed_pairs(self, matrix: np.ndarray) -> tuple[list[dict[str, Any]], float]:
        rounded = np.rint(matrix).astype(int)
        delta = rounded - self.initial_rounded
        changed: list[dict[str, Any]] = []
        n = len(self.curves)
        max_abs_delta = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                raw_delta = float(matrix[i, j] - self.initial_matrix[i, j])
                max_abs_delta = max(max_abs_delta, abs(raw_delta))
                if abs(raw_delta) >= self.tolerance or delta[i, j] != 0:
                    changed.append(
                        {
                            "i": i,
                            "j": j,
                            "initial": float(self.initial_matrix[i, j]),
                            "current": float(matrix[i, j]),
                            "initial_rounded": int(self.initial_rounded[i, j]),
                            "current_rounded": int(rounded[i, j]),
                        }
                    )
        return changed, max_abs_delta

    def current_status(self) -> dict[str, Any]:
        matrix = self._compute_matrix()
        changed, max_abs_delta = self._changed_pairs(matrix)
        return {
            "schema": "pairwise_link_guard_status_v1",
            "triggered": self.triggered,
            "violation_count": self.violation_count,
            "has_topology_change": bool(changed),
            "changed_pair_count": len(changed),
            "changed_pairs": changed,
            "max_abs_delta": max_abs_delta,
            "current_pairwise_gauss_matrix": matrix.tolist(),
            "initial_pairwise_gauss_matrix": self.initial_matrix.tolist(),
            "initial_pairwise_rounded_matrix": self.initial_rounded.tolist(),
            "last_safe_iteration": self.last_safe_iteration,
            "last_safe_objective": self.last_safe_objective,
            "rollback": self.rollback,
            "sample_stride": self.sample_stride,
            "record_interval": self.record_interval,
        }

    def restore_last_safe(self, target: Any) -> bool:
        """Restore *target.x* to the last safe point when rollback is enabled."""
        if not self.rollback or self.last_safe_x is None:
            return False
        target.x = self.last_safe_x.copy()
        return True

    def write_final_audit(self, *, restored: bool) -> None:
        if self.final_path is None:
            return
        payload = self.current_status()
        payload["restored_last_safe"] = bool(restored)
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        self.final_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _record(
        self,
        iteration: int,
        penalty: float,
        max_abs_delta: float,
        changed: list[dict[str, Any]],
    ) -> None:
        if self.history_path is None:
            return
        self._ensure_history()
        with self.history_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    int(iteration),
                    float(penalty),
                    float(max_abs_delta),
                    len(changed),
                    json.dumps(changed, separators=(",", ":")),
                ]
            )
