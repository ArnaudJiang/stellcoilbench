"""Coil-surface clearance guard for optimization."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


class CoilSurfaceDistanceGuard:
    """Penalize and optionally roll back unsafe coil-surface clearance.

    ``CurveSurfaceDistance.J()`` is a thresholded objective, while
    ``shortest_distance()`` is the engineering clearance metric reported in
    results. This guard audits the latter during optimization.
    """

    def __init__(
        self,
        Jcsdist: Any,
        *,
        output_dir: str | Path | None = None,
        interval: int = 5,
        hard_min: float = 0.0,
        soft_min: float | None = None,
        penalty: float = 1e8,
        rollback: bool = True,
    ) -> None:
        self.Jcsdist = Jcsdist
        self.interval = max(1, int(interval))
        self.hard_min = float(hard_min)
        self.soft_min = float(soft_min if soft_min is not None else hard_min)
        self.penalty = float(penalty)
        self.rollback = bool(rollback)
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.history_path = (
            self.output_dir / "cs_guard_history.csv" if self.output_dir else None
        )
        self.final_path = self.output_dir / "cs_guard_final.json" if self.output_dir else None
        self._initialized_history = False
        self.last_safe_x: np.ndarray | None = None
        self.last_safe_iteration: int | None = None
        self.last_safe_objective: float | None = None
        self.min_seen_distance: float | None = None
        self.violation_count = 0
        self.triggered = False

    def shortest_distance(self) -> float | None:
        try:
            return float(self.Jcsdist.shortest_distance())
        except Exception:
            return None

    def evaluate(
        self,
        iteration: int,
        *,
        x: np.ndarray | None = None,
        objective: float | None = None,
    ) -> float:
        """Return clearance penalty for the current curve state."""
        if iteration != 1 and iteration % self.interval != 0:
            return 0.0

        distance = self.shortest_distance()
        if distance is None:
            return 0.0
        self.min_seen_distance = (
            distance
            if self.min_seen_distance is None
            else min(self.min_seen_distance, distance)
        )
        unsafe = distance < self.hard_min
        if unsafe:
            self.violation_count += 1
            self.triggered = True
        else:
            if x is not None:
                self.last_safe_x = np.array(x, dtype=float, copy=True)
            self.last_safe_iteration = int(iteration)
            self.last_safe_objective = None if objective is None else float(objective)

        penalty = 0.0
        if distance < self.soft_min:
            span = max(self.soft_min - self.hard_min, 1e-9)
            normalized_gap = (self.soft_min - distance) / span
            penalty = self.penalty * float(normalized_gap**2)
            if unsafe:
                penalty += self.penalty
        self._record(iteration, distance, penalty, unsafe)
        return penalty

    def current_status(self) -> dict[str, Any]:
        distance = self.shortest_distance()
        unsafe = distance is not None and distance < self.hard_min
        return {
            "schema": "coil_surface_distance_guard_status_v1",
            "triggered": self.triggered,
            "violation_count": self.violation_count,
            "has_clearance_violation": bool(unsafe),
            "current_shortest_distance": distance,
            "min_seen_distance": self.min_seen_distance,
            "hard_min": self.hard_min,
            "soft_min": self.soft_min,
            "penalty": self.penalty,
            "interval": self.interval,
            "last_safe_iteration": self.last_safe_iteration,
            "last_safe_objective": self.last_safe_objective,
            "rollback": self.rollback,
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

    def _ensure_history(self) -> None:
        if self.history_path is None or self._initialized_history:
            return
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "iteration",
                    "shortest_distance",
                    "hard_min",
                    "soft_min",
                    "penalty",
                    "unsafe",
                ]
            )
        self._initialized_history = True

    def _record(
        self,
        iteration: int,
        distance: float,
        penalty: float,
        unsafe: bool,
    ) -> None:
        if self.history_path is None:
            return
        self._ensure_history()
        with self.history_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    int(iteration),
                    float(distance),
                    float(self.hard_min),
                    float(self.soft_min),
                    float(penalty),
                    bool(unsafe),
                ]
            )
