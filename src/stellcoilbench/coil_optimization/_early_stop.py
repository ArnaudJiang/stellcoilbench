"""Early rejection controller for expensive coil optimization scans."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class EarlyStopTriggered(RuntimeError):
    """Raised inside the objective when a configured early-stop rule fires."""

    def __init__(self, status: dict[str, Any]) -> None:
        self.status = status
        super().__init__(str(status.get("reason", "early stop triggered")))


@dataclass
class EarlyStopController:
    """Check cheap rejection rules during scipy objective evaluations."""

    config: dict[str, Any]
    base_curves: list
    Jccdist: Any
    Jcsdist: Any
    output_dir: str | Path | None = None
    link_guard: Any | None = None

    def __post_init__(self) -> None:
        self.enabled = bool(self.config.get("enabled", False))
        self.min_eval = int(self.config.get("min_eval", 150))
        self.check_interval = max(1, int(self.config.get("check_interval", 25)))
        self.sustained_bad_checks = max(
            1, int(self.config.get("sustained_bad_checks", 3))
        )
        self.objective_stall_window = int(
            self.config.get("objective_stall_window", 0) or 0
        )
        self.objective_min_relative_improvement = float(
            self.config.get("objective_min_relative_improvement", 0.0) or 0.0
        )
        self.output_dir = Path(self.output_dir) if self.output_dir else None
        self.history_path = (
            self.output_dir / "early_stop_history.csv" if self.output_dir else None
        )
        self.final_path = (
            self.output_dir / "early_stop_final.json" if self.output_dir else None
        )
        self._history_initialized = False
        self._cc_bad_count = 0
        self._cs_bad_count = 0
        self._objective_window_start: tuple[int, float] | None = None
        self.triggered = False
        self.status: dict[str, Any] = {
            "schema": "early_stop_status_v1",
            "enabled": self.enabled,
            "triggered": False,
            "reason": "",
            "iteration": None,
        }

    def maybe_check(self, iteration: int, objective: float) -> None:
        if not self.enabled:
            return
        if iteration < self.min_eval:
            return
        if iteration % self.check_interval != 0:
            return

        metrics = self._metrics(iteration, objective)
        reason = self._reason(metrics)
        self._record(metrics, reason)
        if reason:
            self.triggered = True
            self.status = {
                "schema": "early_stop_status_v1",
                "enabled": True,
                "triggered": True,
                "reason": reason,
                **metrics,
            }
            self.write_final()
            raise EarlyStopTriggered(self.status)

    def _metrics(self, iteration: int, objective: float) -> dict[str, Any]:
        cc = self._safe_shortest_distance(self.Jccdist)
        cs = self._safe_shortest_distance(self.Jcsdist)
        max_curvature, msc, max_torsion = self._geometry_metrics()
        link_violations = None
        if self.link_guard is not None:
            try:
                link_violations = int(
                    self.link_guard.current_status().get("violation_count", 0)
                )
            except Exception:
                link_violations = None
        return {
            "iteration": int(iteration),
            "objective": float(objective),
            "cc_shortest_distance": cc,
            "cs_shortest_distance": cs,
            "max_curvature": max_curvature,
            "mean_squared_curvature": msc,
            "max_torsion": max_torsion,
            "link_guard_violations": link_violations,
        }

    def _reason(self, metrics: dict[str, Any]) -> str:
        hard_min_cc = self._optional_float("hard_min_cc")
        hard_min_cs = self._optional_float("hard_min_cs")
        if hard_min_cc is not None and self._lt(metrics["cc_shortest_distance"], hard_min_cc):
            self._cc_bad_count += 1
        else:
            self._cc_bad_count = 0
        if hard_min_cs is not None and self._lt(metrics["cs_shortest_distance"], hard_min_cs):
            self._cs_bad_count += 1
        else:
            self._cs_bad_count = 0
        if self._cc_bad_count >= self.sustained_bad_checks:
            return (
                f"cc_shortest_distance<{hard_min_cc} for "
                f"{self._cc_bad_count} checks"
            )
        if self._cs_bad_count >= self.sustained_bad_checks:
            return (
                f"cs_shortest_distance<{hard_min_cs} for "
                f"{self._cs_bad_count} checks"
            )

        for key, metric_key in (
            ("max_curvature_abort", "max_curvature"),
            ("max_msc_abort", "mean_squared_curvature"),
            ("max_torsion_abort", "max_torsion"),
        ):
            limit = self._optional_float(key)
            value = metrics[metric_key]
            if limit is not None and value is not None and float(value) > limit:
                return f"{metric_key}>{limit}"

        max_link = self.config.get("max_link_guard_violations")
        if max_link is not None and metrics["link_guard_violations"] is not None:
            if int(metrics["link_guard_violations"]) > int(max_link):
                return f"link_guard_violations>{int(max_link)}"

        if self._objective_stalled(metrics["iteration"], metrics["objective"]):
            return (
                "objective_stall:"
                f"window={self.objective_stall_window},"
                f"min_relative_improvement={self.objective_min_relative_improvement}"
            )
        return ""

    def _objective_stalled(self, iteration: int, objective: float) -> bool:
        if self.objective_stall_window <= 0:
            return False
        if self.objective_min_relative_improvement <= 0:
            return False
        if self._objective_window_start is None:
            self._objective_window_start = (iteration, objective)
            return False
        start_iteration, start_objective = self._objective_window_start
        if iteration - start_iteration < self.objective_stall_window:
            return False
        denom = max(abs(start_objective), 1.0)
        improvement = (start_objective - objective) / denom
        self._objective_window_start = (iteration, min(objective, start_objective))
        return improvement < self.objective_min_relative_improvement

    def _geometry_metrics(self) -> tuple[float | None, float | None, float | None]:
        try:
            kappas = [np.atleast_1d(c.kappa()).astype(float) for c in self.base_curves]
            max_curvature = float(max(np.max(k) for k in kappas))
            msc = float(max(np.mean(k**2) for k in kappas))
        except Exception:
            max_curvature = None
            msc = None
        try:
            torsions = [
                np.atleast_1d(c.torsion()).astype(float) for c in self.base_curves
            ]
            max_torsion = float(max(np.max(np.abs(t)) for t in torsions))
        except Exception:
            max_torsion = None
        return max_curvature, msc, max_torsion

    @staticmethod
    def _safe_shortest_distance(obj: Any) -> float | None:
        try:
            return float(obj.shortest_distance())
        except Exception:
            return None

    def _optional_float(self, key: str) -> float | None:
        value = self.config.get(key)
        return None if value is None else float(value)

    @staticmethod
    def _lt(value: float | None, limit: float) -> bool:
        return value is not None and float(value) < limit

    def _ensure_history(self) -> None:
        if self._history_initialized or self.history_path is None:
            return
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "iteration",
                    "objective",
                    "cc_shortest_distance",
                    "cs_shortest_distance",
                    "max_curvature",
                    "mean_squared_curvature",
                    "max_torsion",
                    "link_guard_violations",
                    "reason",
                ],
            )
            writer.writeheader()
        self._history_initialized = True

    def _record(self, metrics: dict[str, Any], reason: str) -> None:
        if self.history_path is None:
            return
        self._ensure_history()
        with self.history_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "iteration",
                    "objective",
                    "cc_shortest_distance",
                    "cs_shortest_distance",
                    "max_curvature",
                    "mean_squared_curvature",
                    "max_torsion",
                    "link_guard_violations",
                    "reason",
                ],
            )
            writer.writerow({**metrics, "reason": reason})

    def write_final(self) -> None:
        if self.final_path is None:
            return
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        status = dict(self.status)
        status.setdefault("schema", "early_stop_status_v1")
        status.setdefault("enabled", self.enabled)
        status.setdefault("triggered", self.triggered)
        self.final_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
