"""Progress monitor for batch optimization result directories."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import typer


@dataclass(frozen=True)
class ProgressThresholds:
    """Engineering thresholds used for live progress summaries."""

    cc_min: float = 0.25
    cs_min: float = 0.25
    curvature_max: float = 5.0
    torsion_max: float | None = None
    length_ratio_max: float | None = None


@dataclass(frozen=True)
class ProgressSummary:
    """Aggregated status for a result directory."""

    results_dir: Path
    expected: int | None
    records_found: int
    malformed_records: int
    success_count: int
    failed_count: int
    cc_pass_count: int
    cs_pass_count: int
    curvature_pass_count: int
    link_clean_count: int
    hard_feasible_count: int
    best_hard: dict[str, Any] | None
    best_overall: dict[str, Any] | None
    newest_record_mtime: float | None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric(record: dict[str, Any], *names: str) -> Any:
    metrics = record.get("metrics")
    for name in names:
        if name in record and record[name] is not None:
            return record[name]
        if isinstance(metrics, dict) and name in metrics and metrics[name] is not None:
            return metrics[name]
    return None


def _record_paths(results_dir: Path) -> list[Path]:
    if not results_dir.exists():
        return []
    return sorted(results_dir.glob("runs/*/record.json"))


def _read_records(results_dir: Path) -> tuple[list[tuple[Path, dict[str, Any]]], int]:
    records: list[tuple[Path, dict[str, Any]]] = []
    malformed = 0
    for path in _record_paths(results_dir):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            malformed += 1
            continue
        if isinstance(data, dict):
            records.append((path, data))
        else:
            malformed += 1
    return records, malformed


def infer_expected_from_manifest(manifest_path: Path | None) -> int | None:
    """Infer planned job count from a CSV manifest."""

    if manifest_path is None or not manifest_path.exists():
        return None
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def summarize_progress(
    results_dir: Path,
    *,
    expected: int | None = None,
    thresholds: ProgressThresholds | None = None,
) -> ProgressSummary:
    """Summarize record.json progress for a batch optimization directory."""

    thresholds = thresholds or ProgressThresholds()
    records, malformed = _read_records(results_dir)
    success_count = 0
    failed_count = 0
    cc_pass_count = 0
    cs_pass_count = 0
    curvature_pass_count = 0
    link_clean_count = 0
    hard_feasible: list[dict[str, Any]] = []
    overall: list[dict[str, Any]] = []
    newest_mtime: float | None = None

    for path, record in records:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = None
        if mtime is not None:
            newest_mtime = mtime if newest_mtime is None else max(newest_mtime, mtime)

        status = str(record.get("status") or "").lower()
        failure_reason = record.get("failure_reason")
        if record.get("success") is True:
            success_count += 1
        if status == "failed" or bool(failure_reason):
            failed_count += 1

        avg = _safe_float(_metric(record, "avg_BdotN_over_B", "avg_Bn_over_B"))
        avg_target = _safe_float(_metric(record, "avg_BdotN_over_target_B"))
        cc = _safe_float(_metric(record, "final_min_cc_separation", "min_cc"))
        cs = _safe_float(_metric(record, "final_min_cs_separation", "min_cs"))
        curvature = _safe_float(
            _metric(record, "final_max_curvature", "max_curvature")
        )
        torsion = _safe_float(_metric(record, "final_max_torsion", "max_torsion"))
        length_ratio = _safe_float(
            _metric(record, "final_length_ratio", "length_ratio")
        )
        linking = _safe_float(_metric(record, "final_linking_number", "linking_number"))
        run_id = str(record.get("run_id") or path.parent.name)

        cc_ok = cc is not None and cc >= thresholds.cc_min
        cs_ok = cs is not None and cs >= thresholds.cs_min
        curvature_ok = curvature is not None and curvature <= thresholds.curvature_max
        link_ok = linking is not None and linking == 0.0
        torsion_ok = (
            thresholds.torsion_max is None
            or (torsion is not None and torsion <= thresholds.torsion_max)
        )
        length_ratio_ok = (
            thresholds.length_ratio_max is None
            or (
                length_ratio is not None
                and length_ratio <= thresholds.length_ratio_max
            )
        )

        cc_pass_count += int(cc_ok)
        cs_pass_count += int(cs_ok)
        curvature_pass_count += int(curvature_ok)
        link_clean_count += int(link_ok)

        row = {
            "run_id": run_id,
            "avg_BdotN_over_B": avg,
            "avg_BdotN_over_target_B": avg_target,
            "cc": cc,
            "cs": cs,
            "max_curvature": curvature,
            "max_torsion": torsion,
            "length_ratio": length_ratio,
            "linking_number": linking,
            "path": str(path),
        }
        overall.append(row)
        if cc_ok and cs_ok and curvature_ok and link_ok and torsion_ok and length_ratio_ok:
            hard_feasible.append(row)

    def sort_key(row: dict[str, Any]) -> tuple[bool, float]:
        avg = row.get("avg_BdotN_over_target_B")
        if avg is None:
            avg = row.get("avg_BdotN_over_B")
        return avg is None, float(avg) if avg is not None else 999.0

    return ProgressSummary(
        results_dir=results_dir,
        expected=expected,
        records_found=len(records),
        malformed_records=malformed,
        success_count=success_count,
        failed_count=failed_count,
        cc_pass_count=cc_pass_count,
        cs_pass_count=cs_pass_count,
        curvature_pass_count=curvature_pass_count,
        link_clean_count=link_clean_count,
        hard_feasible_count=len(hard_feasible),
        best_hard=sorted(hard_feasible, key=sort_key)[0] if hard_feasible else None,
        best_overall=sorted(overall, key=sort_key)[0] if overall else None,
        newest_record_mtime=newest_mtime,
    )


def _bar(done: int, expected: int | None, *, width: int = 32) -> str:
    if expected is None or expected <= 0:
        return f"{done} records"
    ratio = max(0.0, min(1.0, done / expected))
    filled = int(round(width * ratio))
    return f"[{'#' * filled}{'.' * (width - filled)}] {done}/{expected} {ratio * 100:5.1f}%"


def _fmt_float(value: Any, digits: int = 6) -> str:
    number = _safe_float(value)
    if number is None:
        return "n/a"
    return f"{number:.{digits}f}"


def format_progress(summary: ProgressSummary) -> str:
    """Render a human-readable progress summary."""

    lines = [
        f"Results: {summary.results_dir}",
        f"Progress: {_bar(summary.records_found, summary.expected)}",
        (
            "Records: "
            f"success={summary.success_count}, failed={summary.failed_count}, "
            f"malformed={summary.malformed_records}"
        ),
        (
            "Engineering gates: "
            f"cc={summary.cc_pass_count}, cs={summary.cs_pass_count}, "
            f"curv={summary.curvature_pass_count}, link_clean={summary.link_clean_count}, "
            f"hard_feasible={summary.hard_feasible_count}"
        ),
    ]
    if summary.best_hard:
        row = summary.best_hard
        lines.append(
            "Best hard: "
            f"{row['run_id']} avg={_fmt_float(row['avg_BdotN_over_B'])} "
            f"avgT={_fmt_float(row['avg_BdotN_over_target_B'])} "
            f"cc={_fmt_float(row['cc'], 4)} cs={_fmt_float(row['cs'], 4)} "
            f"curv={_fmt_float(row['max_curvature'], 4)} "
            f"tors={_fmt_float(row['max_torsion'], 4)} "
            f"lr={_fmt_float(row['length_ratio'], 3)}"
        )
    elif summary.best_overall:
        row = summary.best_overall
        lines.append(
            "Best overall: "
            f"{row['run_id']} avg={_fmt_float(row['avg_BdotN_over_B'])} "
            f"avgT={_fmt_float(row['avg_BdotN_over_target_B'])} "
            f"cc={_fmt_float(row['cc'], 4)} cs={_fmt_float(row['cs'], 4)} "
            f"curv={_fmt_float(row['max_curvature'], 4)}"
        )
    if summary.newest_record_mtime is not None:
        stamp = datetime.fromtimestamp(summary.newest_record_mtime).isoformat(
            timespec="seconds"
        )
        lines.append(f"Newest record: {stamp}")
    lines.append(f"Updated: {datetime.now().isoformat(timespec='seconds')}")
    return "\n".join(lines)


def monitor_progress_cmd(
    results_dir: Path = typer.Argument(
        ...,
        help="Result root containing runs/*/record.json.",
    ),
    expected: int | None = typer.Option(
        None,
        "--expected",
        "-n",
        min=1,
        help="Expected number of records/jobs.",
    ),
    manifest: Path | None = typer.Option(
        None,
        "--manifest",
        help="Optional CSV manifest used to infer expected jobs.",
    ),
    watch: bool = typer.Option(
        False,
        "--watch",
        "-w",
        help="Refresh in-place until interrupted.",
    ),
    interval: float = typer.Option(
        30.0,
        "--interval",
        min=1.0,
        help="Refresh interval in seconds when --watch is enabled.",
    ),
    cc_min: float = typer.Option(0.25, help="Minimum coil-coil clearance."),
    cs_min: float = typer.Option(0.25, help="Minimum coil-surface clearance."),
    curvature_max: float = typer.Option(5.0, help="Maximum allowed curvature."),
    torsion_max: float | None = typer.Option(
        None,
        help="Optional maximum torsion hard gate.",
    ),
    length_ratio_max: float | None = typer.Option(
        None,
        help="Optional maximum length ratio hard gate.",
    ),
) -> None:
    """Show a live progress bar for optimization result directories."""

    inferred_expected = expected or infer_expected_from_manifest(manifest)
    thresholds = ProgressThresholds(
        cc_min=cc_min,
        cs_min=cs_min,
        curvature_max=curvature_max,
        torsion_max=torsion_max,
        length_ratio_max=length_ratio_max,
    )

    while True:
        summary = summarize_progress(
            results_dir,
            expected=inferred_expected,
            thresholds=thresholds,
        )
        if watch:
            typer.echo("\033[2J\033[H", nl=False)
        typer.echo(format_progress(summary))
        if not watch:
            return
        time.sleep(interval)


def register(app: typer.Typer) -> None:
    """Register the monitor-progress command with the Typer app."""

    app.command("monitor-progress")(monitor_progress_cmd)


def main() -> None:
    """Run the lightweight monitor as ``python -m stellcoilbench.monitor_progress``."""

    typer.run(monitor_progress_cmd)


if __name__ == "__main__":
    main()
