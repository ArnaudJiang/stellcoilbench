from __future__ import annotations

import json

from typer.testing import CliRunner

from stellcoilbench.cli import app
from stellcoilbench.monitor_progress import (
    ProgressThresholds,
    format_progress,
    summarize_progress,
)


runner = CliRunner()


def _write_record(root, run_id: str, **overrides) -> None:
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True)
    record = {
        "run_id": run_id,
        "status": "completed",
        "success": True,
        "avg_BdotN_over_B": 0.01,
        "final_min_cc_separation": 0.26,
        "final_min_cs_separation": 0.27,
        "final_max_curvature": 4.9,
        "final_max_torsion": 7.0,
        "final_length_ratio": 1.2,
        "final_linking_number": 0,
    }
    record.update(overrides)
    (run_dir / "record.json").write_text(json.dumps(record), encoding="utf-8")


def test_summarize_progress_counts_hard_feasible(tmp_path):
    _write_record(tmp_path, "good", avg_BdotN_over_B=0.006)
    _write_record(tmp_path, "bad_curvature", final_max_curvature=5.2)

    summary = summarize_progress(
        tmp_path,
        expected=4,
        thresholds=ProgressThresholds(cc_min=0.25, cs_min=0.25, curvature_max=5.0),
    )

    assert summary.records_found == 2
    assert summary.expected == 4
    assert summary.cc_pass_count == 2
    assert summary.cs_pass_count == 2
    assert summary.curvature_pass_count == 1
    assert summary.link_clean_count == 2
    assert summary.hard_feasible_count == 1
    assert summary.best_hard is not None
    assert summary.best_hard["run_id"] == "good"


def test_format_progress_includes_bar_and_best_case(tmp_path):
    _write_record(tmp_path, "good", avg_BdotN_over_B=0.006)
    summary = summarize_progress(tmp_path, expected=2)

    output = format_progress(summary)

    assert "1/2" in output
    assert "50.0%" in output
    assert "hard_feasible=1" in output
    assert "Best hard: good" in output


def test_monitor_progress_cli(tmp_path):
    _write_record(tmp_path, "good", avg_BdotN_over_B=0.006)

    result = runner.invoke(
        app,
        [
            "monitor-progress",
            str(tmp_path),
            "--expected",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert "Progress:" in result.output
    assert "1/2" in result.output
    assert "Best hard: good" in result.output
