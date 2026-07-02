"""Campaign summary report generation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from data_twin.storage.jsonl_store import JsonlStore


def write_campaign_reports(campaign_root: Path | str, report_root: Path | str = "reports/data_twin") -> Path:
    root = Path(campaign_root)
    campaign_id = root.name
    out = Path(report_root) / campaign_id
    out.mkdir(parents=True, exist_ok=True)
    store = JsonlStore(root)
    cases = store.read("cases.jsonl")
    runs = store.read("runs.jsonl")
    metrics = store.read("metrics.jsonl")
    evaluations = store.read("evaluations.jsonl")
    decisions = store.read("decisions.jsonl")

    run_status = Counter(row.get("status", "unknown") for row in runs)
    failure_labels = Counter(label for row in evaluations for label in (row.get("failure_labels") or []))
    metric_avail = Counter()
    for row in metrics:
        metric_avail[row.get("metric_name", "unknown")] += int(bool(row.get("available")))

    (out / "campaign_summary.md").write_text(
        "\n".join([
            f"# Campaign Summary: {campaign_id}",
            "",
            f"- Cases: {len(cases)}",
            f"- Runs: {len(runs)}",
            f"- Metrics: {len(metrics)}",
            f"- Evaluations: {len(evaluations)}",
            f"- Decisions: {len(decisions)}",
            "",
            "## Run Status",
            *[f"- `{key}`: {value}" for key, value in run_status.items()],
            "",
        ]),
        encoding="utf-8",
    )
    (out / "failure_summary.md").write_text(
        "\n".join(["# Failure Summary", "", *[f"- `{k}`: {v}" for k, v in failure_labels.most_common() or [("none", 0)]]]),
        encoding="utf-8",
    )
    (out / "metric_availability.md").write_text(
        "\n".join(["# Metric Availability", "", *[f"- `{k}`: {v} available" for k, v in metric_avail.most_common()]]),
        encoding="utf-8",
    )
    (out / "data_quality_report.md").write_text(
        "\n".join(["# Data Quality Report", "", f"- Cases without runs: {max(0, len(cases) - len({r.get('case_id') for r in runs}))}", "- Missing metrics are represented by `available=false`."]),
        encoding="utf-8",
    )
    (out / "lineage_summary.md").write_text("# Lineage Summary\n\nLineage is available through `data_twin_cli.py lineage`.\n", encoding="utf-8")
    top_lines = ["# Top Cases", ""]
    for metric_name in ("final_squared_flux", "mean_abs_Bn", "max_curvature", "total_coil_length"):
        top = [m for m in metrics if m.get("metric_name") == metric_name and m.get("available")]
        top.sort(key=lambda m: float(m.get("metric_value")))
        top_lines.append(f"## {metric_name}")
        top_lines.extend(f"- `{m.get('case_id')}` / `{m.get('run_id')}`: {m.get('metric_value')}" for m in top[:10])
        top_lines.append("")
    (out / "top_cases.md").write_text("\n".join(top_lines), encoding="utf-8")
    return out
