# Data Twin Core Design

The Data Twin is the persistent memory layer for coil optimization campaigns. It records proposed cases, execution attempts, artifacts, metrics, evaluations, decisions, and lifecycle events. It does not launch optimizations or make autonomous proposals.

## Principles

- JSONL files are the source of truth.
- CSV files are exports for analysis and reports.
- Records are append-only.
- Missing metrics are explicit with `available=false` and `metric_value=null`.
- Failed and incomplete runs are first-class records.
- Campaign metadata, such as `n_coils`, belongs in `CampaignRecord.target_metadata`; core code does not hard-code n=4.

## Campaign Layout

```text
experiments/data_twin/<campaign_id>/
  campaign.yaml
  cases.jsonl
  runs.jsonl
  artifacts.jsonl
  metrics.jsonl
  evaluations.jsonl
  decisions.jsonl
  events.jsonl
  artifacts/<case_id>/<run_id>/
  exports/
```

## Client Model

Future proposers write `CaseRecord` objects with `proposal_source`, `proposal_reason`, `parameters`, and `constraints`.

Future runners write `RunRecord` objects and attach stdout/stderr/config artifacts.

Future evaluators read metrics and write `EvaluationRecord` and `DecisionRecord` objects.

The MVP provides CLI operations for init, validation, case/run addition, artifact attachment, CSV ingest, metric extraction, evaluation, export, reporting, and lineage.
