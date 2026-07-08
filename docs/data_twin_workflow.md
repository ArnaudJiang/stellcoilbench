# Data Twin Collaboration Workflow

Data Twin uses append-only JSONL files as the source of truth and a local
SQLite index as a rebuildable query layer.

Default locations:

```text
experiments/data_twin/<campaign>/*.jsonl
experiments/data_twin/data_twin_index.sqlite
```

## Lifecycle

The intended optimization lifecycle is:

```text
draft -> preflight_passed -> registered -> launch_approved -> running
-> results_ready -> ingested -> screened -> reviewed -> next_plan_created -> closed
```

Current workflow commands write events and collaboration records. The SQLite
index can be rebuilt at any time from JSONL:

```bash
conda run -n stellcoilbench_vmec python scripts/optimization_workflow.py index rebuild
```

## Collaboration Records

Each campaign should have:

- `briefs.jsonl`: experiment intent, owner, collaborators, surface, hypothesis,
  success criteria, and known risks.
- `reviews.jsonl`: human review notes before launch or after screening.
- `decisions.jsonl`: continue, stop, refine, promote, or abandon decisions with
  evidence and next action.

## CLI

Status:

```bash
conda run -n stellcoilbench_vmec python scripts/optimization_workflow.py status \
  --campaign <campaign_id>
```

Review:

```bash
conda run -n stellcoilbench_vmec python scripts/optimization_workflow.py review \
  --campaign <campaign_id> \
  --review-status approved \
  --by <name> \
  --note "preflight checked"
```

Decision:

```bash
conda run -n stellcoilbench_vmec python scripts/optimization_workflow.py decide \
  --campaign <campaign_id> \
  --decision refine \
  --reason "coil-surface clearance is limiting" \
  --next-action "increase cs weight and rescan seeds"
```

Comparison:

```bash
conda run -n stellcoilbench_vmec python scripts/optimization_workflow.py compare \
  --campaign <campaign_a> \
  --campaign <campaign_b>
```
