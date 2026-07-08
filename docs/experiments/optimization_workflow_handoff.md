# Optimization Workflow Handoff

Use `scripts/optimization_workflow.py` as the stable entry point for new
optimization work. Do not launch long Simsopt scans by calling historical
runner scripts directly.

## Generic policy workflow

Generic policy scans go through the Data Twin gated launcher:

```bash
conda run -n stellcoilbench_vmec python scripts/optimization_workflow.py prepare \
  --campaign <campaign_id> \
  --campaign-config <campaign_yaml> \
  --policy <policy_yaml> \
  --results-dir <results_dir> \
  --backend simsopt \
  --expected <planned_jobs> \
  --surface-resolution <surface_grid> \
  --max-parallel-simsopt <workers>
```

Launch only after `prepare` succeeds:

```bash
conda run -n stellcoilbench_vmec python scripts/optimization_workflow.py launch \
  --campaign <campaign_id> \
  --policy <policy_yaml> \
  --results-dir <results_dir> \
  --backend simsopt \
  --expected <planned_jobs> \
  --surface-resolution <surface_grid> \
  --max-parallel-simsopt <workers> \
  --tmux-session <session_name> \
  --yes
```

## Board workflow

The eval000030 board workflow is still experiment-specific internally, but it
is routed through the same repo-level interface:

```bash
conda run -n stellcoilbench_vmec python scripts/optimization_workflow.py prepare \
  --board experiments/wout_squid_eval_000030/board.yaml

conda run -n stellcoilbench_vmec python scripts/optimization_workflow.py launch \
  --board experiments/wout_squid_eval_000030/board.yaml \
  --yes
```

Supported board actions are `plan`, `generate`, `preflight`, `register`,
`prepare`, `status`, `monitor`, `ingest`, `screen`, `close`, and `launch`.

## Current cleanup status

- `scripts/run_simsopt_batch.py` is the generic Simsopt batch runner.
- `scripts/run_round1_wout20260324.py` is now a compatibility wrapper for old
  imports and commands.
- `experiments/wout_squid_eval_000030/workflow/experiment.py` is now treated as
  a legacy eval000030 board adapter; use `scripts/optimization_workflow.py`
  instead of calling it directly.
- Historical reports and Data Twin JSONL records still contain old command
  strings. Those should be preserved as provenance, not rewritten.
- Shell launch scripts under `scripts/launch_round1_eval000030_*.sh` and
  duplicate experiment launch scripts are historical convenience wrappers. New
  launches should use the unified workflow command above.

## Required operating rule

Every non-dry-run optimization launch must have a Data Twin campaign registered
before execution. The generic runner refuses non-dry-run execution unless
`--data-twin-campaign` is supplied by the workflow gate, or an explicit bypass
flag is used for exceptional recovery work.
