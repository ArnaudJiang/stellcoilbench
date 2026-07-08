# Repo Handoff Audit 2026-07-08

## Scope

This audit covers optimization workflow entry points, Data Twin launch gating,
and obvious legacy command paths that could confuse a new Simsopt user.

## Changes made

- Added `scripts/optimization_workflow.py` as the repo-level workflow entry
  point.
- Kept `experiments/wout_squid_eval_000030/workflow/experiment.py` in place as
  a legacy eval000030 board adapter, with a docstring pointing users to the
  repo-level entry point.
- Updated wout0706 Stage0 SOP to use `scripts/optimization_workflow.py`.
- Updated eval000030 shell launch wrappers to call `scripts/run_simsopt_batch.py`
  and require `DATA_TWIN_CAMPAIGN`.
- Updated `register_a1_revised_data_twin.py` to write command snapshots using
  the generic Simsopt runner and Data Twin campaign flag.
- Documented the handoff command surface in
  `docs/experiments/optimization_workflow_handoff.md`.

## Stable user-facing commands

Generic policy scan:

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

Board scan:

```bash
conda run -n stellcoilbench_vmec python scripts/optimization_workflow.py prepare \
  --board experiments/wout_squid_eval_000030/board.yaml
```

## Remaining historical references

These are intentionally not rewritten in this pass:

- Historical SOP: `docs/experiments/wout20260324_round1_sop.md`.
- Historical audit notes: `reports/data_twin_audit.md`.
- Historical experiment reports under
  `experiments/wout_squid_eval_000030/reports/`.
- Existing Data Twin JSONL command snapshots and raw launch logs.

Those records are provenance. New operational docs and wrappers should use
`scripts/optimization_workflow.py`.

## Validation run

- `conda run -n stellcoilbench_vmec python -m py_compile scripts/optimization_workflow.py experiments/wout_squid_eval_000030/workflow/experiment.py scripts/workflow_launch.py scripts/run_simsopt_batch.py`
- `conda run -n stellcoilbench_vmec ruff check scripts/optimization_workflow.py scripts/workflow_launch.py experiments/wout_squid_eval_000030/workflow/experiment.py`
- `conda run -n stellcoilbench_vmec python scripts/optimization_workflow.py --help`
- `conda run -n stellcoilbench_vmec python scripts/optimization_workflow.py plan --board experiments/wout_squid_eval_000030/board.yaml --print-command`
- `conda run -n stellcoilbench_vmec python scripts/optimization_workflow.py launch --campaign wout0706_stage0_gate_n4n5_lenvar_20260707 --policy policy/wout0706_stage0_gate_n4n5_lenvar_policy.yaml --results-dir results/wout0706_stage0_gate_n4n5_lenvar/simsopt --backend simsopt --expected 72 --surface-resolution 64 --max-parallel-simsopt 72 --tmux-session wout0706_stage0_gate --print-command`
- `bash -n` on all six eval000030 launch wrappers.
