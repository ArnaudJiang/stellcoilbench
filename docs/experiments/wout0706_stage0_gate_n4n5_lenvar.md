# wout0706 Stage0 Gate SOP

## Scope

Stage0 Gate for the large-major-radius `wout_0706` surfaces:

- `plasma_surfaces/wout_0706_high_pdrot.nc`
- `plasma_surfaces/wout_0706_medium.nc`
- `plasma_surfaces/wout_0706_low_pdrot.nc`

Matrix axes:

- `ncoils`: 4, 5
- Fourier order: 4, 5
- seeds: 7061, 7062
- length variance weights: 1, 10, 100
- backend: Simsopt only for this Gate, because the new coil length variance objective is Simsopt-side

Policy: `policy/wout0706_stage0_gate_n4n5_lenvar_policy.yaml`

Workflow entry point: `scripts/optimization_workflow.py`

Runner behind the workflow gate: `scripts/run_simsopt_batch.py`

Results root: `results/wout0706_stage0_gate_n4n5_lenvar/simsopt`

## Gate Criteria

- Full dry-run enumerates 72 Simsopt jobs and 0 FOCUS jobs.
- Manifest includes all 3 surfaces and both `ncoils = 4, 5`.
- Generated cases include:
  - `cc_threshold: 0.4`
  - `length_variance_weight: 1, 10, 100`
  - `coil_length_variance: l1`
- No existing result records are overwritten during Stage0.

## Dry-Run Command

```bash
conda run -n stellcoilbench_vmec python scripts/run_simsopt_batch.py \
  --policy policy/wout0706_stage0_gate_n4n5_lenvar_policy.yaml \
  --backend simsopt \
  --dry-run
```

## Launch Command After Approval

```bash
conda run -n stellcoilbench_vmec python scripts/optimization_workflow.py launch \
  --campaign wout0706_stage0_gate_n4n5_lenvar_20260707 \
  --policy policy/wout0706_stage0_gate_n4n5_lenvar_policy.yaml \
  --results-dir results/wout0706_stage0_gate_n4n5_lenvar/simsopt \
  --backend simsopt \
  --expected 72 \
  --surface-resolution 64 \
  --max-parallel-simsopt 72 \
  --tmux-session wout0706_stage0_gate \
  --yes
```

Do not launch the optimization grid without explicit confirmation. Do not call
`scripts/run_simsopt_batch.py` directly for non-dry-run launches. Use
`scripts/optimization_workflow.py prepare` first, then `launch --yes`.
