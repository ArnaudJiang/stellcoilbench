# Repository Guidelines

## Project Structure & Module Organization
StellCoilBench is a Python package under `src/stellcoilbench/`. Core CLI entrypoints live in the package and expose the `stellcoilbench` command. Benchmark case definitions are YAML files in `cases/`; plasma input files are in `plasma_surfaces/`; generated or submitted benchmark outputs are organized under `submissions/`. Tests live in `tests/`, with subdirectories matching major package areas such as `cli`, `update_db`, `validate_config`, and `coil_optimization`. Documentation sources are in `docs/`, while CI and maintenance scripts are in `tools/`.

## Build, Test, and Development Commands
- `pip install -e .` installs the package in editable mode.
- `stellcoilbench list-cases` lists available benchmark cases.
- `stellcoilbench validate-config cases/basic_tokamak.yaml` validates a case file.
- `stellcoilbench submit-case cases/basic_tokamak.yaml` runs a local benchmark case.
- `pytest tests/` runs the test suite.
- `ruff check src/ tests/ tools/ knowledge/` checks lint issues.

Use the `stellcoilbench_vmec` conda environment as this repository's Python environment. Run Python, pytest, Ruff, and optimization scripts from that environment, especially for VMEC, MPI, Simsopt, PyYAML, plotting, or structural dependencies.

## Coding Style & Naming Conventions
The project targets Python 3.12+ and uses Ruff and Black. Follow standard 4-space Python indentation, keep functions focused, and prefer explicit names for physics, optimization, and path-handling code. Test files should be named `test_*.py`, and case definitions should use descriptive lowercase YAML names, such as `basic_tokamak.yaml`.

## Testing Guidelines
Add or update tests for source changes. Prefer focused pytest tests near the affected subsystem, for example `tests/validate_config/` for schema behavior or `tests/update_db/` for leaderboard aggregation. Use targeted runs during development, then run `pytest tests/` before submitting broader changes.

## Commit & Pull Request Guidelines
Git history currently uses concise Conventional Commit style, for example `chore: update StellCoilBench leaderboard`. Use short, imperative commit messages with a clear scope when helpful. Pull requests should describe the change, list commands run, link related issues, and include screenshots or generated artifact notes when documentation, plots, or leaderboards change.

## Agent Optimization Workflow Rules

These rules are mandatory for Codex/agent sessions that touch coil optimization,
Simsopt scans, Data Twin campaigns, experiment boards, policies, launch scripts,
or result ingestion.

- Use `scripts/optimization_workflow.py` as the only agent-facing optimization
  workflow entry point.
- Do not launch or operate optimization campaigns by directly calling legacy
  entry points such as `experiments/wout_squid_eval_000030/workflow/experiment.py`
  or `scripts/run_round1_wout20260324.py`.
- Treat `experiments/wout_squid_eval_000030/workflow/experiment.py` as an
  internal adapter used by `scripts/optimization_workflow.py --board ...`.
- Treat `scripts/run_round1_wout20260324.py` as an import/CLI compatibility
  wrapper for historical records only.
- `scripts/run_simsopt_batch.py` may be used directly for dry-runs and runner
  debugging, but non-dry-run launches must go through
  `scripts/optimization_workflow.py prepare` and `launch`.
- Every non-dry-run optimization launch must have a Data Twin campaign and must
  pass the workflow gate before execution.
- Use Data Twin collaboration commands for handoff and review:
  `index rebuild`, `status`, `review`, `decide`, and `compare`.
- JSONL files under `experiments/data_twin/<campaign>/` are the source of truth.
  `experiments/data_twin/data_twin_index.sqlite` is a rebuildable local index
  and must not be committed.

Canonical agent commands:

```bash
conda run -n stellcoilbench_vmec python scripts/optimization_workflow.py prepare ...
conda run -n stellcoilbench_vmec python scripts/optimization_workflow.py launch ... --yes
conda run -n stellcoilbench_vmec python scripts/optimization_workflow.py status --campaign <campaign_id>
conda run -n stellcoilbench_vmec python scripts/optimization_workflow.py review --campaign <campaign_id> --review-status approved --by <name> --note "..."
conda run -n stellcoilbench_vmec python scripts/optimization_workflow.py decide --campaign <campaign_id> --decision refine --reason "..." --next-action "..."
conda run -n stellcoilbench_vmec python scripts/optimization_workflow.py compare --campaign <campaign_a> --campaign <campaign_b>
```

## eval000030 Coil Optimization SOP

This SOP is the required agent workflow for `plasma_surfaces/wout_squid_eval_000030.nc`. The objective is to map the coil-optimization parameter space, identify robust seed regions, repair promising candidates, verify them at higher resolution, and only then promote final geometry/showcase artifacts. Keep every stage board-driven, Data Twin-backed, and reproducible from files in `experiments/wout_squid_eval_000030/`.

### Non-Negotiable Rules

- Use `conda run -n stellcoilbench_vmec ...` for Python, Ruff, pytest, and optimization commands.
- Use `scripts/optimization_workflow.py --board <board.yaml>` as the only standard stage workflow entrypoint.
- Use one hand-edited board per stage under `experiments/wout_squid_eval_000030/board*.yaml`; do not hand-edit generated policy JSON.
- Write generated policies and manifests under `experiments/wout_squid_eval_000030/policies/generated/`.
- Write raw optimization outputs under `experiments/wout_squid_eval_000030/raw/results/`.
- Write screening, analysis, Pareto, and narrative reports under `experiments/wout_squid_eval_000030/reports/`.
- Register every launched stage into Data Twin under `experiments/data_twin/`.
- Never overwrite an existing result directory. If a setup bug requires rerun, create a suffix such as `v2`, `rerun`, or a dated campaign name.
- Use the workflow lifecycle as a strict state machine: `prepare -> launch -> sync -> screen -> close`.
- `prepare` is the only normal way to create generated policy/manifest files and register planned runs in Data Twin.
- `launch` must refuse to run unless the campaign is registered or partially running. If a campaign is already fully ingested or screened, start a new wave/result directory instead of relaunching it.
- `sync` is the required Data Twin ingestion step after records appear. Raw `record.json` files alone are not a complete experiment.
- `screen` must run only after `sync`; `close` must run only after `screen`.
- Use `status` to audit raw results, Data Twin rows, and lifecycle state before making decisions.
- Long runs must be launched in `tmux`; monitor with `record.json`, `objective_history.csv`, `constraint_history.csv`, Data Twin state, and process counts.
- Rank and promote candidates by final `record.json`/Data Twin metrics, not by backend objective values alone.

Canonical stage commands:

```bash
conda run -n stellcoilbench_vmec python scripts/optimization_workflow.py prepare --board <board.yaml>
tmux new-session -d -s <session_name> 'cd /home/jiangxm/stellcoilbench && MPLCONFIGDIR=/tmp/stellcoilbench_mplconfig conda run -n stellcoilbench_vmec python scripts/optimization_workflow.py launch --board <board.yaml> --yes'
conda run -n stellcoilbench_vmec python scripts/optimization_workflow.py status --board <board.yaml>
conda run -n stellcoilbench_vmec python scripts/optimization_workflow.py sync --board <board.yaml>
conda run -n stellcoilbench_vmec python scripts/optimization_workflow.py screen --board <board.yaml>
conda run -n stellcoilbench_vmec python scripts/optimization_workflow.py close --board <board.yaml>
```

Expected lifecycle states are:

- `draft`: board exists but no Data Twin campaign has been initialized.
- `planned`: generated files or campaign skeleton exist, but planned runs are not fully registered.
- `registered`: planned cases/runs are in Data Twin and launch is allowed.
- `running`: raw records are appearing but Data Twin is not complete.
- `results_complete_uningested`: raw records are complete but `sync` has not been run.
- `ingested_partial`: some raw records have been synced to Data Twin.
- `ingested_complete`: all planned raw records have corresponding evaluations/metrics in Data Twin.
- `screened`: screening decisions/artifacts have been written.
- `closed`: the stage is frozen; do not relaunch or mutate the campaign.

### Baseline Metric Policy

Use this metric vocabulary consistently across all stages:

- Field metric: `avg_BdotN_over_B`, lower is better.
- Coil-coil clearance: `final_min_cc_separation`; target bands are exploratory `>=0.20`, promising `>=0.24`, strict `>=0.25`, and industrial push `>=0.30`.
- Coil-surface clearance: `final_min_cs_separation`; target bands are exploratory `>=0.20`, promising `>=0.24`, strict `>=0.25`, and industrial push `>=0.30`.
- Geometry: prefer `max_curvature <= 5.0`, `mean_squared_curvature <= 5.0`, `arclength_variation <= 0.5`, `max_torsion <= 7.0`, and `length_ratio <= 1.25` unless a stage explicitly relaxes one item for exploration.
- Topology: promote only clean-link candidates. Broad scans may use post-run link screening for speed; final repair and showcase candidates need explicit topology checks.
- Ranking: first filter by topology and hard geometry, then sort by clearance tier and `avg_BdotN_over_B`; keep Pareto alternatives instead of a single lowest-field record.

Thresholds such as length target, cc/cs target, curvature cap, and torsion cap are physical constraints and must stay in reasonable fixed ranges. Weights may be scanned discretely; physical thresholds must not be randomly swept without a written reason in the board or report.

### Parameter-Space Vocabulary

Use this vocabulary when designing new eval000030 boards. The optimization is non-convex; broad scans are meant to discover different local-minimum basins, not to randomly perturb every available scalar.

- `init_family`: the fresh-coil initialization geometry family. This is the main Stage1 basin-discovery axis and should describe the starting geometry, for example `baseline_modular`, `surface_following`, `open_clearance`, `compact_field`, `smooth_long`, `short_compact`, `phase_shifted`, `staggered_z`, `current_guided_shape`, or `warm_parent_perturb`.
- `init_scale` or explicit geometry-initialization fields: physical initial-geometry controls such as `major_radius_scale`, `minor_radius_scale`, `vertical_scale`, `radial_offset`, `normal_offset`, `initial_cs_gap`, `coil_spacing_scale`, `toroidal_phase_offset`, `coil_phase_jitter`, and `z_offset`.
- `current_family`: initial current distribution, for example `uniform`, `scaled_uniform`, `mild_asymmetry`, `outer_inner_bias`, `warm_parent_current`, `current_fixed_shape_free`, or `current_free`.
- `policy_family`: objective-weight strategy. Keep this separate from `init_family`. Standard broad-scan representatives are `balanced`, `clearance_biased`, `smoothness_biased`, and `length_balance_biased`.
- `geometry_weight_scale`: a multiplier on objective weights for geometric penalties.

Current legacy boards use `family` and `geom_scale`. In the current workflow implementation, `family` names such as `seed_balanced`, `seed_clearance_repair`, and `seed_smooth_repair` primarily select objective-weight families, while `geom_scale` multiplies geometric objective weights. Do not treat legacy `geom_scale` as a physical initial-coil size. When a new board intends to scan real geometry, add explicit initialization fields or clearly document the limitation in the board and report.

Recommended Stage1 axes, in priority order:

- Initial geometry: `init_family` plus explicit geometry scales/offsets.
- Seed diversity: `random_seed` and `dof_perturbation`, with duplicate audits.
- Current initialization: total-current scale and per-coil current pattern.
- A small number of `policy_family` choices.
- Resolution/cost settings only after the workflow gate is stable.

Avoid making Stage1 a high-dimensional weight search. Weights should be a small set of physically named strategies; thresholds remain fixed engineering constraints unless the board explicitly states that it is a threshold-push experiment.

### Stage0: Workflow Gate

Purpose: prove that the board, policy generator, runner, Data Twin registration, and metric extraction are working before spending cluster time.

Default setup:

- `order=6`
- `surface_grid=64`
- `coil_quadpoints=128`
- `max_iterations=100-300`
- small case count covering representative families and seeds
- parallelism low enough that failures are easy to inspect

Required actions:

- Run `preflight` and confirm planned cases, generated manifest rows, result directory, and overwrite status.
- Run a short launch in `tmux`.
- Confirm `case.yaml`, `record.json`, objective histories, and Data Twin run records are produced.
- Run `ingest` and `screen`.

Exit criteria:

- All planned smoke cases either complete or fail with understood setup errors.
- Data Twin contains the expected campaign, cases, runs, and metrics.
- The screen report shows usable metric columns and failure categories.

Branching:

- If setup fails, fix workflow/runner/board issues and rerun as a new suffix.
- If metrics are missing or duplicated, stop and fix ingestion before Stage1.
- Do not select final physical parents from Stage0.

Reference board/result:

- `experiments/wout_squid_eval_000030/board.yaml`
- `experiments/wout_squid_eval_000030/raw/results/round1_stage0_gate_res64_q128_o6_20260706`
- `experiments/data_twin/eval000030_round1_stage0_gate_20260706`

### Stage1: Broad Parameter Map

Purpose: cover the plausible parameter space cheaply and find reusable seed regions. This is the primary source of parent diversity and the default restart point when later local repair stalls.

Default setup:

- `order=6`
- `surface_grid=64`
- `coil_quadpoints=128`
- `max_iterations=250-300`
- broad initial geometry families, explicit initialization scales/offsets, current families, a small number of policy families, and seeds
- parallelism usually `64-128`; avoid `256` unless startup overhead has been tested on the current host
- use post-run topology screening rather than heavy link guard unless topology failures dominate

Coverage policy:

- Cover distinct initialization basins: surface-following, open-clearance, smooth-long, phase-shifted/staggered, current-guided, and warm-parent perturbation when parents exist.
- Keep policy families sparse and named: `balanced`, `clearance_biased`, `smoothness_biased`, and `length_balance_biased`.
- Sweep weights only through those named policy families; do not sweep physical thresholds wildly.
- Sweep real geometry through explicit initialization parameters, not through legacy `geom_scale`.
- Include current initialization diversity when the runner supports it, and report whether current differences survive optimization or collapse to the same solution.
- Include enough seeds per region to test whether a region is real or a one-off optimizer accident.
- Preserve all Stage1 outputs even when no case is strictly feasible; Stage1 is a map, not a final-candidate stage.

Screening policy:

- Report total completed, failed, strict feasible count, and tier counts for `cc` and `cs`.
- Summarize success rates by `init_family`, explicit geometry-scale fields, `current_family`, `policy_family`, flux weight, and seed when available.
- For legacy boards, label `family` and `geom_scale` as objective-policy and weight-scale axes unless the board explicitly implements physical initialization changes.
- Detect duplicate outputs and verify seed actually changes initialization.
- Cluster candidates by geometry and metrics before promotion. Do not let many near-duplicate records from one basin crowd out distinct basins.
- Select parent classes: high-cc, high-cs, low-field-error, smooth geometry, topology-clean, and balanced Pareto candidates.

Exit criteria:

- The broad map identifies 3-8 promising regions or shows that the scanned region is physically unproductive.
- Parent candidates have source `run_id`, `coils.json`, policy parameters, and metrics recorded.
- A Stage2 board can be generated from Data Twin or a report table without manual guessing.

Branching:

- If `cs` is good but `cc` is low, bias Stage2 toward coil-coil clearance and lower field pressure.
- If `cc` is good but `cs` is low, bias Stage2 toward surface clearance barriers and screen-first rejection.
- If field error is good but topology/geometry is bad, add smoothness and topology screening before repair.
- If all regions are poor, revise Stage1 parameter ranges before doing local repair.

Reference campaigns:

- Main broad board: `experiments/wout_squid_eval_000030/board_industrial_broad.yaml`
- Main broad Data Twin: `experiments/data_twin/eval000030_round1_stage1_broad_20260706`
- Industrial cc/cs push board: `experiments/wout_squid_eval_000030/board_stageA_cccs03_industrial_broad.yaml`

### Stage2: Region Refinement

Purpose: refine the best Stage1 regions while staying at affordable resolution. Stage2 tests whether a region has stable improvement under more iterations and targeted policy changes.

Default setup:

- `order=6`
- `surface_grid=64`
- `coil_quadpoints=128`
- `max_iterations=400-800`
- selected regions only, usually 3-5 regions per board
- multiple seeds per region
- warm-start from Stage1 parents when the board supports `parents` and `initial_coils_path`

Policy design:

- Adjust weights, not physical thresholds, unless the stage is explicitly a threshold-push experiment.
- Increase `cc`, `cs`, curvature, torsion, and arclength weights only with a stated failure mode.
- Keep a balanced field term so the optimizer does not produce clearance-only coils with unusable `avg_BdotN_over_B`.
- Avoid repeatedly increasing one weight in the same basin after geometry terms are already saturated.

Exit criteria:

- Identify whether each selected region improves, plateaus, or regresses.
- Keep Pareto candidates, not only the lowest field error candidate.
- If best clearance remains below target after several policy variants, stop local weight stacking and move to Stage3 verification or Stage4 strategy matrix.

Branching:

- Improvement with acceptable geometry: send top candidates to Stage3.
- Local plateau below target: return to Stage1 parents and build a Stage4 strategy matrix.
- Regression after warm-start: keep the failed board as evidence, but do not reuse that result directory.

Reference boards:

- `experiments/wout_squid_eval_000030/board_stage2_refine.yaml`
- `experiments/wout_squid_eval_000030/board_stage2b_geometry_repair.yaml`
- `experiments/wout_squid_eval_000030/board_stage2c_geometry_push.yaml`

### Stage3: High-Resolution Verification And Warm-Start Repair

Purpose: check whether promising low-resolution candidates survive higher-resolution metric evaluation, then run targeted repair only from verified parents.

Verification setup:

- Evaluate selected `coils.json` at `surface_grid=128`.
- Use `coil_quadpoints=256` for verification or final repair.
- Record verification metrics separately under `reports/`.
- Do not overwrite the Stage1/Stage2 records; verification is a new artifact.

Warm-start repair setup:

- `order=6` unless a written reason exists to increase order.
- `surface_grid=128`, `coil_quadpoints=256` for final-quality repair.
- `max_iterations=600-1200` depending on convergence and stage budget.
- Parent paths must be normalized to absolute paths in generated runner policies.

Exit criteria:

- Candidate keeps its ranking approximately under high-resolution evaluation.
- `cc`, `cs`, topology, curvature, torsion, arclength variation, length ratio, and field error are all reported.
- A candidate is either promoted to Stage4/final candidate set or rejected with a specific failure mode.

Branching:

- If high-resolution metrics collapse, return to Stage1/Stage2 and select different parent families.
- If high-resolution metrics hold but one geometry metric fails, run a narrow repair board.
- If repair improves field but damages clearance, stop that path and report the tradeoff.

Reference files:

- Verification helper: `experiments/wout_squid_eval_000030/scripts/verify_high_resolution_candidates.py`
- Warm repair board: `experiments/wout_squid_eval_000030/board_stage2d_highres_warm_repair.yaml`

### Stage4: Strategy Matrix And Candidate Promotion

Purpose: when local refinement plateaus, run a deliberate strategy matrix over Stage1 parents, then promote only high-resolution verified Pareto candidates.

Strategy matrix setup:

- Select 6-12 parents from Stage1/Data Twin, covering high-cc, high-cs, low-field-error, smoothness-slack, and balanced Pareto classes.
- Run 3-6 policy families over every parent.
- Use at least two perturbation seeds per parent-policy pair.
- Start at `res64/q128`, `order=6`, `max_iterations=600-800`.
- Keep generated boards and reports tied to exact parent `run_id` and `coils.json`.

Policy families to consider:

- Direct cc push: higher coil-coil separation pressure with moderate smoothness.
- Low-flux open-cc: lower field pressure to let coils open, then restore field later.
- Smooth guided clearance: stronger curvature/torsion/arclength terms while pushing clearance.
- Balanced field-clearance: preserve field while improving both cc and cs.
- Surface barrier repair: harder cs penalty or screening when final shortest-distance cs is low.
- Topology conservative repair: post-run link screening for broad matrices; guarded link checks only for final/high-risk repair.

Promotion criteria:

- Promote a candidate only after high-resolution verification.
- Require clean topology and acceptable geometry before entity/showcase export.
- Keep a Pareto set: lowest field error, highest cc, highest cs, smoothest geometry, and best balanced candidate.
- Record source stage, board, run id, parent id, policy family, metrics, and artifact paths in the final report.

Branching:

- If one policy family clearly dominates, generate a narrower Stage3 repair board from its best verified records.
- If no policy reaches the next clearance tier, revise Stage1 coverage rather than continuing local repair.
- If a candidate is only good at low resolution, keep it as evidence but do not showcase it.

Reference board/report:

- `experiments/wout_squid_eval_000030/board_stage2f_stage1_strategy_matrix.yaml`
- `experiments/wout_squid_eval_000030/reports/round1_stage2f_stage1_strategy_matrix_res64_q128_o6_analysis/stage2f_matrix_report.md`

### Extension Strategy Playbook

Use these branches when the Stage0-Stage4 flow stalls or exposes a specific failure mode.

Data Twin quantitative restart:

- Before creating a new board, query Data Twin and existing reports for completed records, tier counts, and by-region success rates.
- Do not rank from campaigns with registered cases but no metrics.
- The completed broad campaign `experiments/data_twin/eval000030_round1_stage1_broad_20260706` is the primary restart source.

cc plateau:

- If `cc` plateaus below target in warm repair, restart from Stage1 parents instead of stacking larger `cc_weight` in the same basin.
- Lower field pressure can help coils open, but must be followed by field and geometry recovery.
- Track whether improved `cc` comes with worse curvature, torsion, arclength variation, or field error.

cs mismatch:

- Simsopt `CurveSurfaceDistance` objective history and final `shortest_distance()` reporting can differ because they use different evaluation paths/sampling.
- Treat final `record.json` metrics as the shared ranking source.
- If final cs remains low, prefer harder barrier or screen-first rejection over merely increasing a soft weight.

Topology/link:

- Broad scans should normally screen topology after the run for speed.
- Final repair and showcase candidates need explicit link diagnostics.
- If initialized coils are unlinked but optimized coils link, treat this as a missing topology constraint or too-weak geometry policy; do not promote the case.

Resolution/cost:

- Use `res64/q128` for broad maps and strategy matrices.
- Use `res128/q256` for verification and final repair.
- Do not run large `res128/q256` grids until low-resolution experiments show a clear directional signal.

Parallelism:

- Prefer `64-128` concurrent cases for broad scans unless measured otherwise.
- If `record.json` growth is slow while many workers are active, check startup/synchronization overhead, library thread oversubscription, and per-worker multiprocessing/MPI behavior.
- Reducing parallelism can improve throughput when Simsopt startup overhead dominates.

Seed and duplicate audits:

- Every broad scan must check whether seed changes affect initialized coils and final outputs.
- If duplicate result groups appear, inspect `cs_guard_final.json`, `early_stop_final.json`, parent paths, skip/reuse behavior, and rollback behavior before trusting the scan.

Parent paths:

- Board parent paths may be relative, but generated runner policies must use valid absolute `initial_coils_path` values.
- If `initial_coils_path not found` occurs, fix the generator and rerun into a new suffix.

Entity/showcase export:

- Export solid coils only from high-resolution verified, topology-clean candidates.
- Record dimensions, generation script, source `coils.json`, source `run_id`, and output file list.
- If a solidification method introduces unnecessary frame flips or swapped width/height, regenerate all affected showcase files and document the corrected dimensions.

### Historical Findings To Preserve

- The early Stage1 broad scan completed 576/576 cases but found no strict feasible case; `cc` was the dominant hard failure, while `cs` was often acceptable.
- Strong Stage1 parent density appeared in `seed_clearance_repair`, followed by `seed_balanced` and `seed_smooth_repair`.
- `seed_bn_recover` can lower field error but is weak for topology and geometry.
- Stage2/2b/2c produced repair seeds but plateaued below the `cc >= 0.25` target.
- Stage3 high-resolution checks showed some low-resolution candidates were real, but warm repair improved `cc` only modestly and did not reach target.
- Stage2e1 had a parent path bug; `stage2e1_v2` fixed paths but moved `cc` backward while lowering field error.
- Stage2f strategy matrix showed `matrix_low_flux_open_cc` was the only family that moved `cc` beyond about `0.245`, with best observed `cc` about `0.24604`.
- The current industrial direction is to test broader Stage1/Stage4 coverage with `cc/cs` threshold pressure near `0.30`, then generate a Pareto set and only promote verified candidates.
