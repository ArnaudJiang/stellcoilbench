# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Rules

- Always run and test code in the `stellcoilbench_vmec` conda environment.
- Never modify code outside the scope of what is asked. If a function needs modification, only change that function — nothing else in the file.
- Never ask to proceed; always proceed with code changes, running, and debugging.
- After changing source code, run the corresponding unit tests to verify they still pass. If no unit tests exist for the changed code, add them.
- Always run unit tests in the background (use `run_in_background`).

## Project Overview

StellCoilBench is an open benchmark suite for stellarator coil optimization algorithms. It provides standardized case definitions (YAML), automated optimization via `simsopt`, post-processing (VMEC, Poincare plots, quasisymmetry), and CI-driven leaderboard generation. A custom fork of simsopt (`auglag_coils` branch) is used.

## Common Commands

```bash
# Install (editable)
pip install -e .

# Run tests (excludes slow-marked tests by default per pytest.ini)
pytest tests/

# Run a single test file or package
pytest tests/cli/ tests/validate_config/ tests/update_db/

# Run a single test function
pytest tests/cli/test_submit_run.py::test_function_name

# Run slow tests only
pytest tests/ -m slow

# Lint
ruff check src/ tests/ tools/ knowledge/ --ignore F403,F405

# Run a case locally
stellcoilbench submit-case cases/basic_tokamak.yaml

# Run with MPI (optimization is rank-0 only; post-processing uses all ranks)
mpirun -n 4 stellcoilbench submit-case cases/basic_MUSE.yaml

# Update leaderboards from submissions
stellcoilbench update-db

# Autopilot: propose a batch of cases
python -m tools.propose_batch --batch-size 3 --dry-run --seed 42
```

## Architecture

The package lives in `src/stellcoilbench/` with these core modules:

- **`cli/`** — Typer CLI package (`stellcoilbench` command). Commands: validate-config, list-cases, submit-case, run-case, run-ci-case, update-db, generate-submission, post-process. `validate_cmd.py`, `list_cases_cmd.py`, `submit_run.py`, `post_process.py`, `update_db_cmd.py`. Hardware detection and version tracking live in `cli_helpers.py`.

- **`coil_optimization/`** — Core optimization logic. `optimization.py` wraps scipy.optimize (L-BFGS-B, BFGS, SLSQP) and a custom augmented Lagrangian solver. Implements Fourier continuation (progressively increasing coil Fourier orders) and `LinearPenalty` threshold wrappers for objectives. MPI-aware: only rank 0 optimizes, other ranks wait at barrier.

- **`post_processing/`** — Runs after optimization: Poincaré plots (fieldline tracing), QFM surface computation, VMEC equilibrium, quasisymmetry/iota profiles, Boozer surface plots, SIMPLE particle tracing, VTK output, B·n error visualization. MPI-parallel for VMEC and fieldline tracing; plotting on rank 0 only.

- **`update_db/`** — Scans `submissions/` for zip files or `results.json` directories, aggregates metrics, generates per-surface leaderboards in RST/Markdown/JSON under `docs/leaderboard/` and `docs/leaderboards/`. Key submodules: `submission_io.py` (build methods/leaderboard JSON), `_load_submissions.py`, `_writers_common.py`, `_writers_surface.py`, `_writers_reactor.py`, `_writers_metric_defs.py`, `_constraints.py`, `_path_parsing.py`, `_formatting.py`, `_viz_links.py`, `_recompute.py`, `_backfill.py`.

## Git LFS and Forks

- **Submissions** use Git LFS for `.vtu`, `.vts`, `.png`, `.pdf` under `submissions/`. Blobs live in dedicated repo `stellcoilbench-lfs`; only upstream has write access.
- **Fork isolation:** Forks cannot push to upstream LFS (403). Fork owners who add submissions must add `.lfsconfig` pointing to their own LFS backend. See `docs/forking.md`.

- **`validate_config/`** — Validates case YAML and autopilot JSON configs with detailed error reporting.

- **`config_scheme.py`** — Dataclasses: `CaseConfig` (case YAML schema), `SubmissionMetadata`.

- **`evaluate.py`** — Loads case configs, computes evaluation metrics, builds leaderboard data.

- **`path_utils/`** — Path resolution (case/surface), YAML load/dump. **`finite_build/`** — Winding-pack turn-count model. **`reactor_scale.py`** — ARIES-CS scaling. **`mpi_utils.py`** — MPI helpers.

## Key Data Flow

1. User defines a case in `cases/*.yaml` (surface, coil params, optimizer, objective terms)
2. `submit-case` runs optimization → writes `results.json`, `coils.json`, VTK files
3. Results are packaged into `submissions/<surface>/<user>/<case_name>/<datetime>/all_files.zip`
4. CI scans submissions and regenerates leaderboards in `docs/leaderboards/`

## Important Directories

- `cases/` — Benchmark case YAML definitions (also `cases/pending/` for autopilot queue)
- `submissions/` — Output results organized by surface/user/timestamp (Option C: success results)
- `plasma_surfaces/` — Input plasma surface files (VMEC input and FOCUS formats)
- `policy/` — Autopilot tuning (`proposer_policy.yaml`, `autopilot_failures.json`); create `PAUSE_AUTORUN` to halt the loop
- `knowledge/` — LLM context (`llm_context.md`), surface catalog; make_run_card, make_postmortem, llm_client, llm_endpoints
- `docs/leaderboard/` — RST leaderboard (surface_specific, reactor_scale, metric_definitions)
- `docs/leaderboards/` — CI-generated per-surface Markdown leaderboards

## Git Setup (one-time after clone)

To avoid merge conflicts when pulling (leaderboard files are generated by CI and overwritten locally):

```bash
git config merge.theirs.driver 'cat %B > %A'
```

Use `--local` for this repo only (default) or `--global` to apply everywhere. Note: `docs/leaderboard.json` is now gitignored; only `docs/leaderboard.rst` uses `merge=theirs`.

## Build & Dependencies

- Python >= 3.12, build system: Hatchling
- Key dependency: `simsopt` from a custom GitHub fork (`PedroFranciscoGil/simsopt@auglag_coils`)
- CI installs system deps: gfortran, OpenMPI, VMEC2000, SIMPLE, booz_xform, virtual-casing
- When using MPI, set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` to avoid thread oversubscription
- **Structural analysis (FEM)**: ParaStell is required for structural cases with QA/stellarator coils (e.g. Landreman-Paul). Run `bash tools/install_parastell_in_vmec.sh` with stellcoilbench_vmec active, or use `environment-parastell.yml` for a dedicated env.

## CI Workflows

- **`ci.yml`** — Runs on push/PR: lint with ruff, run full test suite on Ubuntu 24.04 with Python 3.13
- **`case-only-pr.yml`** — Case-only PR validation: when only `cases/` and policy files change, validates case YAMLs and can auto-approve/merge
- **`update-db-self-hosted.yml`** — Self-hosted runner: runs benchmark cases (up to 8 parallel, 24h timeout), updates leaderboards, and runs the nonstop autopilot loop (triggered every 10 minutes via cron)

**Repository variables** (Settings → Secrets and variables → Actions → Variables): `CI_BENCHMARK_REPO` (default `akaptano/stellcoilbench`, set for forks); `CONDA_ROOT`, `SIMPLE_EXECUTABLE` (override self-hosted paths).
