# StellCoilBench

**Open benchmark suite for stellarator coil optimization algorithms.** Standardized case definitions (YAML), automated optimization via simsopt, post-processing (VMEC, Poincaré plots), and CI-driven leaderboards.

## Quick Start

```bash
# Add a case and push — CI runs it and updates leaderboards
stellcoilbench submit-case cases/my_case.yaml
git add submissions/ && git commit -m "Add submission" && git push
```

**Or:** Add a case under `cases/` and `git push` — CI will run it automatically.

### Local First Run

1. Create and activate a conda environment: `conda create -n stellcoilbench_vmec python=3.12 && conda activate stellcoilbench_vmec`
2. Install in editable mode: `pip install -e .`
3. Validate a case: `stellcoilbench validate-config cases/basic_tokamak.yaml`
4. List available cases: `stellcoilbench list-cases`
5. Run a case: `stellcoilbench submit-case cases/basic_tokamak.yaml`

See `docs/cases.rst` for case definitions and options.

## Repository Layout

| Directory | Purpose |
|-----------|---------|
| `cases/` | Benchmark case definitions (YAML). See `docs/cases.rst` |
| `cases/pending/` | Autopilot queue (JSON, written by proposer) |
| `cases/done/` | Autopilot results (gitignored for local dev; CI force-adds and tracks via LFS; results also in submissions/ + autopilot_failures.json) |
| `submissions/<surface>/<user>/<case_name>/<datetime>/` | Submission zips and PDFs |
| `submissions/<surface>/auto/<case_id>/` | Autopilot submissions |
| `docs/leaderboards/` | Per-surface leaderboards (CI-generated) |
| `policy/proposer_policy.yaml` | Autopilot tuning and guardrails |

## Commands

```bash
stellcoilbench list-cases                   # List available cases
stellcoilbench validate-config cases/X.yaml  # Validate case file
stellcoilbench submit-case cases/case.yaml   # Run a case locally
stellcoilbench run-ci-case cases/pending/X.json  # Run autopilot case
stellcoilbench update-db                     # Rebuild leaderboards from submissions
```

## Autopilot

Continuous CI loop: propose → run → record. Create `PAUSE_AUTORUN` to halt.

```bash
python -m tools.propose_batch --batch-size 3 --dry-run   # Preview
python tools/build_context.py | python -m json.tool      # Inspect context
```

## Submissions and Git LFS

Large files (VTU, VTS, PNG, PDF, and large JSON) in `submissions/` and `cases/done/` are stored with Git LFS in the main repo. Forks pushing to their fork use the fork's LFS storage by default.

- **Clone with LFS:** `git clone` + `git lfs pull` (or ensure `git lfs install` ran)
- **Code-only clone:** `GIT_LFS_SKIP_SMUDGE=true git clone ...` (skips ~17 GB)

See `docs/forking.md` for fork clone options and adding submissions.

## Documentation

- **ReadTheDocs**: https://stellcoilbench.readthedocs.io/
- **Cases**: `docs/cases.rst`
- **Autopilot**: `docs/autopilot.rst`
- **Forking and LFS**: `docs/forking.md`
- **Leaderboard**: `docs/leaderboard.rst`, `docs/leaderboard/metric_definitions.rst`
