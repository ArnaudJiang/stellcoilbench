Overview
========

StellCoilBench is an open benchmark suite **and** an AI-automated
optimization engine for stellarator coil design.  It serves two
complementary purposes:

1. **Standardized benchmarking** — YAML case definitions, automated
   evaluation against reactor-scale engineering constraints, and
   CI-generated leaderboards let different optimization methods be
   compared on a level playing field.

2. **Autonomous coil optimization** — A nonstop CI autopilot
   continuously proposes, runs, and records optimization cases using a
   configurable policy.  The policy supports a deterministic
   genetic-algorithm (GA) proposer that mutates the best results and
   explores new regions of the design space, as well as an optional
   LLM-powered proposer that reasons about which configurations to try
   next.  See :doc:`autopilot` for details.

Together these make it possible to systematically search the
high-dimensional space of coil configurations (surface, number of coils,
Fourier order, constraint thresholds) across multiple plasma geometries,
building a growing database of reactor-scale-feasible designs.

Installation
------------

.. code-block:: bash

   pip install stellcoilbench

For development: ``pip install -e .``

Optional dependencies enable extra features. Install as needed:

- **Post-processing** (VMEC, Poincaré, quasisymmetry, Boozer plots, SIMPLE): ``pip install .[post-processing]``. Requires VMEC2000 and SIMPLE binaries; see CI workflow for build steps.
- **Structural analysis** (FEM, Von Mises stress): ``pip install .[structural]``. DOLFINx via ``conda install -c conda-forge fenics-dolfinx`` for full parallelism.
- **ParaStell** (finite-build coil mesh, structural FEM): ``pip install .[parastell]`` or ``bash tools/install_parastell_in_vmec.sh`` with stellcoilbench_vmec active.
- **LLM autopilot proposer**: ``pip install .[llm]``. Needs ``ANTHROPIC_API_KEY``.
- **Documentation build**: ``pip install .[DOCS]``.

See the `simsopt wiki <https://github.com/hiddenSymmetries/simsopt/wiki>`_ for VMEC and booz_xform installation details.

MPI Parallelization
------------------

Post-processing (VMEC, fieldline tracing) uses all MPI processes; optimization runs on rank 0 only. Run with ``mpirun -n 4 stellcoilbench submit-case cases/basic_MUSE.yaml`` (or ``--bind-to core --map-by core`` for CPU binding). Set ``OMP_NUM_THREADS=1``, ``MKL_NUM_THREADS=1``, ``OPENBLAS_NUM_THREADS=1`` to avoid thread oversubscription. CI uses 4 MPI processes per case. If MPI is not detected, the code runs in single-process mode.

Quick Start
-----------

**CI workflow (fastest):** Add a case file under ``cases/`` and push. CI runs it and updates the leaderboards.

**Local run:**

.. code-block:: bash

   stellcoilbench validate-config cases/basic_LandremanPaulQA.yaml
   stellcoilbench list-cases
   stellcoilbench submit-case cases/basic_LandremanPaulQA.yaml

This creates a submission in ``submissions/<surface>/<user>/<case_name>/<datetime>/`` with a zip and PDF plots. Regenerate leaderboards locally with ``stellcoilbench update-db``.

Commands
~~~~~~~~

- **validate-config** — Validate a case YAML configuration.
- **list-cases** — List available benchmark cases from ``cases/*.yaml``.
- **submit-case** — Run a case end-to-end: optimization → evaluation → post-processing → submission packaging.
- **run-case** — Run a case without creating a submission. Outputs coils to ``coils_runs/`` by default.
- **run-ci-case** — Run a case in CI mode: optimization only, no post-processing or submission packaging.
- **generate-submission** — Create a submission from existing results (coils, results.json).
- **post-process** — Run post-processing on existing coils: Poincaré plots, VMEC, quasisymmetry, VTK output.
- **update-db** — Regenerate leaderboards from submissions in ``submissions/``.

Submissions are organized as ``submissions/<surface>/<user>/<case_name>/<datetime>/all_files.zip`` (human) or ``submissions/<surface>/auto/<case_id>/results.json`` (autopilot). Each zip contains ``results.json``, ``coils.json``, and optional PDF outputs. CI scans submissions and regenerates leaderboards in ``docs/leaderboard/`` (RST) and ``docs/leaderboards/`` (Markdown).

**Autopilot (autonomous):**

.. code-block:: bash

   # Preview what the GA proposer would generate
   python -m tools.propose_batch --batch-size 10 --dry-run --seed 42

   # Run the full loop locally (propose → run → record)
   python -m tools.propose_batch --batch-size 3
   for f in cases/pending/*.json; do stellcoilbench run-ci-case "$f"; done

Repository Layout
-----------------

- **``cases/``** — YAML case definitions (surface, coils, optimizer). See :doc:`cases`.
- **``submissions/``** — Results: ``submissions/<surface>/<user>/<case_name>/<datetime>/all_files.zip`` (human) or ``submissions/<surface>/auto/<case_id>/results.json`` (autopilot).
- **``plasma_surfaces/``** — VMEC (``input.*``) and FOCUS (``*.focus``) surface files.
- **``policy/``** — Autopilot configuration (``proposer_policy.yaml``). See :doc:`autopilot`.
- **``tools/``** — Proposer (``propose_batch``) and context builder (``build_context.py``).
- **``tests/``** — Unit and integration tests.
- **``docs/``** — Leaderboards and generated documentation.

Contributing
------------

Contributions are limited to (a) adding a benchmark case or plasma surface, or
(b) opening a pull request for source code changes.

**1. Adding a case or plasma surface**

- **New case**: Add a YAML file under ``cases/`` defining a benchmark case (surface,
  coils, optimizer, objectives). See :doc:`cases` for the
  schema. Run ``stellcoilbench validate-config cases/your_case.yaml`` before pushing.
- **New plasma surface**: Add a VMEC input (e.g. ``input.LandremanPaul``) or FOCUS
  surface file under ``plasma_surfaces/``. Ensure the surface is referenced correctly
  in case YAML via ``surface_params.surface``.

After pushing, CI runs your case and updates the leaderboards.

**2. Source code changes**

For changes to ``src/``, ``tools/``, or tests: open a pull request against ``main``.
Run tests (``pytest tests/ -v --tb=short``), lint (``ruff check src/ tests/ --ignore F403,F405``),
and add or update unit tests. Use the ``stellcoilbench_vmec`` conda environment.
