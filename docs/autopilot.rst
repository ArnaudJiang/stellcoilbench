Nonstop CI Autopilot
====================

The autopilot is a self-sustaining loop that continuously proposes,
runs, and records coil-optimization cases on a self-hosted CI runner.
Its purpose is to **autonomously explore the design space** across
multiple plasma surfaces and constraint settings, building the
leaderboard without manual intervention.

How it works
------------

Three phases repeat indefinitely:

1. **Propose** — ``python -m tools.propose_batch`` generates a batch of cases
   (default 10; configurable via ``policy/proposer_policy.yaml``).
2. **Run** — The CI runner executes each case via
   ``stellcoilbench run-ci-case``.  Cases run in parallel (up to 10
   concurrent jobs) with per-case timeouts.
3. **Record** — Results are written to
   ``cases/done/<case_id>/summary.json`` (locally, for runner polling;
   not committed) and to ``submissions/<surface>/auto/<case_id>/``.
   Failures are recorded in ``policy/autopilot_failures.json``.  The
   completed results feed the next proposal round.

A **batch barrier** ensures that the proposer never writes new cases
while ``cases/pending/`` still contains unfinished work.  The CI
workflow ``update-db-self-hosted.yml`` drives the loop: each push to
``main`` (or a 10-minute cron safety net) triggers
``run_autopilot_cases`` → ``propose_autopilot_batch``.  Autopilot
commits (``submissions/``, ``policy/autopilot_failures.json``,
``cases/pending/``) are filtered via ``paths-ignore`` so they do not
re-trigger benchmark jobs, keeping the loop self-sustaining.

Proposer modes
--------------

The proposer supports two modes.  Both share the same guardrails,
validation, and output format; only the case-generation strategy
differs.

Deterministic (genetic-algorithm) policy — default
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default proposer is a lightweight, deterministic genetic-algorithm
(GA) style optimizer.  It requires **no external API** and is fully
reproducible with a ``--seed`` flag.

Each batch is split into two halves (ratio set by
``exploit_fraction``):

- **Exploit (mutation)** — The proposer selects a parent from the
  top-:math:`k` feasible results (ranked by composite score), clones
  its ``case_config``, and applies random perturbations:

  - **Threshold jitter** — Each constraint threshold is multiplied by a
    log-normal factor (:math:`\sigma` from ``threshold_sigma``), nudging
    the optimizer toward a different trade-off on the Pareto frontier.
  - **Structural mutation** — With probability
    ``structural_mutation_prob``, the number of base coils or Fourier
    order is changed to an adjacent value.
  - A **novelty check** (config hash) rejects duplicates of recent runs.

- **Explore** — New random cases with surface, coil count, and order
  drawn from the policy's allowed sets.  Thresholds are sampled from
  log-uniform ranges to create diversity across the design space.

.. code-block:: bash

   # Default (deterministic) proposer
   python -m tools.propose_batch --batch-size 10 --dry-run --seed 42

LLM-powered policy — opt-in
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the ``--llm`` flag is passed, the proposer calls the LLM directly
with context from ``build_context`` (top parents, failure statistics,
surface exploration counts, run cards, postmortems) and
``llm_context.md``.  The LLM returns a list of ``mutate`` / ``explore``
**actions** that are converted into CI case dicts by ``apply_llm_action()``.

Requires ``ANTHROPIC_API_KEY`` (or ``KB_LLM_*`` env vars).  If the LLM
returns an error or fewer than ``batch_size`` valid cases, the proposer
**falls back automatically** to the deterministic GA policy.

.. code-block:: bash

   # LLM proposer (direct mode)
   python -m tools.propose_batch --batch-size 10 --llm --dry-run

The LLM mode is useful when you want the proposer to reason about
which surfaces are under-explored, which constraint combinations have
not yet been tried, or which failure modes to avoid — tasks that go
beyond what a fixed mutation/exploration policy can do.

Guardrails and safe mode
------------------------

The proposer checks a sliding window (default 30) of recent results
before every batch:

- **Fail-rate guard** — If the failure rate exceeds ``max_fail_rate``
  (default 0.6), proposals halt and ``PAUSE_AUTORUN`` is written.
- **Repeated-failure guard** — If the most common failure reason repeats
  more than ``max_common_failure_count`` times, proposals halt.
- **Critical-class guard** — Tracks classes like ``vmec_nonconverged``,
  ``nan_in_objective``, ``timeout``; halts if any exceeds
  ``max_critical_class_count``.

Between the safe-mode threshold (0.35) and the hard halt (0.6), the
proposer enters **safe mode**: mutation sigma is reduced, iteration caps
are lowered, and exploration is restricted to preferred (simpler)
surfaces.

Pausing and resuming
--------------------

Create the file ``PAUSE_AUTORUN`` in the repo root to halt all
proposals.  The proposer checks for this file on every invocation and
exits immediately if it exists.  Guardrail triggers can also create this
file automatically (controlled by ``cooldown.write_pause_file`` in the
policy).  To resume, simply delete the file:

.. code-block:: bash

   # Pause
   touch PAUSE_AUTORUN

   # Resume
   rm PAUSE_AUTORUN

Directory layout
----------------

::

   cases/pending/           Proposer writes new case JSONs here
   cases/done/<case_id>/    summary.json, case.yaml, coils.json
   submissions/<surface>/auto/<case_id>/   Results for leaderboard
   policy/proposer_policy.yaml             Proposer configuration
   tools/propose_batch/                    Proposer package (python -m tools.propose_batch)
   tools/build_context.py                  Context builder

Pending-case format
-------------------

Each file in ``cases/pending/`` is a JSON dict with:

- ``case_id`` — unique timestamp + random suffix
- ``case_config`` — full case YAML equivalent (surface, coils, optimizer,
  objective terms)
- ``resource`` — ``max_total_iterations`` (capped at 10 000) and
  ``timeout_minutes``
- ``parent_ids`` — list of parent case IDs (empty for exploration)
- ``tags`` — ``["exploit"]``, ``["explore"]``, ``["exploit", "llm"]``, etc.
- ``random_seed`` — for reproducibility

Commands
--------

.. code-block:: bash

   # Preview a batch (dry run)
   python -m tools.propose_batch --batch-size 10 --dry-run --seed 42

   # Build context payload from completed cases
   python tools/build_context.py [--out context.json]

   # Run a single CI case locally
   stellcoilbench run-ci-case <case_file> [--output-dir cases/done] \
       [--policy policy/proposer_policy.yaml]

   # Emergency stop
   touch PAUSE_AUTORUN

   # Resume after stop
   rm PAUSE_AUTORUN
