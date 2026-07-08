#!/usr/bin/env python3
"""Compatibility wrapper for the legacy wout_20260324 Round1 runner name.

The implementation has moved to ``scripts.run_simsopt_batch``. Keep this file
so older SOPs, tests, and workflow helpers that import private runner helpers
continue to work while new experiments use the generic runner directly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_simsopt_batch as _impl

_impl.SURFACE = "plasma_surfaces/wout_20260324.nc"
_impl.RESULTS_DIR = Path("results/round1_wout20260324_n4")
_impl.DEFAULT_POLICY = Path("policy/wout20260324_round1_policy.yaml")

globals().update(
    {
        name: getattr(_impl, name)
        for name in dir(_impl)
        if not name.startswith("__")
    }
)
main = _impl.main


if __name__ == "__main__":
    if (
        "--allow-legacy-direct" not in sys.argv
        and os.environ.get("STELLCOILBENCH_WORKFLOW_ENTRYPOINT")
        != "scripts/optimization_workflow.py"
    ):
        raise SystemExit(
            "Direct use of scripts/run_round1_wout20260324.py is disabled for agents. "
            "Use scripts/optimization_workflow.py for workflow operations, or "
            "scripts/run_simsopt_batch.py --dry-run for runner debugging."
        )
    if "--allow-legacy-direct" in sys.argv:
        sys.argv.remove("--allow-legacy-direct")
    main()
