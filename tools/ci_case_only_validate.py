#!/usr/bin/env python3
"""
Validate case-only or case+plasma_surfaces PRs for CI.

Classifies changed files, validates case YAMLs and plasma surface files,
and returns pass/fail. Used by .github/workflows/case-only-pr.yml.

Usage:
  # From workflow (env vars):
  CHANGED_FILES="cases/x.yaml\nplasma_surfaces/input.foo" MAINTAINER_BYPASS=false python -m tools.ci_case_only_validate

  # CLI for local simulation:
  python -m tools.ci_case_only_validate --base main --head HEAD
  python -m tools.ci_case_only_validate --files cases/x.yaml plasma_surfaces/input.foo
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _is_accepted_case_file(path: str) -> bool:
    """True if path is an accepted case YAML (top-level or pending)."""
    if not path.endswith(".yaml"):
        return False
    # cases/*.yaml (top-level only: no slash in the "*" part)
    if path.startswith("cases/") and path.count("/") == 1:
        return True
    # cases/pending/*.yaml
    if path.startswith("cases/pending/") and path.count("/") == 2:
        return True
    return False


def _is_accepted_plasma_surface_file(path: str) -> bool:
    """True if path is under plasma_surfaces/."""
    return path.startswith("plasma_surfaces/")


def _validate_plasma_surface_file(path: Path, repo_root: Path) -> list[str]:
    """Basic validation for plasma surface files. Returns error list (empty if OK)."""
    errors: list[str] = []
    if not path.exists():
        errors.append(f"{path}: file not found")
        return errors
    if not path.is_file():
        errors.append(f"{path}: not a regular file")
        return errors
    # Reasonable size limit (e.g. 50 MB)
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > 50:
        errors.append(f"{path}: file too large ({size_mb:.1f} MB > 50 MB)")
    return errors


def validate_case_only_pr(
    changed_files: list[str],
    maintainer_bypass: bool,
    repo_root: Path | None = None,
) -> tuple[bool, list[str], list[str], list[str]]:
    """Validate a case-only (or case+plasma_surfaces) PR.

    Parameters
    ----------
    changed_files : list[str]
        List of changed file paths (from git diff --name-only).
    maintainer_bypass : bool
        If True, non-case/plasma files are ignored (maintainer in CASE_ONLY_MAINTAINERS).
    repo_root : Path, optional
        Repository root. Defaults to current working directory.

    Returns
    -------
    ok : bool
        True if validation passed (case-only and all valid).
    case_yamls : list[str]
        Accepted case YAML paths.
    non_case : list[str]
        Rejected (non-case, non-plasma) paths.
    errors : list[str]
        Validation error messages.
    """
    repo_root = repo_root or Path.cwd()
    case_yamls: list[str] = []
    plasma_files: list[str] = []
    non_case: list[str] = []
    errors: list[str] = []

    for f in changed_files:
        f = f.strip()
        if not f:
            continue
        if _is_accepted_case_file(f):
            case_yamls.append(f)
        elif _is_accepted_plasma_surface_file(f):
            plasma_files.append(f)
        else:
            if not maintainer_bypass:
                non_case.append(f)

    if non_case:
        return False, case_yamls, non_case, []

    if not case_yamls and not plasma_files:
        return True, [], [], []

    # Validate case YAMLs
    try:
        from stellcoilbench.validate_config import validate_case_yaml_file
    except ImportError as e:
        return False, case_yamls, non_case, [f"Import error: {e}"]

    for f in case_yamls:
        p = repo_root / f
        if not p.exists():
            continue
        errs = validate_case_yaml_file(p, surfaces_dir=repo_root / "plasma_surfaces")
        if errs:
            errors.extend(errs)

    # Validate plasma surface files
    for f in plasma_files:
        p = repo_root / f
        errs = _validate_plasma_surface_file(p, repo_root)
        if errs:
            errors.extend(errs)

    ok = len(errors) == 0
    return ok, case_yamls, non_case, errors


def _main_env() -> int:
    """Run from workflow environment (CHANGED_FILES, MAINTAINER_BYPASS)."""
    changed_raw = os.environ.get("CHANGED_FILES", "")
    bypass = os.environ.get("MAINTAINER_BYPASS", "false").lower() in ("true", "1", "yes")
    changed = [f.strip() for f in changed_raw.split("\n") if f.strip()]
    ok, case_yamls, non_case, errors = validate_case_only_pr(changed, maintainer_bypass=bypass)
    if non_case:
        print("Not a case-only PR. Files outside cases/ or plasma_surfaces/:")
        for f in non_case:
            print(f"  {f}")
        print("Skipping auto-merge.")
        return 0
    if not case_yamls and not any("plasma_surfaces" in f for f in changed):
        print("No case YAML or plasma surface files changed.")
        return 0
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        return 1
    print("All case and plasma surface files valid.")
    return 0


def _main_cli(args: argparse.Namespace) -> int:
    """Run from CLI (--files or --base/--head)."""
    if args.files:
        changed = list(args.files)
    else:
        base = args.base or "main"
        head = args.head or "HEAD"
        result = subprocess.run(
            ["git", "diff", "--name-only", base, head],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            print(f"git diff failed: {result.stderr}", file=sys.stderr)
            return 1
        changed = [f.strip() for f in result.stdout.split("\n") if f.strip()]
    ok, case_yamls, non_case, errors = validate_case_only_pr(
        changed, maintainer_bypass=args.maintainer_bypass
    )
    print("=== Case-only PR validation ===")
    print(f"Changed files: {len(changed)}")
    print(f"Case YAMLs: {case_yamls}")
    print(f"Non-case (rejected): {non_case}")
    if errors:
        print("Validation errors:")
        for e in errors:
            print(f"  {e}")
        print("Result: FAIL")
        return 1
    if non_case:
        print("Result: REJECT (mixed PR)")
        return 0
    print("Result: PASS")
    return 0


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Validate case-only or case+plasma_surfaces PRs.")
    parser.add_argument("--files", nargs="*", help="List of changed files (simulate PR)")
    parser.add_argument("--base", default="main", help="Base ref for git diff (default: main)")
    parser.add_argument("--head", default="HEAD", help="Head ref for git diff (default: HEAD)")
    parser.add_argument("--maintainer-bypass", action="store_true", help="Simulate maintainer bypass")
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Run in CI mode (use CHANGED_FILES, MAINTAINER_BYPASS env vars)",
    )
    args = parser.parse_args()

    if args.ci or (not args.files and os.environ.get("CHANGED_FILES") is not None):
        return _main_env()
    return _main_cli(args)


if __name__ == "__main__":
    sys.exit(main())
