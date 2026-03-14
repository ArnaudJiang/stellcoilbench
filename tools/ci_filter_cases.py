#!/usr/bin/env python3
"""
Filter case YAML files into those that need to run vs those with successful submissions.

Reads case file list from stdin (one path per line) or discovers cases via --cases-dir.
Outputs JSON with to_run and already_successful lists.

Usage:
  find cases -name "*.yaml" -not -path "cases/done/*" -not -path "cases/pending/*" | python tools/ci_filter_cases.py
  python tools/ci_filter_cases.py --cases-dir cases
  python tools/ci_filter_cases.py --cases-dir cases --github-output  # write to GITHUB_OUTPUT
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import yaml
import zipfile


# Fields added during submission that should be ignored when comparing
SUBMISSION_FIELDS = ["source_case_file", "submission_timestamp", "submission_user"]


def normalize_yaml_content(content: str, strip_submission_fields: bool = False) -> str:
    """Normalize YAML content for comparison (sort keys, consistent formatting)."""
    try:
        data = yaml.safe_load(content)
        if strip_submission_fields and isinstance(data, dict):
            for field in SUBMISSION_FIELDS:
                data.pop(field, None)
        return yaml.dump(data, sort_keys=True, default_flow_style=False)
    except Exception:
        return content


def case_has_successful_submission(case_file_path: str | Path) -> bool:
    """
    Return True if a submission zip exists with matching case content and results.json.
    """
    case_path = Path(case_file_path)
    if not case_path.exists():
        return False
    try:
        current_case_content = case_path.read_text()
        current_case_normalized = normalize_yaml_content(current_case_content)
    except Exception:
        return False
    try:
        repo_root = Path.cwd()
        case_file_rel = str(case_path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        case_file_rel = str(case_path.resolve())
    case_file_rel_normalized = case_file_rel.replace("\\", "/")
    submissions_dir = Path("submissions")
    if not submissions_dir.exists():
        return False
    zip_files = list(submissions_dir.rglob("*.zip"))
    if len(zip_files) == 0:
        return False
    for zip_path in zip_files:
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                if (
                    "case.yaml" not in zf.namelist()
                    or "results.json" not in zf.namelist()
                ):
                    continue
                zip_case_content = zf.read("case.yaml").decode("utf-8")
                zip_case_data = yaml.safe_load(zip_case_content)
                submission_source = zip_case_data.get("source_case_file", "")
                submission_source_normalized = submission_source.replace("\\", "/")
                source_filename = Path(submission_source_normalized).name
                case_filename = Path(case_file_rel_normalized).name
                if (
                    submission_source_normalized == case_file_rel_normalized
                    or source_filename == case_filename
                ):
                    zip_case_normalized = normalize_yaml_content(
                        zip_case_content, strip_submission_fields=True
                    )
                    if zip_case_normalized == current_case_normalized:
                        return True
        except Exception:
            continue
    return False


def discover_cases(cases_dir: Path) -> List[str]:
    """Find .yaml case files, excluding done/ and pending/."""
    cases: List[str] = []
    for p in sorted(cases_dir.rglob("*.yaml")):
        rel = str(p.relative_to(cases_dir))
        if rel.startswith("done/") or rel.startswith("pending/"):
            continue
        cases.append(str(p))
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Filter case files into to_run vs already_successful (JSON output)"
    )
    parser.add_argument(
        "--cases-dir",
        type=Path,
        default=None,
        help="Discover cases from this directory (excludes done/ and pending/)",
    )
    parser.add_argument(
        "--github-output",
        action="store_true",
        help="Write cases_to_run_json and has_cases to GITHUB_OUTPUT",
    )
    args = parser.parse_args()

    if args.cases_dir is not None:
        if not args.cases_dir.exists():
            print(
                f"ERROR: Cases directory does not exist: {args.cases_dir}",
                file=sys.stderr,
            )
            return 1
        case_files = discover_cases(args.cases_dir)
    else:
        case_files = [line.strip() for line in sys.stdin if line.strip()]

    if not case_files:
        print("ERROR: No .yaml files found in cases/ directory!", file=sys.stderr)
        return 1

    to_run = [cf for cf in case_files if not case_has_successful_submission(cf)]
    already_successful = [cf for cf in case_files if case_has_successful_submission(cf)]
    result = {"to_run": to_run, "already_successful": already_successful}

    if args.github_output and (gh := os.environ.get("GITHUB_OUTPUT")):
        with open(gh, "a", encoding="utf-8") as f:
            f.write(f"cases_to_run_json={json.dumps(to_run)}\n")
            f.write(f"has_cases={'true' if to_run else 'false'}\n")
        if not to_run:
            print("No cases need to be run - all have successful submissions")
        else:
            print(f"Cases to run: {json.dumps(to_run)}")
    else:
        print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
