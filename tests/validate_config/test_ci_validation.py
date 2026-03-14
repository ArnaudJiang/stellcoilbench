"""
Unit tests for validate_ci_case, validate_ci_case_file, and autopilot smoke tests.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tests.assert_helpers import assert_errors_contain
from stellcoilbench.validate_config import validate_ci_case, validate_ci_case_file

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class TestValidateCiCase:
    """Tests for validate_ci_case (CI autopilot case JSON validation)."""

    def _base_ci_case(self):
        return {
            "case_id": "test_case_1",
            "case_config": {
                "description": "Test",
                "surface_params": {"surface": "input.LandremanPaul2021_QA"},
                "coils_params": {"ncoils": 4, "order": 16},
                "optimizer_params": {"algorithm": "l-bfgs"},
            },
        }

    def test_valid_ci_case(self):
        """Valid CI case passes validation."""
        errors = validate_ci_case(self._base_ci_case())
        assert errors == []

    def test_missing_case_id(self):
        """Missing case_id fails."""
        data = self._base_ci_case()
        del data["case_id"]
        errors = validate_ci_case(data)
        assert_errors_contain(errors, "case_id")

    def test_empty_case_id(self):
        """Empty case_id fails."""
        data = self._base_ci_case()
        data["case_id"] = ""
        errors = validate_ci_case(data)
        assert_errors_contain(errors, "case_id")

    def test_resource_max_iterations_exceeds_cap(self):
        """resource.max_total_iterations exceeding policy cap fails."""
        data = self._base_ci_case()
        data["resource"] = {"max_total_iterations": 50000}
        policy = {"resource_caps": {"max_total_iterations": 10000}}
        errors = validate_ci_case(data, policy=policy)
        assert_errors_contain(errors, "exceeds cap")

    def test_resource_timeout_outside_range(self):
        """resource.timeout_minutes outside allowed range fails."""
        data = self._base_ci_case()
        data["resource"] = {"timeout_minutes": 1}
        policy = {
            "resource_caps": {
                "timeout_minutes_min": 5,
                "timeout_minutes_max": 180,
            }
        }
        errors = validate_ci_case(data, policy=policy)
        assert_errors_contain(errors, "outside allowed range")

    def test_parent_ids_not_list(self):
        """parent_ids must be a list."""
        data = self._base_ci_case()
        data["parent_ids"] = "not a list"
        errors = validate_ci_case(data)
        assert_errors_contain(errors, "parent_ids")

    def test_tags_not_list(self):
        """tags must be a list."""
        data = self._base_ci_case()
        data["tags"] = "not a list"
        errors = validate_ci_case(data)
        assert_errors_contain(errors, "tags")

    def test_random_seed_not_int(self):
        """random_seed must be an integer."""
        data = self._base_ci_case()
        data["random_seed"] = "42"
        errors = validate_ci_case(data)
        assert_errors_contain(errors, "random_seed")

    def test_missing_case_config(self):
        """Missing case_config fails."""
        data = self._base_ci_case()
        del data["case_config"]
        errors = validate_ci_case(data)
        assert_errors_contain(errors, "case_config")


class TestValidateCiCaseFile:
    """Tests for validate_ci_case_file."""

    def test_json_parse_error(self, tmp_path):
        """Invalid JSON returns parse error."""
        f = tmp_path / "bad.json"
        f.write_text("{ invalid json")
        errors = validate_ci_case_file(f)
        assert len(errors) > 0
        assert "JSON" in errors[0]

    def test_root_not_dict(self, tmp_path):
        """Root element must be a JSON object."""
        f = tmp_path / "array.json"
        f.write_text("[1, 2, 3]")
        errors = validate_ci_case_file(f)
        assert_errors_contain(errors, "object")

    def test_valid_file(self, tmp_path):
        """Valid case file passes validation."""
        case = {
            "case_id": "test_1",
            "resource": {"max_total_iterations": 5000, "timeout_minutes": 60},
            "case_config": {
                "description": "test",
                "surface_params": {"surface": "input.LandremanPaul2021_QA"},
                "coils_params": {"ncoils": 4, "order": 8},
                "optimizer_params": {"algorithm": "L-BFGS-B", "max_iterations": 2000},
            },
        }
        f = tmp_path / "case.json"
        f.write_text(json.dumps(case))
        errors = validate_ci_case_file(f)
        assert errors == []


class TestProposeBatchSmoke:
    """Smoke test for propose_batch."""

    def test_propose_batch_minimal_ctx(self):
        import sys

        sys.path.insert(0, str(_REPO_ROOT / "tools"))
        from propose_batch import propose_batch
        from build_context import build_context

        done_dir = _REPO_ROOT / "cases" / "done"
        policy_path = _REPO_ROOT / "policy" / "proposer_policy.yaml"
        ctx = build_context(done_dir, policy_path)
        policy = ctx.get("policy") or {}
        batch = propose_batch(ctx, policy, batch_size=1, seed=42)
        assert isinstance(batch, list)
        assert len(batch) <= 1


class TestRunCiCaseSmoke:
    """Smoke test for run-ci-case CLI."""

    def test_run_ci_case_help(self):
        result = subprocess.run(
            ["stellcoilbench", "run-ci-case", "--help"],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
        )
        assert result.returncode == 0
        assert "run-ci-case" in result.stdout or "run_ci_case" in result.stdout
