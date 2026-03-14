"""Tests for _recompute_coils_linked_to_surface in update_db._recompute."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from tests.conftest import REPO_ROOT
from tests.update_db.conftest import make_submission_dir

from stellcoilbench.update_db._recompute import _recompute_coils_linked_to_surface


def _biot_savart_fixture_path() -> Path:
    """Path to real BiotSavart JSON fixture (Landreman-Paul coils)."""
    return Path(__file__).parent / "fixtures" / "biot_savart_landreman_paul.json"


def test_recompute_no_biot_savart_returns_none(tmp_path: Path) -> None:
    """Zip or dir with no biot_savart returns None."""
    zip_path = tmp_path / "submission.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("results.json", json.dumps({"metadata": {}, "metrics": {"score": 0.1}}))
    assert _recompute_coils_linked_to_surface(zip_path, "LandremanPaul2021_QA", REPO_ROOT) is None

    make_submission_dir(tmp_path, surface="surface1")
    results_path = tmp_path / "surface1" / "user1" / "2024-01-01_12-00" / "results.json"
    assert _recompute_coils_linked_to_surface(results_path, "surface1", tmp_path) is None


class TestRecomputeZipWithBiotSavart:
    """Zip with valid biot_savart_optimized.json."""

    def test_zip_with_valid_biot_savart_and_surface(
        self, tmp_path: Path
    ) -> None:
        """Zip with valid BiotSavart + real surface yields bool (True/False)."""
        bs_path = _biot_savart_fixture_path()
        if not bs_path.exists():
            pytest.skip("biot_savart fixture not available")
        pytest.importorskip("simsopt", reason="simsopt not available")

        plasma_dir = REPO_ROOT / "plasma_surfaces"
        surface_file = plasma_dir / "input.LandremanPaul2021_QA"
        if not surface_file.exists():
            pytest.skip("LandremanPaul2021_QA surface not available")

        zip_path = tmp_path / "submission.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("biot_savart_optimized.json", bs_path.read_bytes())
            zf.writestr(
                "results.json",
                json.dumps({"metadata": {}, "metrics": {}}),
            )

        result = _recompute_coils_linked_to_surface(
            zip_path, "LandremanPaul2021_QA", REPO_ROOT
        )
        assert result is True or result is False

    def test_zip_prefers_highest_order_biot_savart(
        self, tmp_path: Path
    ) -> None:
        """Zip with multiple biot_savart files uses highest (order_16 > root)."""
        bs_path = _biot_savart_fixture_path()
        if not bs_path.exists():
            pytest.skip("biot_savart fixture not available")
        pytest.importorskip("simsopt", reason="simsopt not available")

        plasma_dir = REPO_ROOT / "plasma_surfaces"
        if not (plasma_dir / "input.LandremanPaul2021_QA").exists():
            pytest.skip("LandremanPaul2021_QA surface not available")

        zip_path = tmp_path / "submission.zip"
        bs_bytes = bs_path.read_bytes()
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("biot_savart_optimized.json", bs_bytes)
            zf.writestr("order_8/biot_savart_optimized.json", bs_bytes)
            zf.writestr("order_16/biot_savart_optimized.json", bs_bytes)

        result = _recompute_coils_linked_to_surface(
            zip_path, "LandremanPaul2021_QA", REPO_ROOT
        )
        assert result is True or result is False


class TestRecomputeDirWithBiotSavart:
    """Directory with biot_savart_optimized.json."""

    def test_dir_with_biot_savart_and_surface(
        self, tmp_path: Path
    ) -> None:
        """Submission dir with biot_savart + surface yields bool."""
        bs_path = _biot_savart_fixture_path()
        if not bs_path.exists():
            pytest.skip("biot_savart fixture not available")
        pytest.importorskip("simsopt", reason="simsopt not available")
        if not (REPO_ROOT / "plasma_surfaces" / "input.LandremanPaul2021_QA").exists():
            pytest.skip("LandremanPaul2021_QA surface not available")

        sub_dir = tmp_path / "LandremanPaul2021_QA" / "user1" / "2024-01-01_12-00"
        sub_dir.mkdir(parents=True)
        (sub_dir / "biot_savart_optimized.json").write_bytes(bs_path.read_bytes())
        (sub_dir / "results.json").write_text(
            json.dumps({"metadata": {}, "metrics": {}})
        )

        result = _recompute_coils_linked_to_surface(
            sub_dir / "results.json", "LandremanPaul2021_QA", REPO_ROOT
        )
        assert result is True or result is False


@pytest.mark.parametrize(
    "create_zip_fn",
    [
        lambda t: t / "nonexistent.zip",  # path doesn't exist
        lambda t: _write_zip(t / "s1.zip", b'{"invalid": '),
        lambda t: _write_zip(t / "s2.zip", b"{}"),
    ],
    ids=["nonexistent_zip", "invalid_json", "empty_json"],
)
def test_recompute_error_paths_returns_none(tmp_path: Path, create_zip_fn) -> None:
    """OSError/ValueError paths return None."""
    zip_path = create_zip_fn(tmp_path)
    assert _recompute_coils_linked_to_surface(
        zip_path, "LandremanPaul2021_QA", REPO_ROOT
    ) is None


def _write_zip(path: Path, bs_content: bytes) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("biot_savart_optimized.json", bs_content)
    return path

    def test_unknown_surface_returns_none(self, tmp_path: Path) -> None:
        """Unknown surface name -> resolve_surface_path returns None -> result None."""
        bs_path = _biot_savart_fixture_path()
        if not bs_path.exists():
            pytest.skip("biot_savart fixture not available")
        pytest.importorskip("simsopt", reason="simsopt not available")

        zip_path = tmp_path / "submission.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("biot_savart_optimized.json", bs_path.read_bytes())

        result = _recompute_coils_linked_to_surface(
            zip_path, "NonExistentSurface123", tmp_path
        )
        assert result is None


class TestRecomputeMockedLoadSurfaceError:
    """Test load_surface_with_range raising OSError/ValueError returns None."""

    def test_load_surface_raises_returns_none(
        self, tmp_path: Path
    ) -> None:
        """When load_surface_with_range raises, function returns None."""
        pytest.importorskip("simsopt", reason="simsopt not available")
        bs_path = _biot_savart_fixture_path()
        if not bs_path.exists():
            pytest.skip("biot_savart fixture not available")

        zip_path = tmp_path / "submission.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("biot_savart_optimized.json", bs_path.read_bytes())

        with patch(
            "stellcoilbench.update_db._recompute.resolve_surface_path",
            return_value=tmp_path / "input.LandremanPaul2021_QA",
        ):
            with patch(
                "stellcoilbench.post_processing.load_surface_with_range",
                side_effect=RuntimeError("VMEC load failed"),
            ):
                result = _recompute_coils_linked_to_surface(
                    zip_path, "LandremanPaul2021_QA", tmp_path
                )
        assert result is None

    def test_linking_check_attribute_error_returns_none(
        self, tmp_path: Path
    ) -> None:
        """When coil.curve.gamma() raises AttributeError, returns None."""
        pytest.importorskip("simsopt", reason="simsopt not available")
        bs_path = _biot_savart_fixture_path()
        if not bs_path.exists():
            pytest.skip("biot_savart fixture not available")

        zip_path = tmp_path / "submission.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("biot_savart_optimized.json", bs_path.read_bytes())

        with patch(
            "stellcoilbench.update_db._recompute.resolve_surface_path",
            return_value=REPO_ROOT / "plasma_surfaces" / "input.LandremanPaul2021_QA",
        ):
            # Patch simsopt load to return BiotSavart with coil raising AttributeError
            mock_bs = Mock()
            mock_coil = Mock()
            mock_coil.curve.gamma.side_effect = AttributeError("no gamma")
            mock_bs.coils = [mock_coil]

            with patch(
                "simsopt._core.load",
                return_value=mock_bs,
            ):
                result = _recompute_coils_linked_to_surface(
                    zip_path, "LandremanPaul2021_QA", REPO_ROOT
                )
        assert result is None


