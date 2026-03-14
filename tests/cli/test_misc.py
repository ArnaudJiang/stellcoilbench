"""Miscellaneous CLI tests (zip submission directory, pre-zip cleanup, etc.)."""

import zipfile

from stellcoilbench.cli import _zip_submission_directory
from stellcoilbench.submission_packaging import _remove_pre_zip_artifacts


def test_zip_submission_directory_creates_zip_and_removes_dir(tmp_path):
    submission_dir = tmp_path / "run"
    submission_dir.mkdir()
    (submission_dir / "results.json").write_text("{}")
    (submission_dir / "nested").mkdir()
    (submission_dir / "nested" / "file.txt").write_text("data")
    (submission_dir / "plot.pdf").write_text("pdf content")

    zip_path = _zip_submission_directory(submission_dir)

    assert zip_path.exists()
    assert zip_path == submission_dir / "all_files.zip"
    assert submission_dir.exists()
    assert (submission_dir / "plot.pdf").exists()
    assert not (submission_dir / "results.json").exists()
    assert not (submission_dir / "nested").exists()
    with zipfile.ZipFile(zip_path, "r") as zf:
        assert "results.json" in zf.namelist()
        assert "nested/file.txt" in zf.namelist()
        assert "plot.pdf" not in zf.namelist()


def test_zip_submission_directory_missing_dir(tmp_path):
    submission_dir = tmp_path / "missing"
    zip_path = _zip_submission_directory(submission_dir)
    assert zip_path == submission_dir / "all_files.zip"


def test_zip_submission_directory_empty_dir(tmp_path):
    submission_dir = tmp_path / "empty"
    submission_dir.mkdir()
    zip_path = _zip_submission_directory(submission_dir)
    assert zip_path == submission_dir / "all_files.zip"
    assert not zip_path.exists()


class TestZipSubmissionDirectoryComprehensive:
    """Additional _zip_submission_directory tests."""

    def test_zip_includes_vtk_files(self, tmp_path):
        from stellcoilbench.cli import _zip_submission_directory

        submission_dir = tmp_path / "submission"
        submission_dir.mkdir()
        (submission_dir / "coils.vtu").write_text("VTK data")
        (submission_dir / "surface.vts").write_text("VTK surface data")

        zip_path = _zip_submission_directory(submission_dir)

        assert zip_path.exists()
        with zipfile.ZipFile(zip_path, "r") as zf:
            files_in_zip = zf.namelist()
            assert "coils.vtu" in files_in_zip
            assert "surface.vts" in files_in_zip

    def test_finite_build_coils_only_in_zip(self, tmp_path):
        from stellcoilbench.cli import _zip_submission_directory

        submission_dir = tmp_path / "submission"
        submission_dir.mkdir()
        (submission_dir / "finite_build_coils.vtk").write_text("VTK sweep data")
        (submission_dir / "coils.json").write_text("{}")

        zip_path = _zip_submission_directory(submission_dir)

        assert zip_path.exists()
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            assert "finite_build_coils.vtk" in names
        assert not (submission_dir / "finite_build_coils.vtk").exists()

    def test_remove_pre_zip_artifacts_deletes_vmec_and_qfm_files(self, tmp_path):
        """_remove_pre_zip_artifacts deletes wout*, input.*, qfm_surface.vts before zip."""
        submission_dir = tmp_path / "submission"
        submission_dir.mkdir()
        (submission_dir / "results.json").write_text("{}")
        (submission_dir / "biot_savart_optimized.json").write_text("{}")
        (submission_dir / "wout_xyz.nc").write_text("vmec output")
        (submission_dir / "input.LandremanPaul2021_QA").write_text("vmec input")
        (submission_dir / "qfm_surface.vts").write_text("qfm vtk")
        order_dir = submission_dir / "order_8"
        order_dir.mkdir()
        (order_dir / "wout_abc.nc").write_text("nested wout")

        _remove_pre_zip_artifacts(submission_dir)

        assert (submission_dir / "results.json").exists()
        assert (submission_dir / "biot_savart_optimized.json").exists()
        assert not (submission_dir / "wout_xyz.nc").exists()
        assert not (submission_dir / "input.LandremanPaul2021_QA").exists()
        assert not (submission_dir / "qfm_surface.vts").exists()
        assert not (order_dir / "wout_abc.nc").exists()
