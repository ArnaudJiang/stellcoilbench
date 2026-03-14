"""Tests for version info detection."""

import sys
import types

import pytest

from stellcoilbench.cli import _get_version_info
from tests.cli.conftest import _FakeCompletedProcess


def _setup_git_exception(monkeypatch):
    """Setup: git command raises FileNotFoundError."""

    def fake_run(cmd, **kwargs):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr("subprocess.run", fake_run)


def _setup_simsopt_not_installed(monkeypatch):
    """Setup: subprocess works, simsopt import fails."""

    def fake_run(cmd, **kwargs):
        return _FakeCompletedProcess(returncode=0, stdout="abc123\n")

    monkeypatch.setattr("subprocess.run", fake_run)

    import builtins

    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "simsopt":
            raise ImportError("no simsopt")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)


def _setup_simsopt_no_file(monkeypatch):
    """Setup: simsopt has no __file__."""

    def fake_run(cmd, **kwargs):
        return _FakeCompletedProcess(returncode=0, stdout="abc123\n")

    monkeypatch.setattr("subprocess.run", fake_run)

    fake_simsopt = types.ModuleType("simsopt")
    fake_simsopt.__version__ = "1.0.0"
    fake_simsopt.__file__ = None
    monkeypatch.setitem(sys.modules, "simsopt", fake_simsopt)


def _setup_simsopt_editable(monkeypatch, tmp_path):
    """Setup: simsopt installed from source with .git dir."""

    simsopt_pkg = tmp_path / "simsopt"
    simsopt_pkg.mkdir()
    (tmp_path / ".git").mkdir()
    init_file = simsopt_pkg / "__init__.py"
    init_file.write_text("")

    call_count = {"n": 0}

    def fake_run(cmd, **kwargs):
        call_count["n"] += 1
        if "rev-parse" in cmd and "HEAD" in cmd and "--abbrev-ref" not in cmd:
            return _FakeCompletedProcess(returncode=0, stdout="deadbeef\n")
        elif "--abbrev-ref" in cmd:
            return _FakeCompletedProcess(returncode=0, stdout="main\n")
        elif "remote" in cmd:
            return _FakeCompletedProcess(
                returncode=0, stdout="https://github.com/user/simsopt.git\n"
            )
        return _FakeCompletedProcess(returncode=0, stdout="abc\n")

    monkeypatch.setattr("subprocess.run", fake_run)

    fake_simsopt = types.ModuleType("simsopt")
    fake_simsopt.__version__ = "0.1.dev100+gabcdef"
    fake_simsopt.__file__ = str(init_file)
    monkeypatch.setitem(sys.modules, "simsopt", fake_simsopt)


def _setup_simsopt_git_exception(monkeypatch, tmp_path):
    """Setup: simsopt from source but git fails inside simsopt dir."""

    simsopt_pkg = tmp_path / "simsopt"
    simsopt_pkg.mkdir()
    (tmp_path / ".git").mkdir()
    init_file = simsopt_pkg / "__init__.py"
    init_file.write_text("")

    call_idx = {"n": 0}

    def fake_run(cmd, **kwargs):
        call_idx["n"] += 1
        if call_idx["n"] <= 2:
            return _FakeCompletedProcess(returncode=0, stdout="abc\n")
        raise OSError("git failed")

    monkeypatch.setattr("subprocess.run", fake_run)

    fake_simsopt = types.ModuleType("simsopt")
    fake_simsopt.__version__ = "1.0.0"
    fake_simsopt.__file__ = str(init_file)
    monkeypatch.setitem(sys.modules, "simsopt", fake_simsopt)


def _assert_git_exception(info):
    assert info["stellcoilbench_commit"] == "unknown"


def _assert_simsopt_not_installed(info):
    assert info["simsopt_version"] == "not installed"


def _assert_simsopt_no_file(info):
    assert info["simsopt_version"] == "1.0.0"
    assert "simsopt_branch" not in info


def _assert_simsopt_editable(info):
    assert info["simsopt_version"] == "0.1.dev100+gabcdef"
    assert info["simsopt_commit"] == "deadbeef"
    assert info["simsopt_branch"] == "main"
    assert info["simsopt_remote"] == "https://github.com/user/simsopt.git"


def _assert_simsopt_git_exception(info):
    assert info["simsopt_version"] == "1.0.0"


@pytest.mark.parametrize(
    "scenario,setup_fn,assert_fn,needs_tmp_path",
    [
        ("git_exception", _setup_git_exception, _assert_git_exception, False),
        (
            "simsopt_not_installed",
            _setup_simsopt_not_installed,
            _assert_simsopt_not_installed,
            False,
        ),
        ("simsopt_no_file", _setup_simsopt_no_file, _assert_simsopt_no_file, False),
        ("simsopt_editable", _setup_simsopt_editable, _assert_simsopt_editable, True),
        (
            "simsopt_git_exception",
            _setup_simsopt_git_exception,
            _assert_simsopt_git_exception,
            True,
        ),
    ],
    ids=[
        "git_exception",
        "simsopt_not_installed",
        "simsopt_no_file",
        "simsopt_editable",
        "simsopt_git_exception",
    ],
)
def test_get_version_info(
    monkeypatch, tmp_path, scenario, setup_fn, assert_fn, needs_tmp_path
):
    """_get_version_info handles various scenarios: git failure, missing simsopt, no __file__, editable install, simsopt git failure."""
    if needs_tmp_path:
        setup_fn(monkeypatch, tmp_path)
    else:
        setup_fn(monkeypatch)
    info = _get_version_info()
    assert_fn(info)
