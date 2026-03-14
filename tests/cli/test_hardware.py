"""Tests for hardware and GitHub username detection."""

import builtins
import subprocess
import sys
import pytest

from stellcoilbench.cli import _detect_github_username, _detect_hardware
from tests.cli.conftest import _FakeCompletedProcess


# --- GitHub username parametrization ---


def _setup_github_remote_https(monkeypatch):
    def fake_run(cmd, **kwargs):
        if isinstance(cmd, list) and "remote" in cmd:
            return _FakeCompletedProcess(
                returncode=0, stdout="https://github.com/bob/repo.git\n"
            )
        return _FakeCompletedProcess(returncode=1, stdout="")

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.delenv("GITHUB_ACTOR", raising=False)
    monkeypatch.delenv("GITHUB_USER", raising=False)


def _setup_github_remote_ssh(monkeypatch):
    def fake_run(cmd, **kwargs):
        if isinstance(cmd, list) and "remote" in cmd:
            return _FakeCompletedProcess(
                returncode=0, stdout="git@github.com:alice/repo.git\n"
            )
        return _FakeCompletedProcess(returncode=1, stdout="")

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.delenv("GITHUB_ACTOR", raising=False)
    monkeypatch.delenv("GITHUB_USER", raising=False)


def _setup_github_env_actor(monkeypatch):
    def fake_run(cmd, **kwargs):
        raise FileNotFoundError()

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setenv("GITHUB_ACTOR", "env_user")
    monkeypatch.delenv("GITHUB_USER", raising=False)


def _setup_github_empty(monkeypatch):
    def fake_run(cmd, **kwargs):
        raise FileNotFoundError()

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.delenv("GITHUB_ACTOR", raising=False)
    monkeypatch.delenv("GITHUB_USER", raising=False)


def _setup_github_env_user(monkeypatch):
    def fake_run(cmd, **kwargs):
        return _FakeCompletedProcess(returncode=1, stdout="")

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.delenv("GITHUB_ACTOR", raising=False)
    monkeypatch.setenv("GITHUB_USER", "githubuser")


def _setup_github_git_timeout(monkeypatch):
    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, timeout=2)

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.delenv("GITHUB_ACTOR", raising=False)
    monkeypatch.delenv("GITHUB_USER", raising=False)


def _setup_github_git_not_found(monkeypatch):
    def fake_run(cmd, **kwargs):
        raise FileNotFoundError("git command not found")

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.delenv("GITHUB_ACTOR", raising=False)
    monkeypatch.delenv("GITHUB_USER", raising=False)


def _setup_github_non_github_url(monkeypatch):
    def fake_run(cmd, **kwargs):
        if isinstance(cmd, list) and "remote" in cmd:
            return _FakeCompletedProcess(
                returncode=0, stdout="https://gitlab.com/user/repo.git\n"
            )
        return _FakeCompletedProcess(returncode=1, stdout="")

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.delenv("GITHUB_ACTOR", raising=False)
    monkeypatch.delenv("GITHUB_USER", raising=False)


@pytest.mark.parametrize(
    "setup_fn,expected_username",
    [
        (_setup_github_remote_https, "bob"),
        (_setup_github_remote_ssh, "alice"),
        (_setup_github_env_actor, "env_user"),
        (_setup_github_empty, ""),
        (_setup_github_env_user, "githubuser"),
        (_setup_github_git_timeout, ""),
        (_setup_github_git_not_found, ""),
        (_setup_github_non_github_url, ""),
    ],
    ids=[
        "remote_url_https",
        "remote_url_ssh",
        "env_GITHUB_ACTOR",
        "empty",
        "env_GITHUB_USER",
        "git_timeout",
        "git_not_found",
        "non_github_url",
    ],
)
def test_detect_github_username(monkeypatch, setup_fn, expected_username):
    """_detect_github_username returns username from remote URL or env vars."""
    setup_fn(monkeypatch)
    assert _detect_github_username() == expected_username


# --- Hardware parametrization ---


def _setup_hardware_darwin_cpu_gpu_ram(monkeypatch):
    def fake_run(cmd, **kwargs):
        if cmd[0] == "sysctl":
            return _FakeCompletedProcess(returncode=0, stdout="Test CPU\n")
        if cmd[0] == "nvidia-smi":
            return _FakeCompletedProcess(returncode=0, stdout="GPU1\nGPU2\n")
        return _FakeCompletedProcess(returncode=1, stdout="")

    class _Mem:
        total = 8 * 1024**3

    class _Psutil:
        @staticmethod
        def virtual_memory():
            return _Mem()

    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("platform.processor", lambda: "Fallback CPU")
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setitem(sys.modules, "psutil", _Psutil())


def _setup_hardware_linux_cpu_model(monkeypatch):
    def fake_run(cmd, **kwargs):
        if cmd[0] == "lscpu":
            return _FakeCompletedProcess(returncode=0, stdout="Model name: Fancy CPU\n")
        if cmd[0] == "nvidia-smi":
            return _FakeCompletedProcess(returncode=1, stdout="")
        return _FakeCompletedProcess(returncode=1, stdout="")

    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("platform.processor", lambda: "")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.delitem(sys.modules, "psutil", raising=False)


def _setup_hardware_cpu_exception(monkeypatch):
    def fake_run(cmd, **kwargs):
        raise FileNotFoundError("command not found")

    monkeypatch.setattr("platform.processor", lambda: "")
    monkeypatch.setattr("platform.machine", lambda: "")
    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.delitem(sys.modules, "psutil", raising=False)


def _setup_hardware_gpu_exception(monkeypatch):
    def fake_run(cmd, **kwargs):
        if cmd[0] == "nvidia-smi":
            raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=2)
        if cmd[0] == "lscpu":
            return _FakeCompletedProcess(returncode=1, stdout="")
        return _FakeCompletedProcess(returncode=1, stdout="")

    monkeypatch.setattr("platform.processor", lambda: "TestCPU")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.delitem(sys.modules, "psutil", raising=False)


def _setup_hardware_psutil_import_error(monkeypatch):
    def fake_run(cmd, **kwargs):
        return _FakeCompletedProcess(returncode=1, stdout="")

    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("no psutil")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("platform.processor", lambda: "TestCPU")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    monkeypatch.setattr("platform.system", lambda: "Other")
    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr(builtins, "__import__", mock_import)


def _setup_hardware_detailed_cpu_exception(monkeypatch):
    def fake_run(cmd, **kwargs):
        if cmd[0] == "sysctl":
            raise subprocess.TimeoutExpired(cmd="sysctl", timeout=2)
        if cmd[0] == "nvidia-smi":
            return _FakeCompletedProcess(returncode=1, stdout="")
        return _FakeCompletedProcess(returncode=1, stdout="")

    monkeypatch.setattr("platform.processor", lambda: "TestCPU")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.delitem(sys.modules, "psutil", raising=False)


def _setup_hardware_detect_cpu(monkeypatch):
    monkeypatch.setattr("platform.processor", lambda: "Intel Core i7")


def _setup_hardware_detect_platform(monkeypatch):
    pass  # No mocks; uses real platform


def _setup_hardware_detect_various_platforms_linux(monkeypatch):
    monkeypatch.setattr("platform.processor", lambda: "x86_64")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    monkeypatch.setattr("platform.system", lambda: "Linux")


def _setup_hardware_detect_various_platforms_darwin(monkeypatch):
    monkeypatch.setattr("platform.processor", lambda: "x86_64")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    monkeypatch.setattr("platform.system", lambda: "Darwin")


def _assert_hardware_contains(hardware, substrings, not_substrings=None):
    for s in substrings:
        assert s in hardware, f"Expected '{s}' in hardware string: {hardware!r}"
    if not_substrings:
        for s in not_substrings:
            assert s not in hardware, f"Expected '{s}' not in hardware: {hardware!r}"


def _assert_hardware_is_string(hardware):
    assert isinstance(hardware, str)


def _assert_hardware_non_empty(hardware):
    assert len(hardware) > 0


def _assert_hardware_cpu_or_intel(hardware):
    assert "Intel Core i7" in hardware or "CPU" in hardware


@pytest.mark.parametrize(
    "setup_fn,assert_fn",
    [
        (
            _setup_hardware_darwin_cpu_gpu_ram,
            lambda h: _assert_hardware_contains(
                h, ["CPU: Test CPU", "GPU: GPU1, GPU2", "RAM: 8.0GB"]
            ),
        ),
        (
            _setup_hardware_linux_cpu_model,
            lambda h: _assert_hardware_contains(h, ["CPU: Fancy CPU"]),
        ),
        (_setup_hardware_cpu_exception, _assert_hardware_is_string),
        (
            _setup_hardware_gpu_exception,
            lambda h: _assert_hardware_contains(h, ["CPU: TestCPU"], ["GPU"]),
        ),
        (
            _setup_hardware_psutil_import_error,
            lambda h: _assert_hardware_contains(h, ["CPU: TestCPU"], ["RAM"]),
        ),
        (
            _setup_hardware_detailed_cpu_exception,
            lambda h: _assert_hardware_contains(h, ["CPU: TestCPU"]),
        ),
        (_setup_hardware_detect_cpu, _assert_hardware_cpu_or_intel),
        (_setup_hardware_detect_platform, _assert_hardware_non_empty),
        (_setup_hardware_detect_various_platforms_linux, _assert_hardware_non_empty),
        (_setup_hardware_detect_various_platforms_darwin, _assert_hardware_non_empty),
    ],
    ids=[
        "darwin_cpu_gpu_ram",
        "linux_cpu_model",
        "cpu_exception",
        "gpu_exception",
        "psutil_import_error",
        "detailed_cpu_exception",
        "detect_cpu",
        "detect_platform",
        "detect_various_platforms_linux",
        "detect_various_platforms_darwin",
    ],
)
def test_detect_hardware(monkeypatch, setup_fn, assert_fn):
    """_detect_hardware reports CPU, GPU, RAM under various platform/command scenarios."""
    setup_fn(monkeypatch)
    hardware = _detect_hardware()
    assert_fn(hardware)
