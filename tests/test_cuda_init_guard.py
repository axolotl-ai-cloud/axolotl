"""Tests for the collection-time CUDA init guard in tests/cuda_init_guard.py."""

import pytest
import torch

from tests.cuda_init_guard import OPT_OUT_ENV, CudaInitGuard

pytest_plugins = ["pytester"]


@pytest.fixture
def lazy_init_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: calls.append(1))
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    return calls


def _import_from(path, source, namespace=None):
    code = compile(source, str(path), "exec")
    exec(code, {} if namespace is None else namespace)  # nosec B102


def test_calls_are_attributed_to_the_importing_module(lazy_init_calls, tmp_path):
    vendor = tmp_path / "site-packages" / "vendor"
    query = "import torch\ndef capability():\n    torch.cuda._lazy_init()\n"
    allowed: dict = {}
    other: dict = {}
    guard = CudaInitGuard(allowed_sites=("vendor/specs.py",))
    guard.install()
    try:
        _import_from(vendor / "specs.py", query, allowed)
        _import_from(vendor / "archs.py", query, other)
        # reaches the allowed file below its own <module> frame, like bitsandbytes' cextension
        _import_from(vendor / "cextension.py", "capability()\n", allowed)
        _import_from(tmp_path / "tests" / "utils.py", "\ncapability()\n", other)
        _import_from(
            tmp_path / "tests" / "test_device.py",
            "import torch\n\ntorch.cuda._lazy_init()\n",
        )
    finally:
        guard.uninstall()

    assert len(lazy_init_calls) == 3
    assert torch.cuda._lazy_init.__name__ == "<lambda>"
    assert guard.violations(tmp_path) == [
        "tests/utils.py:2 (via site-packages/vendor/archs.py:3 in capability)",
        "tests/test_device.py:3",
    ]


def test_collection_fails_until_opted_out(pytester, monkeypatch, lazy_init_calls):
    monkeypatch.delenv(OPT_OUT_ENV, raising=False)
    pytester.makeconftest(
        """
        from tests.cuda_init_guard import CudaInitGuard

        guard = CudaInitGuard.from_env()

        def pytest_configure(config):
            if guard is not None:
                config.pluginmanager.register(guard, "cuda-init-guard")
        """
    )
    pytester.makepyfile(
        test_scratch="""
        import torch

        torch.cuda._lazy_init()

        def test_nothing():
            pass
        """
    )
    args = ("--collect-only", "-p", "no:pytest-retry", "-p", "no:randomly")

    result = pytester.runpytest(*args)

    assert result.ret == pytest.ExitCode.INTERRUPTED
    result.stdout.fnmatch_lines(
        ["*CUDA was initialized while importing*", "  test_scratch.py:3"]
    )

    monkeypatch.setenv(OPT_OUT_ENV, "1")
    assert pytester.runpytest(*args).ret == pytest.ExitCode.OK
