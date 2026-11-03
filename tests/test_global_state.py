import torch._dynamo.config as dynamo_config

from tests.global_state import ENFORCE_ENV, GlobalStateGuard, describe_changes

pytest_plugins = ["pytester"]


def test_snapshot_names_registry_entries_and_torch_config(monkeypatch):
    from transformers.integrations.moe import ExpertsInterface

    guard = GlobalStateGuard(config=None)
    before = guard.snapshot()
    monkeypatch.setitem(ExpertsInterface._global_mapping, "_leak_probe", len)
    limit = dynamo_config.recompile_limit
    with dynamo_config.patch(suppress_errors=not dynamo_config.suppress_errors):
        delattr(dynamo_config, "recompile_limit")
        try:
            leak = describe_changes(before, guard.snapshot())
        finally:
            dynamo_config.recompile_limit = limit

    assert (
        "transformers.integrations.moe.ExpertsInterface._global_mapping: +_leak_probe"
        in leak
    )
    assert f"recompile_limit: {limit!r} -> '<deleted>'" in leak
    assert "suppress_errors: " in leak


def test_enforced_leak_errors_the_leaking_test_only(pytester, monkeypatch):
    monkeypatch.setenv(ENFORCE_ENV, "1")
    pytester.makeconftest(
        """
        from tests.global_state import GlobalStateGuard

        def pytest_configure(config):
            config.addinivalue_line("markers", "allows_global_state: exempt")
            config.pluginmanager.register(GlobalStateGuard(config), "global-state")
        """
    )
    pytester.makepyfile(
        test_leak="""
        import pytest
        import torch._dynamo.config as dynamo_config

        def test_leaks():
            dynamo_config.suppress_errors = not dynamo_config.suppress_errors

        @pytest.mark.allows_global_state
        def test_restores_on_purpose():
            dynamo_config.suppress_errors = not dynamo_config.suppress_errors

        def test_clean():
            pass
        """
    )
    # pytest-retry's pytest_configure raises in an in-process (inline) run
    result = pytester.runpytest("-p", "no:pytest-retry", "-p", "no:randomly")

    result.assert_outcomes(passed=3, errors=1)
    result.stdout.fnmatch_lines(
        [
            "*ERROR at teardown of test_leaks*",
            "[[]global-state[]] test_leak.py::test_leaks: torch._dynamo.config: "
            "suppress_errors: *",
        ]
    )
