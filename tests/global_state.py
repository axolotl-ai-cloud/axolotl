"""Name the tests that leave process-global registries, patches or torch config changed.

Each test's state is snapshotted before setup and compared after teardown, so a later test
that only fails in one order (full suite vs testmon subset, one xdist worker vs another) can
be traced to the test that leaked. Teardown also finalizes module/class/session fixtures as
their scope ends, so a scoped fixture's changes land on the first and last test of its scope.

Leaks are listed in a ``[global-state]`` terminal section; with
``AXOLOTL_CI_ENFORCE_GLOBAL_STATE=1`` each one is also a teardown error of that test. Mark a
test ``@pytest.mark.allows_global_state`` when the process-wide change is its point.
"""

import os
import re
import sys

import pytest

ENFORCE_ENV = "AXOLOTL_CI_ENFORCE_GLOBAL_STATE"
MARKER = "allows_global_state"

_PROPERTY = "axolotl_global_state_leak"
_BEFORE = pytest.StashKey[dict]()
_LEAK = pytest.StashKey[str]()

# Classes whose methods axolotl's kernel/adapter registration replaces process-wide.
_PATCHED_CLASSES = (
    ("transformers.modeling_utils", "PreTrainedModel"),
    ("peft.tuners.lora.layer", "ParamWrapper"),
)
# Every ModelLoader.load() installs these whatever the cfg, so only a process's first load
# changes them and which test that is depends on the order.
_INSTALL_ONCE = {
    "transformers.modeling_utils.PreTrainedModel": {
        "_initialize_weights",
        "resize_token_embeddings",
    },
}
_TORCH_CONFIGS = (
    "torch._dynamo.config",
    "torch._inductor.config",
    "torch._functorch.config",
)
_SCANNED_PREFIXES = ("axolotl.", "transformers.")
_FLAG_RE = re.compile(r"^_[A-Z0-9_]*PATCHED[A-Z0-9_]*$")
_HIDDEN = "<deleted>"
_MAX_CHANGES = 6


def _subclasses(cls):
    found, stack = [cls], [cls]
    while stack:
        for sub in stack.pop().__subclasses__():
            if sub not in found:
                found.append(sub)
                stack.append(sub)
    return found


def _torch_config_state(config) -> dict:
    state = config.get_config_copy()
    # deleting a key (e.g. a monkeypatch undo) hides it: every later read raises
    for key, entry in getattr(config, "_config", {}).items():
        if getattr(entry, "hide", False):
            state[key] = _HIDDEN
    return state


def _differs(old, new) -> bool:
    if old is new:
        return False
    try:
        return bool(old != new)
    except Exception:  # pylint: disable=broad-exception-caught
        return True


def _is_scalar(value) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def _change(key, old: dict, new: dict) -> str:
    if key not in old:
        value = new[key]
        return f"+{key}={value!r}" if _is_scalar(value) else f"+{key}"
    if key not in new:
        return f"-{key}"
    if _is_scalar(old[key]) and _is_scalar(new[key]):
        return f"{key}: {old[key]!r} -> {new[key]!r}"
    return f"{key} replaced"


def describe_changes(before: dict, after: dict) -> str:
    """Changes between two snapshots; state that appeared mid-test (a first import) is skipped."""
    parts = []
    for watch in sorted(before.keys() & after.keys()):
        old, new = before[watch], after[watch]
        keys = sorted(
            key
            for key in old.keys() | new.keys()
            if key not in old or key not in new or _differs(old[key], new[key])
        )
        if keys:
            changes = ", ".join(_change(key, old, new) for key in keys[:_MAX_CHANGES])
            more = (
                f" (+{len(keys) - _MAX_CHANGES} more)"
                if len(keys) > _MAX_CHANGES
                else ""
            )
            parts.append(f"{watch}: {changes}{more}")
    return "; ".join(parts)


class GlobalStateGuard:
    """Pytest plugin; see the module docstring."""

    def __init__(self, config):
        self._config = config
        self._enforce = os.environ.get(ENFORCE_ENV) == "1"
        self._leaks: dict[str, str] = {}
        self._scanned: set[str] = set()
        self._module_count = 0
        self._flags: list[tuple[str, str]] = []
        self._interfaces: list[tuple[str, str]] = []

    def _discover(self) -> None:
        """Find ``_*PATCHED*`` flags and registry singletons in newly imported modules."""
        if len(sys.modules) == self._module_count:
            return
        self._module_count = len(sys.modules)
        generic = sys.modules.get("transformers.utils.generic")
        interface_cls = getattr(generic, "GeneralInterface", None)
        for name, module in list(sys.modules.items()):
            if name in self._scanned or not name.startswith(_SCANNED_PREFIXES):
                continue
            self._scanned.add(name)
            for attr, value in list(getattr(module, "__dict__", {}).items()):
                if _FLAG_RE.match(attr) and (value is None or isinstance(value, bool)):
                    self._flags.append((name, attr))
                elif interface_cls is not None and isinstance(value, interface_cls):
                    self._interfaces.append((name, attr))

    def snapshot(self) -> dict:
        self._discover()
        state: dict[str, dict] = {}
        for module_name, cls_name in _PATCHED_CLASSES:
            cls = getattr(sys.modules.get(module_name), cls_name, None)
            if cls is not None:
                name = f"{module_name}.{cls_name}"
                skipped = _INSTALL_ONCE.get(name, set())
                state[name] = {
                    # Python 3.14 caches lazily evaluated annotations in the class dict
                    key: value
                    for key, value in vars(cls).items()
                    if key not in skipped and not key.startswith("__annotat")
                }
        generic = sys.modules.get("transformers.utils.generic")
        if generic is not None:
            for cls in _subclasses(generic.GeneralInterface):
                mapping = vars(cls).get("_global_mapping")
                if isinstance(mapping, dict):
                    name = f"{cls.__module__}.{cls.__qualname__}._global_mapping"
                    state[name] = dict(mapping)
        for module_name, attr in self._interfaces:
            interface = getattr(sys.modules.get(module_name), attr, None)
            local = getattr(interface, "_local_mapping", None)
            if isinstance(local, dict):
                state[f"{module_name}.{attr}._local_mapping"] = dict(local)
        for module_name, attr in self._flags:
            module = sys.modules.get(module_name)
            if module is not None:
                state[f"{module_name}.{attr}"] = {attr: getattr(module, attr, None)}
        for name in _TORCH_CONFIGS:
            config = sys.modules.get(name)
            if config is not None:
                state[name] = _torch_config_state(config)
        return state

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_setup(self, item):
        if item.get_closest_marker(MARKER) is None:
            item.stash[_BEFORE] = self.snapshot()

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_teardown(self, item, nextitem):  # pylint: disable=unused-argument
        result = yield
        before = item.stash.get(_BEFORE, None)
        if before is not None:
            leak = describe_changes(before, self.snapshot())
            if leak:
                item.stash[_LEAK] = leak
                if self._enforce:
                    pytest.fail(
                        f"left process-global state changed (mark it "
                        f"@pytest.mark.{MARKER} if intended): {leak}",
                        pytrace=False,
                    )
        return result

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_makereport(self, item, call):
        report = yield
        if call.when == "teardown" and _LEAK in item.stash:
            report.user_properties.append((_PROPERTY, item.stash[_LEAK]))
        return report

    def pytest_runtest_logreport(self, report):
        for key, leak in report.user_properties:
            if key == _PROPERTY:
                self._leaks[report.nodeid] = leak

    def pytest_terminal_summary(self, terminalreporter):
        if hasattr(self._config, "workerinput") or not self._leaks:
            return
        terminalreporter.write_sep(
            "-",
            f"[global-state] {len(self._leaks)} test(s) left process-global state changed",
        )
        for nodeid, leak in sorted(self._leaks.items()):
            terminalreporter.write_line(f"[global-state] {nodeid}: {leak}")
