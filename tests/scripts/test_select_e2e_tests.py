"""Unit tests for cicd/select_e2e_tests.py against synthetic repositories."""

import importlib.util
import subprocess  # nosec B404
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "cicd" / "select_e2e_tests.py"
_spec = importlib.util.spec_from_file_location("select_e2e_tests", SCRIPT)
assert _spec is not None and _spec.loader is not None
mod = importlib.util.module_from_spec(_spec)
sys.modules["select_e2e_tests"] = (
    mod  # dataclasses resolve postponed annotations via sys.modules
)
_spec.loader.exec_module(mod)

SCHEMA = """
class Cfg:
    base_model: str
    sequence_len: int
    lora_r: int
    gradient_checkpointing: bool
"""
BUILDER = "def build(cfg):\n    return cfg.base_model, cfg.sequence_len\n"
ADAPTER = (
    "from axolotl.utils.collators import pad\n\ndef load(cfg):\n    return cfg.lora_r\n"
)
COLLATORS = "def pad(x):\n    return x\n"
OFFLOAD = "def offload(cfg):\n    return cfg.gradient_checkpointing\n"
KD_ARGS = "class KDArgs:\n    kd_temperature: float\n"
KD_PLUGIN = "def hook(cfg):\n    return cfg.kd_temperature, cfg.sequence_len\n"
STRATEGY = "def load(cfg):\n    return cfg.sequence_len\n"


def _cfg_test(name: str, extra: str = "", values: str = "") -> str:
    return f'''
from axolotl.utils.dict import DictDefault

def test_{name}():
    cfg = DictDefault({{
        "base_model": "org/{name}-model",
        "sequence_len": 512,
        "datasets": [{{"path": "ds", "type": "{values or "alpaca"}"}}],
        {extra}
    }})
    assert cfg
'''


def _git(repo: Path, *args: str) -> None:
    subprocess.run(  # nosec B603 B607
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        env={
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
            "HOME": str(repo),
            "PATH": "/usr/bin:/bin",
        },
    )


def _write(repo: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


@pytest.fixture(name="repo")
def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    _write(
        repo,
        {
            "pyproject.toml": "[project]\nname='axolotl'\n",
            "docs/readme.md": "hi\n",
            "src/axolotl/__init__.py": "",
            "src/axolotl/utils/__init__.py": "",
            "src/axolotl/utils/dict.py": "class DictDefault(dict):\n    pass\n",
            "src/axolotl/utils/collators.py": COLLATORS,
            "src/axolotl/utils/schemas/__init__.py": "",
            "src/axolotl/utils/schemas/config.py": SCHEMA,
            "src/axolotl/core/__init__.py": "",
            "src/axolotl/core/builder.py": BUILDER,
            "src/axolotl/loaders/__init__.py": "",
            "src/axolotl/loaders/adapter.py": ADAPTER,
            "src/axolotl/monkeypatch/__init__.py": "",
            "src/axolotl/monkeypatch/offload.py": OFFLOAD,
            "src/axolotl/prompt_strategies/__init__.py": "",
            "src/axolotl/prompt_strategies/alpaca.py": STRATEGY,
            "src/axolotl/prompt_strategies/chat_template.py": STRATEGY,
            "src/axolotl/prompt_strategies/sharegpt.py": STRATEGY,
            "src/axolotl/prompt_strategies/completion.py": STRATEGY,
            "src/axolotl/integrations/__init__.py": "",
            "src/axolotl/integrations/kd/__init__.py": "",
            "src/axolotl/integrations/kd/args.py": KD_ARGS,
            "src/axolotl/integrations/kd/plugin.py": KD_PLUGIN,
            "src/axolotl/integrations/liger/__init__.py": "",
            "src/axolotl/integrations/liger/plugin.py": "def hook(cfg):\n    return 1\n",
            "src/axolotl/integrations/cce/__init__.py": "",
            "src/axolotl/integrations/cce/plugin.py": "def hook(cfg):\n    return 1\n",
            "tests/conftest.py": "import pytest\n",
            "tests/e2e/test_sft.py": _cfg_test("sft", values="chat_template"),
            "tests/e2e/test_lora.py": _cfg_test(
                "lora", '"lora_r": 8, "gradient_checkpointing": True,'
            ),
            "tests/e2e/test_kd.py": _cfg_test(
                "kd",
                '"plugins": ["axolotl.integrations.kd.KDPlugin"], "kd_temperature": 1.0,',
            ),
            "tests/e2e/test_liger.py": _cfg_test(
                "liger", '"plugins": ["axolotl.integrations.liger.LigerPlugin"],'
            ),
            "tests/e2e/multigpu/test_fsdp.py": _cfg_test("fsdp"),
            "tests/e2e/test_completion.py": _cfg_test(
                "completion", values="completion"
            ),
            "tests/e2e/test_cce.py": _cfg_test(
                "cce", '"plugins": ["axolotl.integrations.cce.CCEPlugin"],'
            ),
        },
    )
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    _git(repo, "checkout", "-qb", "feature")
    return repo


def _select(
    repo: Path,
    change: dict[str, str],
    delete: tuple[str, ...] = (),
    message: str = "change",
):
    _write(repo, change)
    for rel in delete:
        (repo / rel).unlink()
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", message)
    selector = mod.Selector(repo, "tests/e2e", exclude=("tests/e2e/multigpu",))
    return selector.select("main", merge_commit=False)


def test_feature_key_selects_only_tests_setting_it(repo):
    sel = _select(repo, {"src/axolotl/loaders/adapter.py": ADAPTER + "\n# touched\n"})
    assert sel.mode == "subset"
    assert sel.tests == ["tests/e2e/test_lora.py"]


def test_ubiquitous_key_reader_runs_everything(repo):
    sel = _select(repo, {"src/axolotl/core/builder.py": BUILDER + "\n# touched\n"})
    assert sel.mode == "all"
    assert "ubiquitous" in sel.reason


def test_edgeless_module_inherits_from_importer(repo):
    sel = _select(repo, {"src/axolotl/utils/collators.py": COLLATORS + "\n# touched\n"})
    assert sel.mode == "subset"
    assert sel.tests == ["tests/e2e/test_lora.py"]


def test_plugin_named_by_config_value_wins_over_ubiquity(repo):
    sel = _select(
        repo, {"src/axolotl/integrations/kd/plugin.py": KD_PLUGIN + "\n# touched\n"}
    )
    assert sel.mode == "subset"
    assert sel.tests == ["tests/e2e/test_kd.py"]


def test_brand_new_plugin_needs_no_registration(repo):
    sel = _select(
        repo,
        {
            "src/axolotl/integrations/newthing/__init__.py": "",
            "src/axolotl/integrations/newthing/args.py": "class NewArgs:\n    newthing_alpha: float\n",
            "src/axolotl/integrations/newthing/plugin.py": "def hook(cfg):\n    return cfg.newthing_alpha, cfg.base_model\n",
            "tests/e2e/test_newthing.py": _cfg_test(
                "newthing",
                '"plugins": ["axolotl.integrations.newthing.NewPlugin"], "newthing_alpha": 0.5,',
            ),
        },
    )
    assert sel.mode == "subset"
    assert sel.tests == ["tests/e2e/test_newthing.py"]


def test_registry_leaf_nobody_names_selects_nothing(repo):
    sel = _select(
        repo, {"src/axolotl/prompt_strategies/sharegpt.py": STRATEGY + "\n# touched\n"}
    )
    assert sel.mode == "none"


def test_registry_leaf_named_by_dataset_type(repo):
    sel = _select(
        repo,
        {"src/axolotl/prompt_strategies/chat_template.py": STRATEGY + "\n# touched\n"},
    )
    assert sel.mode == "subset"
    assert sel.tests == ["tests/e2e/test_sft.py"]


def test_changed_test_file_selects_itself(repo):
    sel = _select(repo, {"tests/e2e/test_liger.py": _cfg_test("liger", '"x": 1,')})
    assert sel.mode == "subset"
    assert sel.tests == ["tests/e2e/test_liger.py"]


def test_out_of_scope_test_change_selects_nothing(repo):
    sel = _select(
        repo, {"tests/e2e/multigpu/test_fsdp.py": _cfg_test("fsdp", '"x": 1,')}
    )
    assert sel.mode == "none"


def test_docs_only_selects_nothing(repo):
    sel = _select(repo, {"docs/readme.md": "changed\n"})
    assert sel.mode == "none"


@pytest.mark.parametrize(
    "change",
    [
        {"pyproject.toml": "[project]\nname='axolotl'\nversion='1'\n"},
        {"tests/conftest.py": "import pytest\n# touched\n"},
        {"tests/e2e/utils.py": "def helper():\n    return 1\n"},
    ],
)
def test_shared_files_run_everything(repo, change):
    sel = _select(repo, change)
    assert sel.mode == "all"


def test_deleted_module_runs_everything(repo):
    sel = _select(repo, {}, delete=("src/axolotl/monkeypatch/offload.py",))
    assert sel.mode == "all"
    assert "deleted module" in sel.reason


def test_orphan_module_runs_everything(repo):
    sel = _select(
        repo, {"src/axolotl/monkeypatch/offload.py": OFFLOAD + "\n# touched\n"}
    )
    # reads a discriminating key, so it selects the tests setting it even without importers
    assert sel.mode == "subset"
    assert sel.tests == ["tests/e2e/test_lora.py"]
    sel = _select(
        repo, {"src/axolotl/monkeypatch/orphan.py": "def f():\n    return 1\n"}
    )
    assert sel.mode == "all"
    assert "no importers" in sel.reason


def test_commit_tag_forces_full_run(repo):
    sel = _select(repo, {"docs/readme.md": "x\n"}, message="docs [test all]")
    assert sel.mode == "all"


def test_run_all_reason_lists_every_trigger(repo):
    sel = _select(
        repo,
        {
            "pyproject.toml": "[project]\nname='axolotl'\nversion='1'\n",
            "src/axolotl/core/builder.py": BUILDER + "#\n",
        },
    )
    assert sel.mode == "all"
    assert "pyproject.toml" in sel.reason and "builder.py" in sel.reason


def test_declared_keys_are_authoritative(repo):
    sel = _select(
        repo,
        {
            "src/axolotl/monkeypatch/offload.py": '__ci_config_keys__ = ("gradient_checkpointing",)\n'
            + "def offload(cfg):\n    return cfg.base_model\n"
        },
    )
    assert sel.mode == "subset"
    assert sel.tests == ["tests/e2e/test_lora.py"]


def test_declared_unknown_key_runs_everything(repo):
    sel = _select(
        repo,
        {
            "src/axolotl/monkeypatch/offload.py": '__ci_config_keys__ = ("no_such_key",)\n'
            + OFFLOAD
        },
    )
    assert sel.mode == "all"
    assert "no_such_key" in sel.reason


def _grow_base(repo: Path, files: dict[str, str]) -> None:
    """Add files to the base branch so they are not part of the feature diff."""
    _write(repo, files)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base grows")
    _git(repo, "branch", "-f", "main", "HEAD")


WORKER_TEST = (
    "from pathlib import Path\n\n"
    "def test_launch():\n"
    '    worker = Path(__file__).with_name("_fast_worker.py")\n'
    "    assert worker.exists()\n"
)


def test_worker_script_config_counts_for_its_test(repo):
    _grow_base(
        repo,
        {
            "tests/e2e/_fast_worker.py": _cfg_test(
                "worker", '"gradient_checkpointing": True,'
            ),
            "tests/e2e/test_worker_launcher.py": WORKER_TEST,
        },
    )
    sel = _select(
        repo, {"src/axolotl/monkeypatch/offload.py": OFFLOAD + "\n# touched\n"}
    )
    assert sel.mode == "subset"
    assert sel.tests == ["tests/e2e/test_lora.py", "tests/e2e/test_worker_launcher.py"]


def test_changed_worker_selects_the_tests_that_launch_it(repo):
    _grow_base(
        repo,
        {
            "tests/e2e/_fast_worker.py": _cfg_test("worker"),
            "tests/e2e/test_worker_launcher.py": WORKER_TEST,
        },
    )
    sel = _select(repo, {"tests/e2e/_fast_worker.py": _cfg_test("worker", '"x": 1,')})
    assert sel.mode == "subset"
    assert sel.tests == ["tests/e2e/test_worker_launcher.py"]


def test_opaque_test_rides_along_with_every_subset(repo):
    _grow_base(
        repo, {"tests/e2e/test_opaque.py": "def test_opaque():\n    assert True\n"}
    )
    sel = _select(repo, {"src/axolotl/loaders/adapter.py": ADAPTER + "\n# touched\n"})
    assert sel.mode == "subset"
    assert sel.tests == ["tests/e2e/test_lora.py", "tests/e2e/test_opaque.py"]


def test_opaque_test_does_not_turn_none_into_subset(repo):
    _grow_base(
        repo, {"tests/e2e/test_opaque.py": "def test_opaque():\n    assert True\n"}
    )
    sel = _select(repo, {"docs/readme.md": "changed again\n"})
    assert sel.mode == "none"
