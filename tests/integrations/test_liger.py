"""
config validation tests for swiglu args
"""

from typing import Optional

import pytest

from axolotl.utils.config import prepare_plugins, validate_config
from axolotl.utils.dict import DictDefault


@pytest.fixture(autouse=True)
def _isolate_liger_kernel_impl_env():
    # register()/_set_kernel_impl write LIGER_KERNEL_IMPL into os.environ; without
    # this, the leak makes the first later import of liger_kernel.ops raise on CPU CI
    import os
    import sys

    saved_env = os.environ.get("LIGER_KERNEL_IMPL")
    saved_ops = sys.modules.get("liger_kernel.ops")
    yield
    if saved_env is None:
        os.environ.pop("LIGER_KERNEL_IMPL", None)
    else:
        os.environ["LIGER_KERNEL_IMPL"] = saved_env
    if saved_ops is None:
        sys.modules.pop("liger_kernel.ops", None)
    else:
        sys.modules["liger_kernel.ops"] = saved_ops


@pytest.fixture(name="minimal_liger_cfg")
def fixture_cfg():
    return DictDefault(
        {
            "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v0.6",
            "learning_rate": 0.000001,
            "datasets": [
                {
                    "path": "mhenrichsen/alpaca_2k_test",
                    "type": "alpaca",
                }
            ],
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "plugins": ["axolotl.integrations.liger.LigerPlugin"],
        }
    )


class TestPreparePluginsScope:
    def test_skips_plugins_retained_from_previous_configs(self, minimal_liger_cfg):
        from axolotl.integrations.base import PluginManager

        calls = []

        class StalePlugin:
            def register(self, cfg):
                calls.append(cfg)

        manager = PluginManager.get_instance()
        manager.plugins["tests.stale.StalePlugin"] = StalePlugin()
        try:
            prepare_plugins(DictDefault(dict(minimal_liger_cfg)))
        finally:
            manager.plugins.pop("tests.stale.StalePlugin", None)
        assert not calls


class TestValidationErrorCleanup:
    LIGER = "axolotl.integrations.liger.LigerPlugin"

    def test_rejected_config_restores_env(self, monkeypatch, minimal_liger_cfg):
        import os
        import sys

        from axolotl.integrations.base import PluginManager
        from axolotl.integrations.liger import plugin as liger_plugin

        monkeypatch.delitem(sys.modules, "liger_kernel.ops", raising=False)
        monkeypatch.delenv("LIGER_KERNEL_IMPL", raising=False)
        monkeypatch.setattr(liger_plugin, "_env_before_write", None)
        monkeypatch.setattr(liger_plugin, "_last_written", None)

        cfg = DictDefault({"liger_kernel_impl": "cutedsl"} | minimal_liger_cfg)
        prepare_plugins(cfg)
        assert os.environ["LIGER_KERNEL_IMPL"] == "cutedsl"

        PluginManager.get_instance().on_config_validation_error(cfg)
        assert "LIGER_KERNEL_IMPL" not in os.environ

    def test_cleanup_scoped_to_config_plugins(self, monkeypatch, minimal_liger_cfg):
        import os
        import sys

        from axolotl.integrations.base import PluginManager
        from axolotl.integrations.liger import plugin as liger_plugin

        monkeypatch.delitem(sys.modules, "liger_kernel.ops", raising=False)
        monkeypatch.delenv("LIGER_KERNEL_IMPL", raising=False)
        monkeypatch.setattr(liger_plugin, "_env_before_write", None)
        monkeypatch.setattr(liger_plugin, "_last_written", None)

        cfg = DictDefault({"liger_kernel_impl": "cutedsl"} | minimal_liger_cfg)
        prepare_plugins(cfg)

        # rejected config listed other plugins only: liger's write stays
        PluginManager.get_instance().on_config_validation_error(
            DictDefault({"plugins": ["some.other.Plugin"]})
        )
        assert os.environ["LIGER_KERNEL_IMPL"] == "cutedsl"


class TestValidation:
    """
    Test the validation module for liger
    """

    _caplog: Optional[pytest.LogCaptureFixture] = None

    @pytest.fixture(autouse=True)
    def inject_fixtures(self, caplog):
        caplog.set_level("WARNING")
        self._caplog = caplog

    def test_deprecated_swiglu(self, minimal_liger_cfg):
        test_cfg = DictDefault(
            {
                "liger_swiglu": False,
            }
            | minimal_liger_cfg
        )

        with self._caplog.at_level("WARNING", logger="axolotl.integrations.liger.args"):
            prepare_plugins(test_cfg)
            updated_cfg = validate_config(test_cfg)
            # TODO this test is brittle in CI
            # assert (
            #     "The 'liger_swiglu' argument is deprecated"
            #     in self._caplog.records[0].message
            # )
            assert updated_cfg.liger_swiglu is None
            assert updated_cfg.liger_glu_activation is False

    def test_conflict_swiglu_ligergluactivation(self, minimal_liger_cfg):
        test_cfg = DictDefault(
            {
                "liger_swiglu": False,
                "liger_glu_activation": True,
            }
            | minimal_liger_cfg
        )

        with pytest.raises(
            ValueError,
            match=r".*You cannot have both `liger_swiglu` and `liger_glu_activation` set.*",
        ):
            prepare_plugins(test_cfg)
            validate_config(test_cfg)

    def test_use_token_scaling_require_flce(self, minimal_liger_cfg):
        test_cfg = DictDefault(
            {
                "liger_fused_linear_cross_entropy": False,
                "liger_use_token_scaling": True,
            }
            | minimal_liger_cfg
        )

        with pytest.raises(
            ValueError,
            match=r"`liger_use_token_scaling: true` requires `liger_fused_linear_cross_entropy` enabled.",
        ):
            prepare_plugins(test_cfg)
            validate_config(test_cfg)

    def test_kernel_impl_accepted(self, minimal_liger_cfg):
        import sys

        # the plugin refuses to switch backends once liger_kernel.ops is imported;
        # an earlier test may have imported it (autouse fixture restores it after)
        sys.modules.pop("liger_kernel.ops", None)
        test_cfg = DictDefault({"liger_kernel_impl": "cutedsl"} | minimal_liger_cfg)

        prepare_plugins(test_cfg)
        updated_cfg = validate_config(test_cfg)
        assert updated_cfg.liger_kernel_impl == "cutedsl"

    def test_kernel_impl_rejects_unknown_backend(self, minimal_liger_cfg):
        test_cfg = DictDefault({"liger_kernel_impl": "triton2"} | minimal_liger_cfg)

        with pytest.raises(ValueError):
            prepare_plugins(test_cfg)
            validate_config(test_cfg)


class TestKernelImplEnv:
    """LIGER_KERNEL_IMPL env handling in LigerPlugin._set_kernel_impl"""

    def test_sets_env_before_liger_import(self, monkeypatch):
        import sys

        from axolotl.integrations.liger.plugin import LigerPlugin

        monkeypatch.delitem(sys.modules, "liger_kernel.ops", raising=False)
        monkeypatch.delenv("LIGER_KERNEL_IMPL", raising=False)

        LigerPlugin._set_kernel_impl("cutile")

        import os

        assert os.environ["LIGER_KERNEL_IMPL"] == "cutile"

    def test_omitted_option_undoes_previous_config_write(self, monkeypatch):
        import os
        import sys

        from axolotl.integrations.liger import plugin as liger_plugin
        from axolotl.integrations.liger.plugin import LigerPlugin
        from axolotl.utils.dict import DictDefault

        monkeypatch.delitem(sys.modules, "liger_kernel.ops", raising=False)
        monkeypatch.setenv("LIGER_KERNEL_IMPL", "cutile")
        monkeypatch.setattr(liger_plugin, "_env_before_write", None)
        monkeypatch.setattr(liger_plugin, "_last_written", None)

        # a config sets cutedsl (e.g. then fails validation), the next one omits it
        LigerPlugin().register(DictDefault({"liger_kernel_impl": "cutedsl"}))
        assert os.environ["LIGER_KERNEL_IMPL"] == "cutedsl"
        LigerPlugin().register(DictDefault({}))
        assert os.environ["LIGER_KERNEL_IMPL"] == "cutile"

    def test_omitted_option_keeps_foreign_env_value(self, monkeypatch):
        import os
        import sys

        from axolotl.integrations.liger import plugin as liger_plugin
        from axolotl.integrations.liger.plugin import LigerPlugin
        from axolotl.utils.dict import DictDefault

        monkeypatch.delitem(sys.modules, "liger_kernel.ops", raising=False)
        monkeypatch.setenv("LIGER_KERNEL_IMPL", "cutile")
        monkeypatch.setattr(liger_plugin, "_env_before_write", None)
        monkeypatch.setattr(liger_plugin, "_last_written", None)

        # user-set env value, no plugin write in this process: leave it alone
        LigerPlugin().register(DictDefault({}))
        assert os.environ["LIGER_KERNEL_IMPL"] == "cutile"

    def test_foreign_overwrite_becomes_restore_point(self, monkeypatch):
        import os
        import sys

        from axolotl.integrations.liger import plugin as liger_plugin
        from axolotl.integrations.liger.plugin import LigerPlugin
        from axolotl.utils.dict import DictDefault

        monkeypatch.delitem(sys.modules, "liger_kernel.ops", raising=False)
        monkeypatch.delenv("LIGER_KERNEL_IMPL", raising=False)
        monkeypatch.setattr(liger_plugin, "_env_before_write", None)
        monkeypatch.setattr(liger_plugin, "_last_written", None)

        # unset -> plugin cutile -> foreign cutedsl -> plugin ascend -> omitted
        LigerPlugin().register(DictDefault({"liger_kernel_impl": "cutile"}))
        os.environ["LIGER_KERNEL_IMPL"] = "cutedsl"
        LigerPlugin().register(DictDefault({"liger_kernel_impl": "ascend"}))
        LigerPlugin().register(DictDefault({}))
        assert os.environ["LIGER_KERNEL_IMPL"] == "cutedsl"

    def test_foreign_overwrite_relinquishes_ownership(self, monkeypatch):
        import os
        import sys

        from axolotl.integrations.liger import plugin as liger_plugin
        from axolotl.integrations.liger.plugin import LigerPlugin
        from axolotl.utils.dict import DictDefault

        monkeypatch.delitem(sys.modules, "liger_kernel.ops", raising=False)
        monkeypatch.delenv("LIGER_KERNEL_IMPL", raising=False)
        monkeypatch.setattr(liger_plugin, "_env_before_write", None)
        monkeypatch.setattr(liger_plugin, "_last_written", None)

        LigerPlugin().register(DictDefault({"liger_kernel_impl": "cutile"}))
        os.environ["LIGER_KERNEL_IMPL"] = "cutedsl"
        LigerPlugin().register(DictDefault({}))
        assert os.environ["LIGER_KERNEL_IMPL"] == "cutedsl"

    def test_already_imported_checks_loaded_backend_not_env(self, monkeypatch):
        import liger_kernel.ops.backends.registry as registry

        from axolotl.integrations.liger.plugin import LigerPlugin

        # env matches the request, but liger imported with the default backend
        monkeypatch.setenv("LIGER_KERNEL_IMPL", "cutedsl")
        monkeypatch.setattr(registry, "IMPL_REGISTRY", {})
        with pytest.raises(ValueError, match="already imported with backend"):
            LigerPlugin._set_kernel_impl("cutedsl")

    def test_register_hook_sets_env_from_raw_cfg(self, monkeypatch):
        import os
        import sys

        from axolotl.integrations.liger.plugin import LigerPlugin

        monkeypatch.delitem(sys.modules, "liger_kernel.ops", raising=False)
        monkeypatch.delenv("LIGER_KERNEL_IMPL", raising=False)

        LigerPlugin().register(DictDefault({"liger_kernel_impl": "cutedsl"}))

        assert os.environ["LIGER_KERNEL_IMPL"] == "cutedsl"

    @pytest.mark.parametrize("bad_value", [True, 1, "triton2", ["cutedsl"]])
    def test_register_hook_leaves_invalid_values_to_schema(
        self, monkeypatch, bad_value
    ):
        import os
        import sys

        from axolotl.integrations.liger.plugin import LigerPlugin

        monkeypatch.delitem(sys.modules, "liger_kernel.ops", raising=False)
        monkeypatch.delenv("LIGER_KERNEL_IMPL", raising=False)

        LigerPlugin().register(DictDefault({"liger_kernel_impl": bad_value}))

        assert "LIGER_KERNEL_IMPL" not in os.environ

    def test_utils_prepare_plugins_invokes_register(
        self, monkeypatch, minimal_liger_cfg
    ):
        import os
        import sys

        from axolotl.utils.config import prepare_plugins as utils_prepare_plugins

        monkeypatch.delitem(sys.modules, "liger_kernel.ops", raising=False)
        monkeypatch.delenv("LIGER_KERNEL_IMPL", raising=False)

        test_cfg = DictDefault({"liger_kernel_impl": "cutile"} | minimal_liger_cfg)
        utils_prepare_plugins(test_cfg)

        assert os.environ["LIGER_KERNEL_IMPL"] == "cutile"

    def test_raises_when_liger_already_imported_with_other_backend(self, monkeypatch):
        import sys
        import types

        from axolotl.integrations.liger.plugin import LigerPlugin

        monkeypatch.setitem(
            sys.modules, "liger_kernel.ops", types.ModuleType("liger_kernel.ops")
        )
        monkeypatch.delenv("LIGER_KERNEL_IMPL", raising=False)

        with pytest.raises(ValueError, match=r"already imported"):
            LigerPlugin._set_kernel_impl("cutedsl")

    def test_noop_when_liger_already_imported_with_same_backend(self, monkeypatch):
        import sys

        import liger_kernel.ops.backends.registry as registry

        from axolotl.integrations.liger.plugin import LigerPlugin

        # simulate liger having actually loaded cutedsl: its impl module is imported
        # (cutedsl may not self-register without nvidia-cutlass-dsl, so inject it)
        info = registry.ImplInfo(
            name="cutedsl", devices=("cuda",), module_path="tests.fake_cutedsl_ops"
        )
        monkeypatch.setitem(registry.IMPL_REGISTRY, "cutedsl", info)
        monkeypatch.setitem(sys.modules, info.module_path, sys)
        # cached submodule import doesn't re-import a popped parent package
        import liger_kernel.ops  # noqa: F401

        assert "liger_kernel.ops" in sys.modules

        LigerPlugin._set_kernel_impl("cutedsl")
