"""KD plugin collator hook."""

from axolotl.integrations.base import BasePlugin, PluginManager
from axolotl.integrations.kd import KDPlugin
from axolotl.utils.dict import DictDefault


class FallbackCollatorPlugin(BasePlugin):
    def get_collator_cls_and_kwargs(self, cfg, is_eval=False):
        return dict, {"from": "fallback"}


def test_kd_collator_hook_defers_without_kd_trainer():
    assert KDPlugin().get_collator_cls_and_kwargs(DictDefault({})) is None


def test_kd_plugin_without_kd_trainer_does_not_shadow_later_plugins():
    manager = PluginManager.get_instance()
    manager.plugins["kd"] = KDPlugin()
    manager.plugins["fallback"] = FallbackCollatorPlugin()

    assert manager.get_collator_cls_and_kwargs(DictDefault({})) == (
        dict,
        {"from": "fallback"},
    )
