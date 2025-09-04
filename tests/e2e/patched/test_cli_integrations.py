"""
test cases to make sure the plugin args are loaded from the config file
"""

from pathlib import Path

import yaml

from axolotl.cli.config import load_cfg
from axolotl.utils.dict import DictDefault


# pylint: disable=duplicate-code
class TestPluginArgs:
    """
    test class for plugin args loaded from the config file
    """

    def test_liger_plugin_args(self, temp_dir):
        test_cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "learning_rate": 0.000001,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    },
                ],
                "plugins": ["axolotl.integrations.liger.LigerPlugin"],
                "liger_layer_norm": True,
                "liger_rope": True,
                "liger_rms_norm": False,
                "liger_glu_activation": True,
                "liger_fused_linear_cross_entropy": True,
            }
        )

        with open(Path(temp_dir) / "config.yaml", "w", encoding="utf-8") as fout:
            fout.write(yaml.dump(test_cfg.to_dict()))
        cfg = load_cfg(str(Path(temp_dir) / "config.yaml"))
        assert cfg.liger_layer_norm is True
        assert cfg.liger_rope is True
        assert cfg.liger_rms_norm is False
        assert cfg.liger_glu_activation is True
        assert cfg.liger_fused_linear_cross_entropy is True
