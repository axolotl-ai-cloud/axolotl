import importlib
from pathlib import Path

from axolotl.integrations.base import BasePlugin
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class KernelsPlugin(BasePlugin):
    def get_input_args(self):
        return "axolotl.integrations.kernels.KernelsArgs"

    def pre_model_load(self, cfg):
        if cfg.use_scattermoe:
            self._register_kernels()
            self._kernelize_model(cfg.model_config_type)
        elif cfg.use_sonicmoe:
            if not importlib.util.find_spec("sonicmoe"):
                raise RuntimeError(
                    "SonicMoE is not installed. Install it with "
                    "`pip install git+https://github.com/Dao-AILab/sonic-moe@022992fef6a6aee53e0c3ba709e22f740cec547e`"
                )
            from axolotl.integrations.kernels.sonicmoe import patch_sonicmoe

            LOG.info(
                f"Applying SonicMoE patches for model type: {cfg.model_config_type}"
            )
            patch_sonicmoe(
                cfg.model_config_type,
                torch_compile=bool(getattr(cfg, "torch_compile", False)),
            )

    def _register_kernels(self):
        from kernels import (
            LocalLayerRepository,
            Mode,
            register_kernel_mapping,
        )

        plugin_root = Path(__file__).parent
        register_kernel_mapping(
            {
                "HFScatterMoEParallelExperts": {
                    "cuda": {
                        Mode.TRAINING: LocalLayerRepository(
                            repo_path=plugin_root / "libs" / "scattermoe_lora",
                            package_name="scattermoe_lora",
                            layer_name="HFScatterMoEGatedMLP",
                        ),
                        Mode.INFERENCE: LocalLayerRepository(
                            repo_path=plugin_root / "libs" / "scattermoe_lora",
                            package_name="scattermoe_lora",
                            layer_name="HFScatterMoEGatedMLP",
                        ),
                    },
                }
            }
        )

    def _kernelize_model(self, model_type: str):
        from kernels import replace_kernel_forward_from_hub

        from axolotl.integrations.kernels.constants import resolve_moe_block_classes

        for model_moe_cls in resolve_moe_block_classes(model_type):
            replace_kernel_forward_from_hub(
                model_moe_cls, "HFScatterMoEParallelExperts"
            )
