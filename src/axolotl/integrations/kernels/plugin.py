import importlib

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
                    "`pip install git+https://github.com/Dao-AILab/sonic-moe@3e3f36eeefc324b10042db22921bfe9ab53ed2d1`"
                )
            from axolotl.integrations.kernels.sonicmoe import patch_sonicmoe

            LOG.info(
                f"Applying SonicMoE patches for model type: {cfg.model_config_type}"
            )
            patch_sonicmoe(cfg.model_config_type)

    def _register_kernels(self):
        from kernels import (
            LayerRepository,
            Mode,
            register_kernel_mapping,
        )

        register_kernel_mapping(
            {
                "HFScatterMoEParallelExperts": {
                    "cuda": {
                        Mode.TRAINING: LayerRepository(
                            repo_id="axolotl-ai-co/scattermoe",
                            layer_name="HFScatterMoEGatedMLP",
                        ),
                        Mode.INFERENCE: LayerRepository(
                            repo_id="axolotl-ai-co/scattermoe",
                            layer_name="HFScatterMoEGatedMLP",
                        ),
                    },
                }
            }
        )

    def _kernelize_model(self, model_type: str):
        from kernels import replace_kernel_forward_from_hub

        from axolotl.integrations.kernels.constants import resolve_moe_block_cls

        model_moe_cls = resolve_moe_block_cls(model_type)
        replace_kernel_forward_from_hub(model_moe_cls, "HFScatterMoEParallelExperts")
