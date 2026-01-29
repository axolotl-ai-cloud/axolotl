from kernels import (
    LayerRepository,
    Mode,
    register_kernel_mapping,
    replace_kernel_forward_from_hub,
)

from axolotl.integrations.base import BasePlugin
from axolotl.utils.callbacks.models import get_causal_lm_model_cls_prefix


class KernelsPlugin(BasePlugin):
    def get_input_args(self):
        return "axolotl.integrations.kernels.KernelsArgs"

    def pre_model_load(self, cfg):
        if cfg.use_scattermoe:
            self._register_kernels()
            self._kernelize_model(cfg.model_config_type)

    def _register_kernels(self):
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
        if model_type == "olmoe":
            from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock

            replace_kernel_forward_from_hub(
                OlmoeSparseMoeBlock, "HFScatterMoEParallelExperts"
            )
        else:
            try:
                model_moe_cls = get_model_moe_block(model_type)
                replace_kernel_forward_from_hub(
                    model_moe_cls, "HFScatterMoEParallelExperts"
                )
            except Exception as err:
                raise ValueError(f"Unsupported model type: {model_type}") from err


def get_model_moe_block(model_type: str):
    module_path = f"transformers.models.{model_type}.modeling_{model_type}"
    model_cls_prefix, _ = get_causal_lm_model_cls_prefix(model_type)
    module = __import__(module_path, fromlist=[f"{model_cls_prefix}SparseMoeBlock"])
    model_cls = getattr(module, f"{model_cls_prefix}SparseMoeBlock")
    return model_cls
