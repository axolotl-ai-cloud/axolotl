import importlib
import os
from pathlib import Path

import torch

from axolotl.integrations.base import BasePlugin
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def _check_sonicmoe_gpu_compat():
    """Validate GPU compute capability for SonicMoE and configure env.

    Supported: Hopper (sm_90), Blackwell (sm_100 - sm_103).
    B300 (sm_103) additionally requires Triton 3.6.0.
    """
    if not torch.cuda.is_available():
        return

    cc = torch.cuda.get_device_capability()

    if cc < (9, 0):
        raise RuntimeError(
            f"SonicMoE requires Hopper (sm_90) or Blackwell (sm_100+) GPU, "
            f"but detected sm_{cc[0]}{cc[1]}."
        )

    if cc > (10, 3):
        raise RuntimeError(
            f"SonicMoE does not yet support sm_{cc[0]}{cc[1]}. "
            f"Supported: Hopper (sm_90) and Blackwell (sm_100 - sm_103)."
        )

    # Blackwell (sm_100+): enable QuACK GEMM kernels
    if cc >= (10, 0):
        os.environ.setdefault("USE_QUACK_GEMM", "1")
        LOG.info(
            f"Blackwell GPU (sm_{cc[0]}{cc[1]}) detected, enabling USE_QUACK_GEMM=1"
        )

    # B300 (sm_103): requires Triton 3.6.0
    if cc == (10, 3):
        triton_spec = importlib.util.find_spec("triton")
        if triton_spec is None:
            raise RuntimeError(
                "B300 (sm_103) requires Triton 3.6.0, but Triton is not installed."
            )
        import triton

        triton_version = tuple(int(x) for x in triton.__version__.split(".")[:2])
        if triton_version != (3, 6):
            raise RuntimeError(
                f"B300 (sm_103) requires Triton 3.6.x, but found {triton.__version__}."
            )


class KernelsPlugin(BasePlugin):
    def get_input_args(self):
        return "axolotl.integrations.kernels.KernelsArgs"

    def pre_model_load(self, cfg):
        from axolotl.integrations.kernels.constants import (
            SPARSE_MOE_BLOCK,
            is_experts_only_model,
        )

        # Prefer text backbone type for VLMs, but fall back to base type
        # when the text type isn't in the supported mapping (e.g. qwen3_5_moe_text)
        moe_model_type = cfg.model_config_type_text or cfg.model_config_type
        if (
            moe_model_type not in SPARSE_MOE_BLOCK
            and not is_experts_only_model(moe_model_type)
            and cfg.model_config_type in SPARSE_MOE_BLOCK
        ):
            moe_model_type = cfg.model_config_type

        if cfg.use_scattermoe:
            self._register_kernels()
            if is_experts_only_model(moe_model_type):
                # Models like Gemma4 where MoE is embedded in the decoder layer
                # — register ScatterMoE in the ExpertsInterface so that
                # @use_experts_implementation dispatches to it.
                self._register_experts_interface()
                cfg.experts_implementation = "scattermoe"
            else:
                self._kernelize_model(moe_model_type)
        elif cfg.use_sonicmoe:
            if not importlib.util.find_spec("sonicmoe"):
                raise RuntimeError(
                    "SonicMoE is not installed. See installation instructions at "
                    "https://github.com/axolotl-ai-cloud/axolotl/blob/main/src/axolotl/integrations/kernels/README.md#sonicmoe-installation"
                )

            _check_sonicmoe_gpu_compat()

            if is_experts_only_model(moe_model_type):
                from axolotl.integrations.kernels.libs.sonicmoe.gemma4_experts import (
                    patch_gemma4_sonicmoe,
                )

                LOG.info(
                    f"Applying SonicMoE experts-level patch for model type: {moe_model_type}"
                )
                patch_gemma4_sonicmoe()
            else:
                from axolotl.integrations.kernels.libs.sonicmoe import patch_sonicmoe

                LOG.info(f"Applying SonicMoE patches for model type: {moe_model_type}")
                patch_sonicmoe(
                    moe_model_type,
                    torch_compile=bool(getattr(cfg, "torch_compile", False)),
                    base_model_type=cfg.model_config_type,
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

    def add_callbacks_pre_trainer(self, cfg, model):
        callbacks = []
        if cfg.use_scattermoe:
            from axolotl.integrations.kernels.autotune_callback import (
                AutotuneReportCallback,
            )

            callbacks.append(AutotuneReportCallback())
        return callbacks

    def _kernelize_model(self, model_type: str):
        from kernels import replace_kernel_forward_from_hub

        from axolotl.integrations.kernels.constants import resolve_moe_block_classes

        for model_moe_cls in resolve_moe_block_classes(model_type):
            replace_kernel_forward_from_hub(
                model_moe_cls, "HFScatterMoEParallelExperts"
            )

    def _register_experts_interface(self):
        """Register ScatterMoE in the transformers ExpertsInterface.

        This allows @use_experts_implementation-decorated Experts classes
        to dispatch to ScatterMoE when config._experts_implementation == "scattermoe".
        """
        from axolotl.integrations.kernels.libs.scattermoe_lora.gemma4_experts import (
            register_scattermoe_experts,
        )

        register_scattermoe_experts()
        LOG.info("Registered 'scattermoe' in transformers ExpertsInterface")
