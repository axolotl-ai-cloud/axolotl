import importlib
import os
from importlib.metadata import PackageNotFoundError, version as _pkg_version

import torch
from packaging.version import Version

from axolotl.integrations.base import BasePlugin
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_SONICMOE_MIN_VERSION = "0.1.2"


def _check_sonicmoe_version():
    """Require sonic-moe >= 0.1.2 for the ``concat_layout=True`` path."""
    try:
        installed = _pkg_version("sonic-moe")
    except PackageNotFoundError as err:
        raise RuntimeError(
            f"sonic-moe is not installed. Install >= {_SONICMOE_MIN_VERSION} from "
            "https://github.com/Dao-AILab/sonic-moe."
        ) from err
    if Version(installed) < Version(_SONICMOE_MIN_VERSION):
        raise RuntimeError(
            f"sonic-moe {installed} is too old; require >= {_SONICMOE_MIN_VERSION}. "
            "Upgrade from https://github.com/Dao-AILab/sonic-moe."
        )


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
        """Register the requested kernel into ``ALL_EXPERTS_FUNCTIONS`` and pin cfg.

        Architecture-agnostic: routing stays in each model's SparseMoEBlock; only
        the experts call is dispatched through the registry.
        """
        if cfg.use_scattermoe:
            from axolotl.integrations.kernels.libs.scattermoe_lora.experts import (
                register_scattermoe_experts,
            )

            register_scattermoe_experts()
            cfg.experts_implementation = "scattermoe"
            LOG.info("Registered 'scattermoe' in transformers ExpertsInterface")
        elif cfg.use_sonicmoe:
            _check_sonicmoe_version()
            _check_sonicmoe_gpu_compat()

            from axolotl.integrations.kernels.libs.sonicmoe.experts import (
                register_sonicmoe_experts,
            )

            register_sonicmoe_experts()
            cfg.experts_implementation = "sonicmoe"
            LOG.info("Registered 'sonicmoe' in transformers ExpertsInterface")

    def add_callbacks_pre_trainer(self, cfg, model):
        callbacks = []
        if cfg.use_scattermoe:
            from axolotl.integrations.kernels.autotune_callback import (
                AutotuneReportCallback,
            )

            callbacks.append(AutotuneReportCallback())
        return callbacks
