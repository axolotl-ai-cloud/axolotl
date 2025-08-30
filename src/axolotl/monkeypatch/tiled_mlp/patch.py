"""Monkeypatch for Tiled MLP implementation"""

import math
import os

import torch
import torch.distributed as dist

from axolotl.utils.callbacks.models import get_causal_lm_model_cls_prefix
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def patch_tiled_mlp(model_type, use_original_mlp=True, cfg_num_shards=None):
    from deepspeed.runtime.sequence_parallel.ulysses_sp import (
        TiledMLP as DeepSpeedTiledMLP,
    )

    from axolotl.monkeypatch.tiled_mlp.base import DeepSpeedTiledMLPMoE, TiledMLP

    try:
        # Dynamically import the module and MLP class
        module_path = f"transformers.models.{model_type}.modeling_{model_type}"
        model_cls_prefix, _ = get_causal_lm_model_cls_prefix(model_type)
        module = __import__(module_path, fromlist=[f"{model_cls_prefix}MLP"])
        mlp_cls = getattr(module, f"{model_cls_prefix}MLP")

        if use_original_mlp:
            mlp_forward = mlp_cls.forward
        else:

            def generic_mlp_forward(self_, hs):
                return self_.down_proj(
                    self_.act_fn(self_.gate_proj(hs)) * self_.up_proj(hs)
                )

            mlp_forward = torch.compile(generic_mlp_forward)

        is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1

        def tiled_mlp_forward(self, x):
            input_shape = x.shape
            seqlen = input_shape[-2]
            hidden = input_shape[-1]
            if cfg_num_shards is None:
                num_shards = math.ceil(seqlen / hidden)
                if is_distributed:
                    num_shards_tensor = torch.tensor(num_shards, device=x.device)
                    dist.all_reduce(num_shards_tensor, op=dist.ReduceOp.MAX)
                    num_shards = num_shards_tensor.item()
            else:
                num_shards = cfg_num_shards

            if not self._compute_params:
                self._compute_params = [p for p in self.parameters() if p.requires_grad]

            compute_params = self._compute_params
            if not self._tiled_mlp_dist_impl:
                if (
                    self._compute_params
                    and any(
                        hasattr(p, "ds_id") or hasattr(p, "param_idx_in_group")
                        for p in self._compute_params
                    )
                ) or os.environ.get("ACCELERATE_USE_DEEPSPEED", "false") == "true":
                    if model_type == "gpt_oss":
                        self._tiled_mlp_dist_impl = DeepSpeedTiledMLPMoE
                    else:
                        self._tiled_mlp_dist_impl = DeepSpeedTiledMLP
                else:
                    self._tiled_mlp_dist_impl = TiledMLP

            down_res = self._tiled_mlp_dist_impl.apply(
                mlp_forward,
                self,
                x,
                num_shards,
                compute_params,
            )
            return down_res

        mlp_cls.forward = tiled_mlp_forward
        mlp_cls._compute_params = []
        mlp_cls._tiled_mlp_dist_impl = None
        LOG.info(
            f"Successfully monkey-patched TiledMLP for model_type: {model_type}",
            main_process_only=True,
        )
    except (ImportError, AttributeError) as e:
        raise RuntimeError(
            f"Could not import MLP class for model_type: {model_type}. Error: {str(e)}"
        ) from e
