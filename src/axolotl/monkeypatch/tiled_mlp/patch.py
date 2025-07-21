"""Monkeypatch for Tiled MLP implementation"""

import math
import os
from functools import partial

import torch
import torch.distributed as dist
from deepspeed.runtime.sequence_parallel.ulysses_sp import TiledMLP

from axolotl.monkeypatch.tiled_mlp.fsdp import TiledMLPFSDP
from axolotl.utils.callbacks.models import get_causal_lm_model_cls_prefix
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def patch_tiled_mlp_distributed(
    tiled_mlp_cls, model_type, use_original_mlp=False, cfg_num_shards=None
):

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

            if not self._compute_params:  # pylint: disable=protected-access
                self._compute_params = [  # pylint: disable=protected-access
                    p for p in self.parameters() if p.requires_grad
                ]

            compute_params = self._compute_params  # pylint: disable=protected-access

            down_res = tiled_mlp_cls.apply(
                mlp_forward,
                self,
                x,
                num_shards,
                compute_params,
            )
            return down_res

        mlp_cls.forward = tiled_mlp_forward
        mlp_cls._compute_params = []  # pylint: disable=protected-access
        LOG.info(
            f"Successfully monkey-patched TiledMLP for model_type: {model_type}",
            main_process_only=True,
        )
    except (ImportError, AttributeError) as e:
        raise RuntimeError(
            f"Could not import MLP class for model_type: {model_type}. "
            f"Error: {str(e)}"
        ) from e


patch_tiled_mlp_deepspeed = partial(patch_tiled_mlp_distributed, TiledMLP)
patch_tiled_mlp_fsdp = partial(patch_tiled_mlp_distributed, TiledMLPFSDP)
