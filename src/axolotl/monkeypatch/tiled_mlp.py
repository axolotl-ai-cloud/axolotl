"""Monkeypatch for Tiled MLP implementation"""

import math

import torch
import torch.distributed as dist


def patch_tiled_mlp(model_type, use_original_mlp=False, cfg_num_shards=None):
    from deepspeed.runtime.sequence_parallel.ulysses_sp import TiledMLP

    try:
        # Dynamically import the module and MLP class
        module_path = f"transformers.models.{model_type}.modeling_{model_type}"
        model_cls_prefix = "".join(
            [part.capitalize() for part in model_type.split("_")]
        )
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

        def tiled_mlp_forward(self, x):
            input_shape = x.shape
            seqlen = input_shape[-2]
            hidden = input_shape[-1]
            if cfg_num_shards is None:
                num_shards = math.ceil(seqlen / hidden)
                num_shards_tensor = torch.tensor(num_shards, device=x.device)
                dist.all_reduce(num_shards_tensor, op=dist.ReduceOp.MAX)
                num_shards = num_shards_tensor.item()
            else:
                num_shards = cfg_num_shards

            compute_params = [
                self.down_proj.weight,
                self.gate_proj.weight,
                self.up_proj.weight,
            ]

            down_res = TiledMLP.apply(
                mlp_forward,
                self,
                x,
                num_shards,
                compute_params,
            )
            return down_res

        mlp_cls.forward = tiled_mlp_forward
    except (ImportError, AttributeError) as e:
        raise RuntimeError(
            f"Could not import MLP class for model_type: {model_type}. "
            f"Error: {str(e)}"
        ) from e
