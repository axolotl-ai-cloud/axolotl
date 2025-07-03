import math

import torch
import torch.distributed as dist


def patch_tiled_mlp(model_type):
    from deepspeed.runtime.sequence_parallel.ulysses_sp import (
        SequenceTiledCompute,
        TiledMLP,
        sequence_tiled_compute,
    )

    # def tiled_mlp_forward(self, x):
    #     input_shape = x.shape
    #     seqlen = input_shape[-2]
    #     hidden = input_shape[-1]
    #     num_shards = math.ceil(seqlen / hidden)
    #     num_shards_tensor = torch.tensor(num_shards, device=x.device)
    #     dist.all_reduce(num_shards_tensor, op=dist.ReduceOp.MAX)
    #     num_shards = num_shards_tensor.item()
    #
    #     compute_params = [self.down_proj.weight, self.gate_proj.weight, self.up_proj.weight]
    #
    #     def mlp_forward(self_, hs):
    #         return self_.down_proj(self_.act_fn(self_.gate_proj(hs)) * self_.up_proj(hs))
    #
    #     mlp_forward_compiled = torch.compile(mlp_forward)
    #
    #     down_res = TiledMLP.apply(
    #         mlp_forward_compiled,
    #         self,
    #         x,
    #         num_shards,
    #         compute_params,
    #     )
    #     return down_res


    try:
        # Dynamically import the module and MLP class
        module_path = f"transformers.models.{model_type}.modeling_{model_type}"
        model_cls_prefix = "".join(
            [part.capitalize() for part in model_type.split("_")]
        )
        module = __import__(module_path, fromlist=[f"{model_cls_prefix}MLP"])
        mlp_cls = getattr(module, f"{model_cls_prefix}MLP")

        mlp_forward_orig = mlp_cls.forward

        def mlp_forward_sequence_tiled_compute(self, x):
            kwargs_to_shard = dict(x=x)
            kwargs_to_pass = dict(self=self)
            grad_requiring_tensor_key = "x"
            compute_params = [self.down_proj.weight, self.gate_proj.weight, self.up_proj.weight]
            seqlen = x.shape[1]
            num_shards = 4

            def mlp_forward_orig(self, x):
                return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

            return SequenceTiledCompute.apply(
                mlp_forward_orig,
                seqlen,
                num_shards,
                keys_to_shard,
                keys_to_pass,
                grad_requiring_tensor_key,
                compute_params,
                output_unshard_dimension,
                output_reduction,
                *args_to_shard,
                *args_to_pass,
            )
            return sequence_tiled_compute(
                mlp_forward_orig,
                seqlen,
                num_shards,
                kwargs_to_shard,
                kwargs_to_pass,
                grad_requiring_tensor_key,
                compute_params,
                output_unshard_dimension=1,  # x
                output_reduction=None,
            )

        mlp_cls.forward = mlp_forward_sequence_tiled_compute
    except (ImportError, AttributeError) as e:
        raise RuntimeError(
            f"Could not import MLP class for model_type: {model_type}. "
            f"Error: {str(e)}"
        ) from e
