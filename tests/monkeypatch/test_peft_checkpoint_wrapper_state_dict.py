"""An activation-checkpointed FSDP2 adapter must survive a sharded save and reload.

Torch's checkpoint wrapper strips ``_checkpoint_wrapped_module.`` from state-dict keys
but not from module names; PEFT 0.21 derives adapter key prefixes from module names.
"""

from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed.fsdp import StateDictType


def test_checkpoint_wrapped_adapter_round_trips_sharded_state(tmp_path, monkeypatch):
    from accelerate import FullyShardedDataParallelPlugin, PartialState
    from accelerate.utils import fsdp_utils
    from peft import LoraConfig, get_peft_model
    from torch.distributed.device_mesh import init_device_mesh
    from transformers import LlamaConfig, LlamaForCausalLM

    from axolotl.monkeypatch.accelerate.fsdp2 import fsdp2_prepare_model
    from axolotl.monkeypatch.peft.state_dict import (
        patch_peft_checkpoint_wrapper_prefixes,
    )
    from axolotl.utils import distributed as axolotl_distributed

    for key, value in {"RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0"}.items():
        monkeypatch.setenv(key, value)
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1
    )
    axolotl_distributed.distributed_state = None
    PartialState._reset_state()
    try:
        PartialState(cpu=True)
        patch_peft_checkpoint_wrapper_prefixes()
        base = LlamaForCausalLM(
            LlamaConfig(
                hidden_size=64,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                vocab_size=64,
            )
        )
        base.requires_grad_(False)
        model = get_peft_model(
            base, LoraConfig(r=4, target_modules=["q_proj", "v_proj"])
        )
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp_shard",))
        plugin = FullyShardedDataParallelPlugin(
            fsdp_version=2,
            auto_wrap_policy="TRANSFORMER_BASED_WRAP",
            transformer_cls_names_to_wrap=["LlamaDecoderLayer"],
            activation_checkpointing=True,
        )
        accelerator = SimpleNamespace(
            device=torch.device("cpu"),
            is_main_process=True,
            process_index=0,
            num_processes=1,
            is_fsdp2=True,
            wait_for_everyone=dist.barrier,
            state=SimpleNamespace(
                fsdp_plugin=plugin,
                device_mesh=mesh,
                parallelism_config=SimpleNamespace(fsdp_dim_names=("dp_shard",)),
            ),
        )
        model = fsdp2_prepare_model(accelerator, model)
        assert any("_checkpoint_wrapped_module" in n for n, _ in model.named_modules())

        save_plugin = SimpleNamespace(
            fsdp_version=2,
            state_dict_type=StateDictType.SHARDED_STATE_DICT,
            state_dict_config=SimpleNamespace(offload_to_cpu=True, rank0_only=False),
            optim_state_dict_config=SimpleNamespace(rank0_only=False),
        )
        directory = str(Path(tmp_path) / "sharded")
        saved = {
            name: p.to_local().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        fsdp_utils.save_fsdp_model(
            save_plugin, accelerator, model, directory, adapter_only=True
        )
        with torch.no_grad():
            for p in model.parameters():
                if p.requires_grad:
                    p.to_local().add_(1.0)
        fsdp_utils.load_fsdp_model(
            save_plugin, accelerator, model, directory, adapter_only=True
        )
        for name, p in model.named_parameters():
            if p.requires_grad:
                torch.testing.assert_close(p.to_local(), saved[name], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()
        PartialState._reset_state()
        axolotl_distributed.distributed_state = None
