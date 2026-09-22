"""The non-staged expert quantizer must stay off CPU-staged NF4 loads."""

import pytest

from axolotl.loaders.patch_manager import PatchManager
from axolotl.utils.dict import DictDefault


@pytest.mark.parametrize("staged", [True, False])
def test_moe_expert_quantization_patch_defers_to_staged_loader(monkeypatch, staged):
    from axolotl.monkeypatch import moe_quant

    on_load = []
    monkeypatch.setattr(
        moe_quant, "patch_moe_quantization_on_load", lambda cfg: on_load.append(cfg)
    )
    monkeypatch.setattr(
        moe_quant, "patch_peft_target_parameters_matching", lambda: None
    )

    manager = PatchManager.__new__(PatchManager)
    manager.cfg = DictDefault(
        quantize_moe_experts=True,
        load_in_4bit=True,
        adapter="qlora",
        fsdp_version=2 if staged else None,
        qlora_sharded_model_loading=staged,
        fsdp_config={"cpu_ram_efficient_loading": True} if staged else None,
    )

    manager._apply_moe_expert_quantization_patch()

    assert bool(on_load) is not staged
