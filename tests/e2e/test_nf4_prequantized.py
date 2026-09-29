"""Loading prequantized bitsandbytes checkpoints published on the Hub."""

import pytest
import torch

from tests.hf_offline_utils import disable_hf_offline


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
@disable_hf_offline
def test_cuda_staged_loading_adopts_published_prequantized_checkpoint():
    import bitsandbytes as bnb
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

    from axolotl.loaders.nf4 import load_nf4_model
    from axolotl.utils.dict import DictDefault

    base_model = "axolotl-ai-co/SmolLM2-135M-bnb-nf4-bf16"
    model_config = AutoConfig.from_pretrained(base_model)
    model = load_nf4_model(
        AutoModelForCausalLM,
        model_config,
        {
            "dtype": torch.bfloat16,
            "device_map": {"": "cpu"},
            "quantization_config": BitsAndBytesConfig(
                **model_config.quantization_config
            ),
        },
        DictDefault(
            base_model=base_model,
            nf4_backend="bitsandbytes",
            torch_dtype=torch.bfloat16,
        ),
        "cuda",
    )
    saved = load_file(hf_hub_download(base_model, "model.safetensors"))
    key = "model.layers.0.self_attn.q_proj.weight"
    reference = bnb.nn.Params4bit.from_prequantized(
        data=saved[key],
        quantized_stats={
            name.removeprefix(f"{key}."): value
            for name, value in saved.items()
            if name.startswith(f"{key}.")
        },
        device="cuda",
    )
    torch.testing.assert_close(
        model.model.layers[0].self_attn.q_proj.weight.cuda(),
        bnb.functional.dequantize_4bit(reference, reference.quant_state),
        rtol=0,
        atol=0,
    )
