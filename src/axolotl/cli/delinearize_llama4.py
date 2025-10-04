"""
CLI tool to delinearize quantized/Linearized Llama-4 models.
"""

import os
from pathlib import Path
from typing import Generator, Union

import fire
import torch
from accelerate import init_empty_weights
from transformers import AutoProcessor


def iter_convert_patched_to_hf(model_state_dict, num_experts) -> Generator:
    keys = list(model_state_dict.keys())
    for key in keys:
        if ".feed_forward.experts." not in key:
            yield key, model_state_dict[key]
        if ".feed_forward.experts.gate_projs" in key:
            # gate gets fused with up so skip the yield on this and we'll fuse it when asking for the up
            continue
        if ".feed_forward.experts.up_projs" in key:
            if ".feed_forward.experts.up_projs.0." in key:
                # handle the re-shape and fusing of gate and up, and conversion from linear to parameter
                prefix = key.split(".up_projs.0.")[0]
                key = f"{prefix}.gate_up_proj"
                # grab all the up_projs and gate_projs across all experts
                gate_stacked = torch.stack(
                    [
                        model_state_dict[
                            f"{prefix}.gate_projs.{expert_idx}.weight"
                        ].transpose(0, 1)
                        for expert_idx in range(num_experts)
                    ]
                )
                up_stacked = torch.stack(
                    [
                        model_state_dict[
                            f"{prefix}.up_projs.{expert_idx}.weight"
                        ].transpose(0, 1)
                        for expert_idx in range(num_experts)
                    ]
                )
                gate_up_proj = torch.cat((gate_stacked, up_stacked), dim=-1)
                del gate_stacked, up_stacked
                yield key, gate_up_proj
            else:
                del model_state_dict[key]
                continue
        if ".feed_forward.experts.down_projs" in key:
            if ".feed_forward.experts.down_projs.0." in key:
                # handle the re-shape and fusing of gate and up, and conversion from linear to parameter
                prefix = key.split(".down_projs.0.")[0]
                key = f"{prefix}.down_proj"
                # grab all the down_projs across all experts
                down_stacked = torch.stack(
                    [
                        model_state_dict[
                            f"{prefix}.down_projs.{expert_idx}.weight"
                        ].transpose(0, 1)
                        for expert_idx in range(num_experts)
                    ]
                )
                yield key, down_stacked
            else:
                del model_state_dict[key]
                continue


def do_cli(model: Union[Path, str], output: Union[Path, str]) -> None:
    """
    Convert a patched HF format Llama4 model (with separated projections)
    back to the original HF format (with fused projections).

    Args:
        model: Path to the patched HF model
        output: Path to save the converted model
    """
    print(f"Loading model from {model}")
    from axolotl.monkeypatch.models.llama4.modeling import (
        patch_llama4_linearized_modeling,
    )

    unpatch_llama4 = patch_llama4_linearized_modeling()
    from transformers import Llama4ForConditionalGeneration

    model_ = Llama4ForConditionalGeneration.from_pretrained(model, dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(model)
    processor.save_pretrained(output)

    device = model_.device.type
    if device == "cuda":
        print(
            f"peak memory allocated: {torch.cuda.max_memory_allocated() / 1024**2} MB"
        )
        print(f"peak memory reserved: {torch.cuda.max_memory_reserved() / 1024**2} MB")
    model_config = model_.config
    config = model_.config.get_text_config()

    # Get key dimensions from the config
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    num_experts = config.num_local_experts

    print(
        f"Model dimensions: hidden_size={hidden_size}, intermediate_size={intermediate_size}, num_experts={num_experts}"
    )

    # Create output directory if it doesn't exist
    os.makedirs(output, exist_ok=True)

    # Get state dict
    state_dict = model_.state_dict()
    del model_

    # Create a new state dict for the converted model
    converted_state_dict = {}

    # First, copy all keys that don't need modification
    for key, value in iter_convert_patched_to_hf(state_dict, num_experts):
        converted_state_dict[key] = value

    del state_dict
    if device == "cuda":
        torch.cuda.empty_cache()
        print("State dict converted.")
        print(
            f"peak memory allocated: {torch.cuda.max_memory_allocated() / 1024**2} MB"
        )
        print(f"peak memory reserved: {torch.cuda.max_memory_reserved() / 1024**2} MB")
    # Ideally re-load the model import to load the converted state dict
    # Save the converted model
    with init_empty_weights():
        unpatch_llama4()
        model_ = Llama4ForConditionalGeneration(model_config)

    if device == "cuda":
        print("State dict loaded into model.")
        print(
            f"peak memory allocated: {torch.cuda.max_memory_allocated() / 1024**2} MB"
        )
        print(f"peak memory reserved: {torch.cuda.max_memory_reserved() / 1024**2} MB")
    model_.load_state_dict(converted_state_dict, strict=False, assign=True)
    print(f"Saving converted model to {output}...")
    model_.save_pretrained(output)

    print(f"Model successfully converted and saved to {output}")


if __name__ == "__main__":
    fire.Fire(do_cli)
