"""Fresh-process native NVFP4 merged-adapter inference validation."""

import argparse

import torch


def _logits(model):
    input_ids = torch.tensor([[1, 2, 3, 4]], device="cuda")
    with torch.no_grad():
        return (
            model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
            .logits.detach()
            .float()
            .cpu()
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--merged", required=True)
    parser.add_argument("--expected", required=True)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM
    from transformers.integrations.deepspeed import unset_hf_deepspeed_config

    from axolotl.cli.utils.lora_merge import merge_lora_sharded_efficient

    unset_hf_deepspeed_config()
    merge_lora_sharded_efficient(args.base, args.adapter, args.merged, device="cpu")
    model = (
        AutoModelForCausalLM.from_pretrained(args.merged, torch_dtype=torch.bfloat16)
        .cuda()
        .eval()
    )
    assert type(model.model.layers[0].self_attn.q_proj.weight).__name__ == "NVFP4Tensor"
    expected = torch.load(args.expected, map_location="cpu", weights_only=True)[
        "logits"
    ]
    torch.testing.assert_close(_logits(model), expected, rtol=0, atol=0)


if __name__ == "__main__":
    main()
