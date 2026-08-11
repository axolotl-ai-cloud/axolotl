# Finetune Muse Glimmer with Axolotl

Muse Glimmer is a 30B open agentic model from Meta Superintelligence Labs, found on [HuggingFace](https://huggingface.co/meta-models/Muse-Glimmer-30B). It pairs a dense text decoder with a frozen ViT-G/14 perception encoder, and is built to run agent workloads locally on a single consumer GPU.

The architecture is unusual in a few ways that matter when you fine-tune it: attention repeats a `[local, local, local, global]` pattern with a 2048 sliding window, RoPE is applied to the local layers only (the global layers are NoPE), attention output is sigmoid-gated, and the logits are scaled and `tanh`-softcapped after the LM head.

## Getting started

1. Install Axolotl from source following the [installation guide](https://docs.axolotl.ai/docs/installation.html#sec-edge-build).

2. The architecture landed in `transformers` v5.15.0, so make sure you are on at least that version:

    ```bash
    uv pip install 'transformers>=5.15.0'
    ```

3. Run the fine-tuning:

    ```bash
    # QLoRA, language model only
    axolotl train examples/muse-glimmer/qlora.yaml

    # QLoRA, language model + vision tower
    axolotl train examples/muse-glimmer/qlora-vision.yaml

    # full finetune
    axolotl train examples/muse-glimmer/fft.yaml
    ```

Let us know how it goes. Happy finetuning! 🚀

### Tips

- Meta kept the perception encoder **frozen** during their own training, which is what `qlora.yaml` mirrors. Reach for `qlora-vision.yaml` only if your images differ substantially from natural photographs.
- `lora_target_modules` is a regex over full module paths rather than suffixes on purpose. The text decoder's attention carries its own `self_attn.gate_proj` next to `mlp.gate_proj`, so a bare `gate_proj` suffix would also adapt the attention gate. The vision tower likewise names its output projection `attn.proj`, not `o_proj`.
- Cut Cross Entropy and fused LoRA kernels are **not** available for this architecture. There is no `MuseGlimmerForCausalLM` for CCE's generic patch to attach to, and the logits are multiplied by `output_multiplier` then softcapped after `lm_head`, which a fused head does not reproduce. Axolotl raises if you enable either.
- Liger wires up the SwiGLU MLP and the vision tower's LayerNorms. RMSNorm, RoPE and fused linear cross entropy are skipped: the text stack mixes two RMSNorm variants on two different epsilons, and the NoPE global layers receive no position embeddings at all.
- The chat template is Harmony-style. Assistant turns render as `<|start|>assistant to=user<|message|>...<|eot|>`, and `add_generation_prompt` stops at `<|start|>assistant` so the model generates the ` to=user` recipient itself. Axolotl trains that recipient prefix as part of the assistant span; if you supply your own template, keep that property or the model will never learn to open its turn.
- Reasoning traces go in a separate `reasoning_content` field on the assistant message and render as a `to=self` block closed by `<|eom|>`. They are trained on by default, as their own assistant span.
- Assistant turns do not all close with `<|eot|>`: an explicit `recipient` other than `user`, `end_turn: false`, or a non-final tool call closes with `<|eom|>` instead. The vision path therefore bounds assistant spans on the next `<|start|>` rather than on a terminator, which means `train_on_eos` cannot gate the assistant terminator: it is always trained. Terminators for `system`, `user` and `tool` turns still honor the setting.
- Reasoning strength is set in the system prompt (`low` / `medium` / `high` / `xhigh`, defaulting to `high`). Keep it consistent between training and inference.
- Tool calls use an ATEM XML block rather than JSON, and the template raises if `tool_call.function.arguments` is a JSON string. Pass a dict.
- Read more on how to load your own dataset at [docs](https://docs.axolotl.ai/docs/dataset_loading.html).

## Dataset Format

The configs use the OpenAI Messages format as described [here](https://docs.axolotl.ai/docs/dataset-formats/conversation.html#chat_template). Image parts use `{"type": "image"}`, not `image_url`:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image"},
        {"type": "text", "text": "What is in this image?"}
      ]
    },
    {
      "role": "assistant",
      "content": "A cat sitting on a windowsill."
    }
  ]
}
```

To train reasoning traces, add `reasoning_content` alongside `content` on the assistant message:

```json
{
  "role": "assistant",
  "reasoning_content": "The animal has pointed ears and whiskers.",
  "content": "A cat sitting on a windowsill."
}
```

## Related Resources

- [Muse Glimmer on HuggingFace](https://huggingface.co/meta-models/Muse-Glimmer-30B)
- [Meta AI Research announcement](https://research.meta.ai/blog/introducing-muse-glimmer-open-agentic-model)
- [Multimodal docs](https://docs.axolotl.ai/docs/multimodal.html)
