# Finetune Muse Glimmer with Axolotl

Muse Glimmer is a 30B open agentic model from Meta Superintelligence Labs, found on [HuggingFace](https://huggingface.co/meta-models/Muse-Glimmer-30B). It pairs a dense text decoder with a frozen ViT-G/14 perception encoder, and is built to run agent workloads locally on a single consumer GPU.

The architecture is unusual in a few ways that matter when you fine-tune it: attention repeats a `[local, local, local, global]` pattern with a 2048 sliding window, RoPE is applied to the local layers only (the global layers are NoPE), attention output is sigmoid-gated, and the logits are scaled and `tanh`-softcapped after the LM head.

## Getting started

1. Install Axolotl from source following the [installation guide](https://docs.axolotl.ai/docs/installation.html#sec-edge-build).

2. Install [Cut Cross Entropy](https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy) to reduce training VRAM usage. Both configs enable it.

3. Run the fine-tuning:

    ```bash
    # QLoRA, language model only (~27 GiB VRAM)
    axolotl train examples/muse-glimmer/qlora.yaml

    # QLoRA, language model + vision tower (~23 GiB VRAM)
    axolotl train examples/muse-glimmer/qlora-vision.yaml
    ```

Let us know how it goes. Happy finetuning! 🚀

### Tips

- Both configs were validated on a single RTX PRO 6000 Blackwell. With Cut Cross Entropy on they peak under 32 GiB reserved, so a 32 GiB card should be enough; without it, budget ~40 GiB.
- On Blackwell (sm_120) there is no `flash-attn` wheel for current torch/CUDA builds, and `attn_implementation: flash_attention_2` raises rather than falling back. Point it at the hub kernel instead, which needs `kernels>=0.16.0`:

    ```yaml
    attn_implementation: kernels-community/flash-attn2
    ```

- `eot_tokens: ["<|eot|>", "<|eom|>"]` is **required**, and the configs set it. The template closes turns with `<|eot|>`, which is not the tokenizer's `eos_token` (`<|end_of_text|>`). Leave it out and the terminator never enters the loss, so the model never learns to stop.
- There is no full-finetune config here yet. 30B in bf16 is ~60 GiB of weights before optimizer state, so it needs multiple GPUs with DeepSpeed ZeRO-3 rather than a single card.
- Meta kept the perception encoder **frozen** during their own training, which is what `qlora.yaml` mirrors. Reach for `qlora-vision.yaml` only if your images differ substantially from natural photographs.
- `lora_target_modules` is a regex over full module paths rather than suffixes on purpose. The text decoder's attention carries its own `self_attn.gate_proj` next to `mlp.gate_proj`, so a bare `gate_proj` suffix would also adapt the attention gate. The vision tower likewise names its output projection `attn.proj`, not `o_proj`.
- Cut Cross Entropy is supported and worth enabling: the vocabulary is 202,048 tokens, so the materialized logits dominate activation memory. The patch folds `output_multiplier` into the hidden states and applies the `tanh` softcap inside the fused kernel, matching the eager path.
- Fused LoRA kernels are **not** available for this architecture: the fused QKV/O rewrite cannot express the sigmoid-gated attention output. Axolotl disables them automatically.
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
