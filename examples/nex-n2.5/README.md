# Finetune Nex-N2.5 with Axolotl

[Nex-N2.5-mini](https://huggingface.co/nex-agi/Nex-N2.5-mini) is a `Qwen3_5MoeForConditionalGeneration` checkpoint (`model_type: qwen3_5_moe`): the Qwen3.5-35B-A3B hybrid Gated DeltaNet + attention MoE with 256 experts, 8 active, and early-fusion vision. Its `config.json` and weight layout match `Qwen/Qwen3.5-35B-A3B` key-for-key, so everything in [examples/qwen3.5](../qwen3.5/) applies here; what differs is the chat template.

## Getting started

Follow the [Qwen3.5 setup](../qwen3.5/README.md#getting-started) (Cut Cross Entropy, and FLA for sample packing), then:

```bash
axolotl train examples/nex-n2.5/mini-qlora.yaml        # text-only QLoRA
axolotl train examples/nex-n2.5/mini-vision-lora.yaml  # vision + text LoRA
```

Both configs are the Qwen3.5-35B-A3B examples with the base model swapped.

## Chat template

Use `chat_template: tokenizer_default`, not `qwen3_5`. The two render single-turn samples identically, but Nex's template differs on multi-turn data:

- every assistant turn gets a `<think>...</think>` block (empty when there is no `reasoning_content`), whereas `qwen3_5` only renders it on the final turn and drops earlier `reasoning_content`;
- system messages are allowed mid-conversation.

Axolotl masks the empty `<think>\n\n</think>\n\n` scaffolding and trains the content plus `<|im_end|>` in both cases. Vision runs dispatch to the Qwen3.5 multimodal collator by `model_type`, so `tokenizer_default` works for `mini-vision-lora.yaml` too.

## DeltaNet weight scale drift

The DeltaNet layers are unchanged from Qwen3.5-35B-A3B, which has known weight scale drift in the late linear-attention layers (36-38) under AdamW with rare expert activation. If a long run destabilises, enable the same guard:

```yaml
normalize_weight_scales:
  - name_pattern: 'linear_attn\.conv1d\.weight'
    threshold: 1.3
```

## Edge0

The [Edge0](https://huggingface.co/Edge0) checkpoints are MLX int4 exports (`Edge0-35B-A3B-preview` of `Qwen/Qwen3.6-35B-A3B`), not transformers-loadable weights, and Axolotl has no support for them. Fine-tune the base model instead; see [Qwen3.6 and Edge0](../qwen3.5/README.md#qwen36-and-edge0).

## Related Resources

- [Qwen3.5 examples](../qwen3.5/README.md) — LoRA targets for DeltaNet, routed and shared experts
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
