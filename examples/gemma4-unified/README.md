# Finetune Google's Gemma 4 Unified with Axolotl

[Gemma 4 Unified](https://huggingface.co/docs/transformers/model_doc/gemma4_unified) is the **encoder-free** multimodal member of the [Gemma 4](https://huggingface.co/collections/google/gemma-4) family. Unlike standard Gemma 4, it has **no vision tower and no audio tower** — raw image patches and raw 16 kHz waveform frames are projected directly into the language model's embedding space through lightweight `LayerNorm/RMSNorm → Linear` pipelines. The text backbone is otherwise the same Gemma 4 decoder (mixed sliding/global attention, `global_head_dim=512`, optional KV sharing).

> Requires `transformers>=5.10.1`.

## Getting started

1. Install Axolotl following the [installation guide](https://docs.axolotl.ai/docs/installation.html).
2. Run the finetuning example:

```bash
axolotl train examples/gemma4-unified/12b-vision-lora.yaml
```

## Key config

```yaml
processor_type: AutoProcessor
chat_template: gemma4_unified      # inherits Gemma 4 turn format + media masking
freeze_mm_modules: true            # train the text backbone only
skip_prepare_dataset: true
remove_unused_columns: false
sample_packing: false              # not supported for multimodal
```

Audio and image+audio training use the same path — add `{"type": "audio", ...}` content to your chat-template dataset and the `Gemma4UnifiedProcessor` chunks the raw waveform into soft tokens automatically.

## Attention

- **Flash Attention**: FA2 (max head_dim=256) / FA4 (max head_dim=128) cannot serve the global layers' `global_head_dim=512`. Use `attn_implementation: sdpa` (the example default).
- **Hybrid FA2** (`flash_attention: true` + `gemma4_hybrid_attn_impl: true` — FA2 on sliding layers, sdpa on global layers) currently covers only the **text-only** backbone (`Gemma4UnifiedForCausalLM`). The multimodal `Gemma4UnifiedForConditionalGeneration` this example trains builds its masks via `create_masks_for_generate`, which bypasses the hybrid patch — so use `sdpa` for vision/audio runs.

## Acceleration (opt-in, experimental for unified)

- **Liger kernels** (`liger_rms_norm`, `liger_glu_activation`, `liger_layer_norm`): supported. `liger_cross_entropy` is accepted but a no-op here — the model computes loss via `loss_function` (HF `ForCausalLMLoss`), not `nn.CrossEntropyLoss` (same as standard Gemma 4). RoPE and fused-linear-CE are skipped. Note `liger_glu_activation` assumes the default MLP width, so do not combine it with `use_double_wide_mlp` checkpoints.
- **Fused RMSNorm+RoPE attention** (`fused_attn_kernel: true`): applicable and supported — the unified attention shares Gemma 4's q/k/v-norm + RoPE structure, and the kernel reads cross-layer shared KV correctly. Opt-in (not auto-on like standard Gemma 4) pending GPU validation against released unified checkpoints.
- **Cut Cross Entropy**: registered upstream for `gemma4_unified`/`gemma4_unified_text`, but omitted from this example. The multimodal `Gemma4UnifiedForConditionalGeneration` computes its loss through Axolotl's external loss path (`accepts_loss_kwargs=False`), which materializes logits and bypasses CCE — so CCE has no effect on this vision config. It applies only to the text-only `gemma4_unified_text` path.

## Limitations

- **LoRA QKV/O kernels** (`lora_qkv_kernel`, `lora_o_kernel`): not auto-enabled. The cross-layer KV-sharing layers have no independent k/v projections, so the generic QKV-LoRA source rewrite cannot attach; they work only alongside `fused_attn_kernel: true` (which supplies the fused `apply_qkv`/`apply_o`). Enabling them without `fused_attn_kernel` raises a clear config error.
- **lora_target_linear**: incompatible for multimodal models — use `lora_target_modules` with a regex restricted to the text backbone (`model.language_model.layers...`).
- **Sample packing**: not supported (raw pixel/waveform tokens are interleaved in-sequence).

## Related Resources

- [Gemma 4 Unified model docs](https://huggingface.co/docs/transformers/model_doc/gemma4_unified)
- [Axolotl Docs](https://docs.axolotl.ai)
- [Axolotl Discord](https://discord.gg/7m9sfhzaf3)
