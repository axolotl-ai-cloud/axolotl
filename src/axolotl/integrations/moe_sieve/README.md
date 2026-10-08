# MoE-Sieve

Apply LoRA to the most frequently routed 25% of packed experts independently in
each layer, following [MoE-Sieve](https://arxiv.org/abs/2603.24044). All base experts
continue to run under the original routing policy. The selected set stays fixed
throughout training.

Add these settings to a text SFT configuration. This example targets the
Transformers v5 Qwen3 MoE layout:

```yaml
plugins:
  - axolotl.integrations.moe_sieve.MoeSievePlugin
adapter: moe_sieve

moe_sieve:
  selection_file: ./expert-selection.json
  fraction: 0.25
  calibration_samples: 256

lora_r: 32
lora_alpha: 64
lora_dropout: 0
lora_target_linear: true
lora_target_parameters:
  - gate.weight
experts_implementation: eager
```

The plugin adds exact packed-expert parameter targets from the profile.
`lora_target_linear` also adapts attention and shared/dense linear layers;
`gate.weight` adapts Qwen3's parameter-based router. Other architectures may use
different router names. Explicit `lora_target_modules` can replace
`lora_target_linear` to control which non-expert layers receive adapters.

Calibrate once using the same config, then train:

```bash
python -m axolotl.integrations.moe_sieve.profile train.yml
axolotl train train.yml
```

Calibration loads the configured SFT training split and its tokenization, shuffles
with the configured seed (42 if unset), and profiles up to `calibration_samples`
examples, one at a time. It counts actual dispatched expert IDs, excludes padding,
and does not restrict counts to supervised labels. Selection uses
`floor(fraction * num_experts)` with expert-ID tie-breaking. Empty selections and
layers without observed routing fail explicitly. The JSON records counts, tensor
shapes, model/revision, seed, and a hash of the calibration inputs. Calibration is
currently single-process, text-only, and requires a map-style SFT dataset.

The adapter config embeds the selection. `lora_model_dir`, explicit checkpoint
resume, and automatic checkpoint resume use that saved selection without needing
the original profile file. Keep `adapter: moe_sieve` and the plugin enabled for
inference and merging. `axolotl merge-lora` defaults to the legacy PEFT merge path
for this adapter; the memory-efficient merge path is not supported.

## Current support

- Packed 3D expert parameters on modules exposing `num_experts`, with expert IDs
  passed into their forward as one 2D int64 tensor. Unsupported routing signatures
  fail during calibration. Tiny Qwen3 MoE is covered by CPU integration tests.
- Unquantized float32, float16, or bfloat16 weights; eager or batched-mm expert
  execution; one adapter per model. Both packed weight orientations are handled
  using PEFT's `is_transposed` convention.
- Quantization, FSDP, DeepSpeed, expert/tensor/context parallelism, optimized
  ScatterMoE/SonicMoE adapters, ReLoRA, and compilation are not supported yet.
- Expert adapter parameters and optimizer state scale with the selected expert
  count. Forward still materializes the effective full packed weight; this does
  not promise proportional total-memory or runtime savings.
- Expert dropout is zero, a limitation of this parameter-based implementation.
  This differs from the paper's module-based dropout setting.

## Extension boundary

`selection.py` contains profiling and selection without PEFT dependencies.
`peft.py` owns compact factor allocation, packed-weight updates, serialization,
and config-local custom dispatch. It subclasses PEFT's parameter wrapper and
reuses its merge/disable lifecycle. It does not modify PEFT global registries or
installed files. `register_selected_experts` is the boundary to replace if PEFT
adds a public custom parameter-adapter registration API.

Plain `PeftModel.from_pretrained` does not register this custom implementation.
Outside Axolotl, explicitly load `MoeSieveLoraConfig`, call
`register_selected_experts(base_model, config)`, and pass that config to
`PeftModel.from_pretrained` before using the adapter.
