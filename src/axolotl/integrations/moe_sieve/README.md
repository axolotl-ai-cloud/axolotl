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
single-process, text-only, and requires a map-style SFT dataset. The profiler
disables FSDP, EP, and CP while loading the calibration model; the training YAML
can retain its distributed settings. The unsharded base must fit for calibration.

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
- FSDP2/HSDP, torch all-to-all EP, and Ringmaster CP can be composed. EP requires
  FSDP2 and `FULL_STATE_DICT` checkpoints, including when EP spans the whole world.
  Use `expert_parallel_backend: torch`.
- ScatterMoE LoRA and SonicMoE LoRA use compact trainable factors. Their existing
  kernels receive temporary zero-filled factor slots for unselected experts;
  gradients map back to the compact parameters. This preserves parameter and
  optimizer savings, but does not eliminate kernel work for frozen expert slots.
- Quantization, FSDP1, DeepSpeed, tensor parallelism, ReLoRA, and compilation are
  not supported. TP remains deferred.
- Expert adapter parameters and optimizer state scale with the selected expert
  count. Forward still materializes the effective full packed weight; this does
  not promise proportional total-memory or runtime savings.
- Expert dropout is zero, a limitation of this parameter-based implementation.
  This differs from the paper's module-based dropout setting.

## Distributed configuration

For eight GPUs with HSDP and EP, add:

```yaml
fsdp_version: 2
fsdp_config:
  cpu_ram_efficient_loading: false
  offload_params: false
  auto_wrap_policy: TRANSFORMER_BASED_WRAP
  transformer_layer_cls_to_wrap: Qwen3MoeDecoderLayer
  state_dict_type: FULL_STATE_DICT
  reshard_after_forward: true
expert_parallel_size: 2
expert_parallel_backend: torch
dp_shard_size: 2
dp_replicate_size: 2
```

For four-GPU FSDP2+EP, omit `dp_replicate_size`. Pure two-GPU EP uses
`expert_parallel_size: 2` with the same FSDP2 block and no extra DP axes.
CP composes on another axis; for Ulysses with SDPA use:

```yaml
attn_implementation: sdpa
context_parallel:
  size: 2
  backend: ulysses
  load_balance: none
```

The product of EP, CP, DP-shard, and DP-replicate sizes must match the process
count. Ulysses must divide the model's KV-head count. Existing Ringmaster kernel
and architecture restrictions still apply.

The global selected set is never rebalanced or changed to fill EP ranks. A rank
with no selected experts has zero expert-adapter elements but still computes its
frozen base experts and participates in token dispatch/combine. Checkpoints gather
factors in the original global selected-expert order, including uneven and empty
owners. An imbalanced selected set can still create unequal adapter compute and
optimizer memory between EP ranks.

For optimized expert kernels, add `axolotl.integrations.kernels.KernelsPlugin`
to `plugins` alongside `MoeSievePlugin`, then set either `use_scattermoe: true` or
`use_sonicmoe: true`. SonicMoE requires its compatible CUTLASS DSL dependency and
GPU architecture; the kernels plugin validates these before training.

The slow CUDA regression matrix is in
`tests/e2e/multigpu/test_moe_sieve.py`. It forces all selected experts onto one EP
rank, trains, resumes, and compares with uninterrupted training. Direct fused
kernel output/input-gradient/adapter-gradient comparisons live in
`tests/integrations/moe_sieve/test_kernels.py`.

GPU verification uses a synthetic four-layer Qwen3 MoE with eight experts per
layer and two selected experts, on 2–8 H100s. CP verification uses Ulysses with
SDPA; CPU parameter offload and CPU RAM-efficient loading are disabled. These
checks establish correctness for that configuration, not performance or coverage
of every model and CP backend.

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
