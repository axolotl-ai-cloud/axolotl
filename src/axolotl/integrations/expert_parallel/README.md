# Expert Parallel (DeepEP) Integration

Replaces the MoE dispatch/combine path with DeepEP's fused kernels. See repo-root `DEEPEP_SETUP.md` for install and `BENCHMARK.md` for measured speedups.

## Enable in YAML

EP is enabled by setting `expert_parallel_size > 1` — same shape as `tensor_parallel_size` and `dp_shard_size`. No separate enable flag.

```yaml
plugins:
  - axolotl.integrations.expert_parallel.ExpertParallelPlugin

expert_parallel_size: 2          # 1 = disabled (default); > 1 = enabled
```

For composition with FSDP at 4+ GPUs, set both `expert_parallel_size` and `dp_shard_size`. The product must equal `world_size`:

```yaml
# 4-GPU example: ep × dp_shard = 2 × 2 = 4
expert_parallel_size: 2
dp_shard_size: 2
fsdp_config:
  fsdp_version: 2
  auto_wrap_policy: TRANSFORMER_BASED_WRAP
  transformer_layer_cls_to_wrap: Qwen3MoeDecoderLayer
  state_dict_type: FULL_STATE_DICT
  sharding_strategy: FULL_SHARD
```

The plugin builds a 2D `DeviceMesh` with axes `(ep, dp_shard)`. EP groups are *strided* (orthogonal to FSDP's contiguous dp_shard groups), so the two parallelisms run on disjoint process groups and don't fight.

The plugin auto-composes with whatever local-experts kernel you've already configured. You set the kernel via the standard axolotl knobs and the plugin transparently upgrades it:

| Your existing config                              | Local kernel under DeepEP   |
|---------------------------------------------------|-----------------------------|
| `use_scattermoe: true`                            | ScatterMoE (Triton)         |
| `use_sonicmoe: true`                              | grouped_mm + warning¹        |
| `experts_implementation: grouped_mm` / `batched_mm` | grouped_mm (transformers) |
| `experts_implementation: eager`                   | eager Python loop           |
| (unset)                                           | grouped_mm (default)        |

The master flags `use_scattermoe` / `use_sonicmoe` are the source of truth for the custom MoE kernels — `experts_implementation: scattermoe` is set BY the kernels validator AS A CONSEQUENCE of `use_scattermoe: true`, so we only check the master flag (avoids misfires if a user sets the string without the flag).

You don't need to know the composite registered names (`deep_ep`, `deep_ep_grouped_mm`, `deep_ep_scattermoe`) — the plugin maps your existing `use_scattermoe` / `experts_implementation` selection onto the right one.

¹ SonicMoE composition is future work — see "SonicMoE composition" below.

## How it works

`pre_model_load` registers the three composite names in `transformers.integrations.moe.ALL_EXPERTS_FUNCTIONS`, patches `PreTrainedModel.get_correct_experts_implementation` to whitelist them, and rewrites `cfg.experts_implementation` to the composite name based on your other config.

`post_model_build` walks the model and slices each Experts module's `gate_up_proj` / `down_proj` along the experts dim — each rank owns `E / ep_size` experts. The registered forward function then routes tokens via DeepEP, runs the local kernel on the received tokens, and combines back.

`Buffer.dispatch` and `Buffer.combine` are wrapped in `torch.autograd.Function` (`_DeepEPDispatch` / `_DeepEPCombine`) so backward propagates correctly: backward-of-dispatch is combine, backward-of-combine is dispatch (handle is reused).

## When EP pays off

EP saves memory by sharding experts across ranks. The throughput story is more nuanced and depends on whether the saved comm (no expert grad sync under full FT) outweighs the added comm (dispatch + combine per layer).

Measured at 2-rank A100 NVLink (the floor case — see `BENCHMARK.md §5`):

| Mode | Throughput vs DDP | Memory vs DDP |
|---|---|---|
| DDP + EP, LoRA frozen experts (Qwen3-30B-A3B 48 layers) | −9% | ~−47% |
| DDP + EP, full FT (tiny 4-layer × 32-expert) | −2% (within noise) | ~−16% |
| FSDP2 only, full FT (same tiny model) | −14% | ~−27% |

**Memory wins are reliable.** **Throughput at 2-rank intranode is essentially flat to slightly negative** because DeepEP's per-layer dispatch overhead sits in the same ballpark as the saved `all_reduce` work. EP is roughly throughput-neutral; FSDP alone gives more memory savings but pays a real all_gather cost.

The interesting composition (FSDP + EP on orthogonal mesh axes) needs at least 4 GPUs (`world = dp_shard × ep` with both ≥ 2). At 2 GPUs you can pick one or the other. Above 4 ranks both should compose — FSDP shards non-expert params, EP shards experts, communications happen on disjoint process groups.

If you need EP for memory reasons (model doesn't fit otherwise), this is the right tool today. If you need EP for throughput reasons, validate at your target scale before assuming the win.

## Constraints (v1)

- Models must use `@use_experts_implementation` (canonical 3D `gate_up_proj` / `down_proj`). Mixtral's `ModuleList[BlockSparseTop2MLP]` is out of scope.
- `num_experts` must be divisible by `expert_parallel_size`. Validated at `post_model_build`.
- World size must equal `expert_parallel_size × dp_shard_size × tensor_parallel_size × context_parallel_size`. Validated at `pre_model_load` and at process-group construction with a clear error message.
- 3+ axis composition (EP × DP × TP/CP) is not yet supported in v1; raises `NotImplementedError` with a clear message.
- Low-latency (LL) kernels are inter-node only by design (pure RDMA via IBGDA). Single-node + intranode setups always use the standard kernels and don't benefit from LL.
- FP8 dispatch needs Hopper + DISABLE_SM90_FEATURES=0.

## Hardware

Ampere (sm_80, A100) or Hopper (sm_90, H100), all-pairs NVLink. See `DEEPEP_SETUP.md` for build steps and driver/CUDA forward-compat notes.

## SonicMoE composition (future work)

`use_sonicmoe: true` + `expert_parallel_enabled: true` currently falls back to grouped_mm with a warning. SonicMoE today is a Gemma4-only direct rebind on `Gemma4TextExperts.forward` (see `axolotl/integrations/kernels/libs/sonicmoe/gemma4_experts.py`).

The path forward is the `EXPERTS_ONLY_BLOCK` constant in `axolotl/integrations/kernels/constants.py:60-67` — a `model_type → Experts class name` table. Once SonicMoE registers via `ALL_EXPERTS_FUNCTIONS.register("sonicmoe", sonicmoe_experts_forward)` keyed off this constant (matching the ScatterMoE registration pattern at `kernels/libs/scattermoe_lora/gemma4_experts.py:209`), composition with EP is a one-line change here: add a `_sonicmoe_local` helper that lazy-imports and calls the registered fn, plus a `"sonicmoe"` branch in `_infer_local_kernel`. No EP-side architectural change needed.
