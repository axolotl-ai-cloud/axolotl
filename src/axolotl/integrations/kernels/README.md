# Kernels Integration

MoE (Mixture of Experts) kernels speed up training for MoE layers and reduce VRAM costs. Transformers v5 introduced a uniform dispatch point for the per-expert grouped GEMMs via the `experts_implementation` config kwarg:

```python
class ExpertsInterface(GeneralInterface):
    _global_mapping = {
        "batched_mm": batched_mm_experts_forward,
        "grouped_mm": grouped_mm_experts_forward,
        "sonicmoe": sonicmoe_experts_forward,  # upstream HF integration
    }
```

Axolotl registers two additional implementations into this same global registry: **ScatterMoE** (Triton, runs on any CUDA GPU) and a LoRA-aware **SonicMoE** variant (CUTLASS / cute-DSL, Hopper or newer). Routing — softmax/sigmoid top-k, group selection, shared experts, bias correction, etc. — stays in each model's `SparseMoEBlock`, where transformers handles all per-architecture variation. Axolotl only swaps the experts forward.

## Usage

Add the following to your axolotl YAML config:

```yaml
plugins:
  - axolotl.integrations.kernels.KernelsPlugin

use_kernels: true

# Choose one (mutually exclusive):
use_scattermoe: true
# OR
use_sonicmoe: true
```

`experts_implementation` is auto-set to `scattermoe` / `sonicmoe` from the kernel flag, but you can override to `eager` / `batched_mm` / `grouped_mm` to compare against the transformers reference implementations.

### SonicMoE installation

**Prerequisites:**
- NVIDIA Hopper (H100/H200) or Blackwell (B200/GB200/B300) GPU
- CUDA 12.9+ (13.0+ for B300)
- PyTorch 2.7+
- For B300: Triton 3.6.x

The sonic-moe kernel ships through the HF [`kernels`](https://github.com/huggingface/kernels) package. Transformers v5.8+ auto-fetches a prebuilt kernel from [`kernels-community/sonic-moe`](https://huggingface.co/kernels-community/sonic-moe) on first use:

```bash
uv pip install kernels "nvidia-cutlass-dsl==4.6.0" "apache-tvm-ffi>=0.1.10,<0.2"
```

`apache-tvm-ffi` is an undeclared runtime dependency of `nvidia-cutlass-dsl` 4.6.0 (absent from its `Requires-Dist`, so pip will not pull it); `<0.1.10` breaks `cute.compile`, so pin it explicitly.

**Note:** Blackwell support is in upstream beta. On Blackwell GPUs Axolotl automatically sets `USE_QUACK_GEMM=1` to enable the Blackwell kernels.

**cutlass-dsl 4.8 with the hub kernel.** The `kernels-community/sonic-moe` `ep-support` build
(90d2076) vendors quack 0.6.1, which calls `nvvm.mbarrier_arrive(None, addr, ...)` with the
result type as the first positional argument. cutlass-dsl 4.8 made the result type the
keyword-only `results=`, so that call raises `TypeError` at the first GEMM compile. When SonicMoE
is enabled, Axolotl wraps `nvvm.mbarrier_arrive` to accept the old form. It does this only when
the installed binding has the 4.8 signature, and all other calls pass through unchanged. It is a
compatibility shim to drop once the hub build vendors quack 0.6.3 or later. A check of the
vendored quack's other NVVM/LLVM call shapes against 4.6.0, 4.6.2 and 4.8.0 found no other
mismatch.

The cutlass-dsl version check reads the installed `quack-kernels` requirement. When a hub kernel
has already loaded its own quack, the check logs both versions and skips that requirement,
because the installed `quack-kernels` is not the quack that runs.


## How It Works

The `KernelsPlugin` runs once before model loading and:

1. Calls `register_scattermoe_experts()` or `register_sonicmoe_experts()`, which inserts the kernel forward into `transformers.integrations.moe.ALL_EXPERTS_FUNCTIONS`.
2. Sets `cfg.experts_implementation` to the matching name.
3. When the model loads, transformers' `@use_experts_implementation` decorator on each model's `Experts` class reads `config._experts_implementation` and dispatches to our registered forward.

That's the entire integration — there is no per-architecture SparseMoEBlock monkey-patch, no per-model routing code, and no weight-layout conversion. As new MoE models adopt the decorator upstream they immediately benefit from both kernels.

## BF16 LoRA Support

Both kernels train PEFT adapters on `gate_up_proj` / `down_proj` (and `gate` for the router) end-to-end:

- **ScatterMoE** fuses the LoRA `B @ A` product into the per-expert grouped GEMM via custom Triton kernels (`parallel_linear_lora`). No extra materialization pass.
- **SonicMoE** materializes `W_eff = W + scaling * (B @ A)` per expert inside a custom `MoELoRAMaterialize` `autograd.Function` and passes the effective weight into the CUTLASS kernel. Backward decomposes `dW_eff` into `dA` and `dB` via the chain rule, so LoRA parameters train without modifying the kernel.

Both paths detect PEFT `ParamWrapper` on individual expert parameters (`target_parameters` API) and unwrap them before dispatch.

### ScatterMoE NVFP4 (W4A16) LoRA

Train LoRA on ModelOpt NVFP4 checkpoint via ScatterMoE. Routed experts are dequantized to bf16 (W4A16).

Requires:
- CUDA GPU with Triton.
- `qwen3_moe`, `qwen3_next`, `deepseek_v4`, `glm_moe_dsa`, and `gemma4_text`

**Tip:** in our tests, the Triton `dequant` path below is currently faster end to end than the ScatterMoE NVFP4 (if the arch is supported).

### SonicMoE NVFP4 (W4A4) LoRA

Train LoRA on ModelOpt NVFP4 checkpoint via Quack (e.g. `nvidia/Qwen3-30B-A3B-NVFP4`).

Requires:
- Blackwell SM100 for W4A4, others for W4A16
- `qwen3_moe` / `qwen3_next`

Install the pinned quack kernels (other versions untested):

```bash
uv pip install "quack-kernels==0.6.1" "nvidia-cutlass-dsl==4.6.0" "apache-tvm-ffi>=0.1.10,<0.2"
```

`AXOLOTL_SONICMOE_NVFP4_BACKEND` picks the expert GEMM (unset = auto):

| Backend | Compute | Runs on |
|---------|---------|---------|
| `fp4_cute` | native W4A4 tensor cores (quack) | Blackwell B200/GB200 (SM100/110) |
| `dequant` | dequant to bf16 (W4A16) | any CUDA GPU with Triton |

Advanced tuning knobs (fused up-proj, per-tensor-scale fold, fp8 DeepGEMM backward) are exposed as other `AXOLOTL_SONICMOE_NVFP4_*` env vars; defaults are correct for normal training.

**Tip:** in our tests, the Triton `dequant` path is currently faster (and lower mem) end to end than the `fp4_cute` path for < 100B param MoE model. However, `fp4_cute` should overtake when expert matmul become more dominant cost.

### SonicMoE under expert parallelism: token padding

With expert parallelism, DeepEP sends each token once to every rank that holds at least one of
its top-k experts, so the number of tokens a rank receives changes every step and layer. quack
autotunes per exact shape, and a first sweep costs minutes, so the bf16 SonicMoE EP forward pads
the received tokens with zero-compute sentinel tokens up to a bucket and slices the output back.
By default (`sonicmoe_ep_row_padding: pow2`) the bucket is the next power of two, never below
1024.

sonic-moe then sizes its expert buffers as tokens x top_k, with a sentinel in every slot whose
expert lives on another rank. On Beam (EP=8, 16384 tokens per rank, top_k 8, 32 of 256 experts
per rank) a rank receives at most 8 x 16384 = 131072 tokens and about 1 - (7/8)^8 = 66% of that
(~86k) on average. pow2 pads those ~86k tokens to 131072, about 52% waste, and the expert buffers
become 131072 x 8 = 1048576 rows for about 131k real local (token, expert) pairs. The 10 GiB
allocation that ran out of memory was one of them: `[1048576, 5120]` bf16, that is 131072 padded
tokens x 8 slots x 5120.

High-water padding keeps the shapes stable without rounding up to the next power of two:

```yaml
sonicmoe_ep_row_padding: highwater
sonicmoe_ep_row_padding_granule: 4096     # tokens; default
sonicmoe_ep_row_padding_headroom: 1.03    # default
```

A call that fits the current mark pads to it. A larger call pads to its own count times the
headroom, rounded up to the granule, but never past the next power of two, so a receive at the
131072-token cap stays at 131072. The mark is shared by all MoE layers and kept separately for
training and eval, so short eval batches do not pad to the training size.

The mark only rises between optimizer steps (and after evaluation), to the largest size the
previous step used. Within a step, each call's size depends only on its own count. That keeps an
activation-checkpointing recompute the same shape as its forward even when a later layer received
more tokens in between; a mark that moved mid-step made torch raise `CheckpointError` on the
recompute. The kernels plugin registers the callback that promotes the marks whenever highwater
padding or compaction is on. It rises a few times early in training and then
stays put, so training sees a handful of sizes that are not powers of two (multiples of 4096
tokens, so multiples of 32768 expert rows with top_k 8). Marks are per rank, so ranks can settle
on different sizes. The NVFP4 EP paths drop the sentinels instead of padding and do not use this
setting.

The larger cost is the tokens x top_k expansion itself: about 7/8 of the expert-buffer rows on
Beam are sentinels. sonic-moe's `moe_general_routing_inputs` takes flat per-pair routing
(`router_scores`, `token_indices` sorted ascending, `expert_indices`), and
`sonicmoe_ep_compact_routing` uses it to pass only the local (token, expert) pairs:

```yaml
sonicmoe_ep_compact_routing: true
sonicmoe_ep_pair_padding_granule: 32768   # pairs; default
sonicmoe_ep_row_padding: highwater        # recommended with it, see below
```

On Beam the expert buffers drop from 131072 padded tokens x 8 = 1,048,576 slot rows to about
131k pairs (16384 tokens x 8 experts spread over 8 ranks), padded up to the pair mark.

- **Pair padding.** The pair count is padded high-water style like the tokens (count x
  `sonicmoe_ep_row_padding_headroom` rounded up to the pair granule, or the previous steps' mark;
  separate marks for training and eval, promoted only between steps, capped at tokens x top_k). Pad pairs are unused slots of the grid in token order with
  expert id `num_experts` and score 0, so `token_indices` stays sorted and no token gets more than
  top_k pairs.
- **Token padding still applies.** The received tokens are padded as before because the token
  count is in the kernel shapes too: it is the gather source of the gate/up GEMM and sizes the
  per-token offsets and the output. That costs tokens x hidden, not tokens x top_k x hidden. Every
  token keeps at least one pair, a sentinel one if it has no local expert, just as it has a full
  sentinel row on the grid, so pow2 token padding to 131072 also adds up to ~45k sentinel pairs.
  Use `highwater` token padding with compaction.
- **Sentinel masking.** On the grid most slots are sentinels, so the EP path depends on
  sonic-moe dropping `expert >= E` lanes (the kernels-community EP sentinel fix, 9aaebcd;
  Dao-AILab/sonic-moe#51). With compaction only the pad pairs and the one pair of a token with no
  local expert are sentinels.
- **Where it applies.** Dense gated bf16 experts on the EP path. NVFP4 / merge-aware, multi-LoRA,
  non-gated experts and the non-EP path keep their current routes. Building the pairs reads the
  pair count on the host (one sync per MoE call), so under `torch.compile` the grid is used, with
  a warning.

### SonicMoE expert LoRA without merged weights

By default, bf16 SonicMoE experts with LoRA build `W_eff = W + scaling * B @ A` for every local
expert in every forward (and again in an activation-checkpoint recompute), run sonic-moe on it,
and in backward compute the full-rank `dW_eff` with the expert weight-gradient GEMMs before
projecting it onto `lora_A` / `lora_B`. On a Beam EP=8 rank (32 local experts, H 5120, I 2560)
that is about 2.5 GB of merged weights plus a same-size gradient buffer per MoE layer, and two
full-size weight-gradient GEMMs the frozen base never needs.

```yaml
sonicmoe_lora_split: true
```

keeps the frozen base in sonic-moe's fused quack GEMMs and feeds the LoRA in as a low-rank
addend: `scaling * (x A^T) B^T` enters the gate/up GEMM before its SwiGLU/GeGLU epilogue and
`scaling * (a A^T) B^T` the down GEMM. In backward, `gemm_dgated` gives the base part of the
gate/up gradient and a Triton kernel adds the down-LoRA part through the GLU backward; the router
score gradient gets the LoRA term too. LoRA gradients come from rank-r GEMMs only. Per layer and
micro-step this removes the merged weights and the full-rank weight gradients, and cuts the
expert-GEMM FLOPs by about 31% (about 23% with activation checkpointing, which repeats the
forward); `scripts/benchmark_sonicmoe_lora_split.py` measures both paths.

It applies to bf16 gated experts with a frozen base, no expert bias and a SwiGLU/GeGLU
activation, on the EP and non-EP dense paths (with or without `sonicmoe_ep_compact_routing`).
Anything else logs a warning once and materializes as before.

## Model Support

Any model whose `Experts` class is decorated with `@use_experts_implementation` upstream works automatically. As of transformers 5.8 this includes (verified). The `bf16` columns are base (unquantized) support; the `NVFP4` columns mark which kernel trains LoRA on a ModelOpt NVFP4 checkpoint of that arch:

| Model Type        | ScatterMoE (bf16) | SonicMoE (bf16) | NVFP4 (ScatterMoE) | NVFP4 (SonicMoE) |
|-------------------|:---------:|:--------:|:---:|:---:|
| `mixtral`         |    Yes    |   Yes    |  -  |  -  |
| `qwen2_moe`       |    Yes    |   Yes    |  -  |  -  |
| `qwen3_moe`       |    Yes    |   Yes    | Yes | Yes |
| `qwen3_next`      |    Yes    |   Yes    | Yes | Yes |
| `qwen3_5_moe`     |    Yes    |   Yes    |  -  |  -  |
| `olmoe`           |    Yes    |   Yes    |  -  |  -  |
| `mistral4`        |    Yes    |   Yes    |  -  |  -  |
| `glm_moe_dsa`     |    Yes    |   Yes    | Yes |  -  |
| `deepseek_v3`     |    Yes    |   Yes    |  -  |  -  |
| `minimax_m2`      |    Yes    |   Yes    |  -  |  -  |
| `ernie4_5_moe`    |    Yes    |   Yes    |  -  |  -  |
| `hunyuan_v1_moe`  |    Yes    |   Yes    |  -  |  -  |
| `gemma4_text`     |    Yes    |   Yes    | Yes |  -  |
| `gpt_oss`         |    Yes    |   No     |  -  |  -  |
| `nemotron_h`      |    Yes    |   Yes¹   | Yes | Yes¹ |

NVFP4 for `deepseek_v4` is supported via ScatterMoE with `use_dsv4_kernels` (its own fused-kernel path), so it is not a row above.

¹ `nemotron_h` covers Nemotron-3 latentmoe: **non-gated** relu² experts (`up_proj`/`down_proj`, no gate), operating in `moe_latent_size` when set (tokens arrive pre-projected by the block's `fc1/fc2_latent_proj`). ScatterMoE handles that layout natively; SonicMoE routes it through a `torch._grouped_mm` MLP because the current sonic-moe op layer only allows gated epilogues, and `AXOLOTL_SONICMOE_NONGATED_FUSED=1` switches to the exact REGLU rewrite (`relu²(h) == h·relu(h)`) once a build relaxes that assert. Nemotron-3 NVFP4 checkpoints (modelopt `MIXED_PRECISION`) keep the routed experts packed NVFP4 and dequantize every other quantized linear to bf16 at load; quantization there is per layer, not per module name, so one converter per suffix picks its branch from the keys each layer ships.

`gpt_oss` carries the decorator with `is_concatenated=False, is_transposed=True, has_bias=True` and uses a clamped sigmoid-GLU activation. The ScatterMoE forward handles the transposed/interleaved/biased layout and that epilogue via its Triton path (no weight transpose, interleaved gate/up, per-expert bias folded into the grouped GEMM).

### Epilogue check (why `gpt_oss` is No on SonicMoE)

SonicMoE picks its fused epilogue from `config.hidden_act`, which cannot express a clamped or otherwise non-plain GLU. `GptOssConfig.hidden_act` is `"silu"`, so `gpt_oss` silently ran plain SwiGLU (10.4% top-1 agreement, teacher-forced NLL 2.30 -> 8.12 on `openai/gpt-oss-20b`).

Axolotl now probes each model's declared `_apply_gate` numerically at load and raises if the chosen path cannot reproduce it, pointing at `expert_backend: scattermoe`.

## Feature comparison

| Feature                          | ScatterMoE | SonicMoE |
|----------------------------------|:----------:|:--------:|
| Kernel backend                   | Triton     | CUTLASS / cute-DSL |
| GPU requirement                  | Any CUDA   | Hopper+ |
| LoRA path                        | Fused in Triton kernel | `MoELoRAMaterialize` + custom autograd |
| LoRA overhead                    | Lower (fused) | Higher (materialization pass) |
| Selective expert dequantization  | Yes (~97% memory savings) | No |
| Weight format                    | Standard `[E, 2*I, H]` | Standard `[E, 2*I, H]` (concat layout, no interleave) |

## Note on MegaBlocks

We tested [MegaBlocks](https://huggingface.co/kernels-community/megablocks) but were unable to ensure numerical accuracy, so we did not integrate it. It was also incompatible with many newer model architectures in transformers.

## Autotune warm-start cache

Triton keeps each `@triton.autotune` winner only in the process, so every run re-benchmarks every
(shape, dtype) key. Under expert parallelism that benchmark lands at different moments on different
ranks, which DeepEP's combine barrier reads as a straggler. `autotune_cache_path` persists the winners:

```yaml
plugins:
  - axolotl.integrations.kernels.KernelsPlugin
use_scattermoe: true
autotune_cache_path: /shared/cache/autotune/b200.json
```

After the kernels load, the file is read and every record whose kernel is loaded, whose candidate
config matrix still matches, and whose key is not yet measured is inserted into that autotuner's
cache. The file is rewritten at every checkpoint and at train end with the union of all ranks'
winners. Files record the GPU they were measured on and are ignored on a different GPU model, since
the winners are device-specific.

Each record carries a `source`: `measured` (benchmarked in a run, with the full `timings_ms` table
per candidate config, `configs_benchmarked`, `failed_configs` for configs that did not compile or
ran out of resources, and `sweep_seconds`, the sweep's wall time including compiles), `profile`
(decided by `autotune_profiles`; timed only when the profile left several candidates), `seeded`
(taken from the file), or `untimed` (benchmarked before timing capture reached that kernel, or a
disk-cache hit). A re-run that only seeds a key, or pins it over the same candidates, keeps the
measured record on file.

With `use_sonicmoe: true`, the SonicMoE/quack autotune winners (and any Triton kernels bundled
with SonicMoE) are written on the same schedule to a second file next to it, `b200.quack.json`,
with the same record layout plus `call_args` (the tensor shapes and key arguments of the tuned
call, by argument name), the CuTe DSL version and, under `quack_loaded`, the version and location
of every quack that was imported (the hub sonic-moe kernel vendors its own). Measured winners in
that file are seeded back into quack when its keys are plain strings, which is what the vendored
quack 0.6.1 builds; quack 0.6.5 keys keep tuples and dtypes and are not seeded.

`python scripts/autotune_report.py /shared/cache/autotune/b200.json` summarizes both files (shape
keys, winners, timing spread, sweep time per kernel; `--quack-cache-dir $QUACK_CACHE_DIR` adds
quack's own `*.autotune.json` files) and ends with an `autotune_profiles` block for the keys whose
sweep took at least `--min-sweep-seconds` (default 60), each matched to its exact shapes and
pinned to its winner, or to its `--top N` fastest configs.

## Autotune profiles

`autotune_profiles` pins or narrows a kernel's candidates. An entry applies to the loaded
autotuners named by `kernel` (a name or glob, optionally module-qualified) on a GPU whose name
contains `gpu`. Profiles are applied after model load, at train start, and during the first steps,
so kernels that import late (SonicMoE's quack) are covered.

```yaml
autotune_profiles:
  - kernel: <kernel name or glob>
    backend: quack           # optional: triton (default) or quack
    gpu: B200                # optional substring of the GPU name
    match: {...}             # optional; without it every unseen key uses the profile
    configs: [{...}, ...]    # one = no benchmark; several = a short search
    allow: {...}             # quack only: filter the candidates instead of listing them
```

For Triton kernels, `match` keys are the kernel's arguments (a list = any of) and each config is a
`triton.Config` (`BLOCK_*` kwargs, `num_warps`, `num_stages`).

For quack kernels, `match` is checked on each call against the tuned function's arguments:

- `arg`: a scalar argument, e.g. `activation: swiglu` (a list = any of)
- `arg.shape`: a tensor's full shape, `null` for any size
- `arg.dimN`: one dimension of a tensor, negative `N` counting from the end (a list = any of)
- a scalar or `dimN` can also take a range, e.g. `"A.dim0": { ge: 262144 }` (`ge`, `gt`, `le`,
  `lt`; several bounds must all hold)

Each config lists `GemmConfig` fields and selects every candidate that has those values, so a
partial config like `{tile_m: 256, tile_n: 256}` keeps all the variants of that tile. A config
that matches no candidate is built from the autotuner's own `GemmConfig`. Calls that match no
profile keep quack's full search. If a matched profile leaves one candidate, it is not
benchmarked. A narrowed list changes quack's disk-cache hash, so it is swept once.

On Beam (B200, EP=8) the first sweep of the large buckets costs about 400-450 s per rank, compiles
included, while the 16k/32k-row buckets tune in under 30 s. So the profile covers only the
65536/131072-row keys, narrowed to their measured winners and a close runner-up:

```yaml
autotune_profiles:
  - kernel: gemm_gated_tuned          # gate_up: B is [experts, 5120, 5120]
    backend: quack
    gpu: B200
    match: { "B.shape": [32, 5120, 5120], "postact_out.dim0": [65536, 131072] }
    configs:
      - { tile_m: 256, tile_n: 512, cluster_m: 2, cluster_n: 1, is_dynamic_persistent: true, use_tma_gather: true }
      - { tile_m: 128, tile_n: 256, cluster_m: 2, cluster_n: 1, is_dynamic_persistent: true, use_tma_gather: true }
  - kernel: gemm_tuned                # down: B is [experts, 2560, 5120]
    backend: quack
    gpu: B200
    match: { "B.shape": [32, 2560, 5120], "A.dim0": [65536, 131072] }
    configs:
      - { tile_m: 256, tile_n: 256, cluster_m: 2, cluster_n: 1 }
      - { tile_m: 128, tile_n: 256, cluster_m: 1, cluster_n: 1 }
      - { tile_m: 128, tile_n: 224, cluster_m: 2, cluster_n: 1 }
```

### Allow filters (quack)

Instead of listing configs, a quack entry can filter the candidates by field value with `allow`.
The filter runs after quack's own validity pruning (`prune_invalid_gemm_configs`), never instead
of it, so it can only remove configs quack would otherwise compile and benchmark. A key that
benchmarks under a filter is still recorded as `measured`, with `candidate_config_count` set to
the number of configs actually swept.

- `<field>: [values]` for any `GemmConfig` field (`tile_m`, `tile_n`, `swap_ab`,
  `is_dynamic_persistent`, `use_tma_gather`, ...); a single value is allowed as is.
- `cluster: [[cluster_m, cluster_n], ...]`.
- `where: [{if: {...}, then: {...}}, ...]`: a config whose fields match every `if` entry must
  also match every `then` entry. Configs that do not match `if` are not affected.

`match` works as above; without it the filter covers every call of that kernel on that GPU. If
the filter removes every valid config for a call, that call keeps quack's valid list and a
warning is logged once per shape.

`python scripts/autotune_report.py b200.json --quack-cache-dir $QUACK_CACHE_DIR --check-allow
config.yaml` checks each allow filter in the config against the measured keys in `b200.quack.json`
and in quack's own disk cache. It lists every key whose winner, or any config within `--within`
of it (default 5%), the filter would prune, and exits 1 if a winner would be pruned.

On Beam (B200, bf16 SonicMoE) the three expert GEMMs are `gemm_gated_tuned` (gate_up forward),
`gemm_tuned` (down) and `gemm_dgated_tuned` (gate_up backward, the slowest first sweep at about
510 s per rank).

- **`gemm_gated_tuned`:** the winner had `tile_n` equal to `tile_m` or twice it on 36 of the 37
  measured shapes. The exception, 256x224, beat the best surviving config by 1.6%. Both ratios
  have to stay: each lost by up to 22-47% on some shapes the other won.
- **`gemm_dgated_tuned` and `gemm_tuned`:** they do not follow that ratio. None of their measured
  winners use `tile_n` 64 or 128, and a 2x2 cluster only ever won with `tile_m` 256.

The filter below keeps these measured winners:

- `gemm_gated_tuned`: 128x256 c2x1, 256x512 c2x1
- `gemm_tuned`: 128x256 c2x1 and c1x1, 128x224 c2x1, 256x256 c2x1, 128x160 c1x2, 256x192 c2x2,
  128x192 c1x1
- `gemm_dgated_tuned`: 256x256 c2x1

It cuts each first sweep from 76 to 24 candidates for `gemm_gated_tuned`, from 76 to 52 for
`gemm_dgated_tuned` and from 74 to 42 for `gemm_tuned`:

```yaml
autotune_profiles:
  - kernel: gemm_gated_tuned
    backend: quack
    gpu: B200
    allow:
      where:
        - { if: { tile_m: 128 }, then: { tile_n: [128, 256] } }
        - { if: { tile_m: 256 }, then: { tile_n: [256, 512] } }
  - kernel: gemm_dgated_tuned
    backend: quack
    gpu: B200
    allow:
      tile_n: [160, 192, 224, 256, 512]
  - kernel: gemm_tuned
    backend: quack
    gpu: B200
    allow:
      tile_n: [160, 192, 224, 256, 512]
      where:
        - { if: { cluster: [2, 2] }, then: { tile_m: 256 } }
```

The gated kernels need no cluster rule. Both gather their input (`A_idx`), and quack's validity
pruning then keeps only `cluster_n` 1 and drops `swap_ab`, which leaves 76 of the 296-config
`dgated` grid and of the 148-config `gated` grid. The 1x2 cluster stays for `gemm_tuned` because
it picked 128x160 c1x2 on one rank. For `match`, the argument names are `A, B, preact_out,
postact_out, ...` for `gemm_gated_tuned`, `A, B, out, ...` for `gemm_tuned` and `A, B, PreAct,
dx_out, postact_out, colvec_scale, activation, colvec_reduce, cu_seqlens_m, A_idx, ...` for
`gemm_dgated_tuned`.

### Pinning the large training buckets (Beam B200)

The filter above still sweeps every new shape. For the training buckets, whose first sweep is the
expensive one, a range `match` pins the measured winners for every row count from 262144 up,
while smaller shapes (eval, short batches) keep the filtered search. The rows these GEMMs see are
expert slots, tokens x top_k: `postact_out`, `A` of the down-projection forward and `PreAct` hold
1048576 rows when a rank pads to 131072 received tokens, sentinel slots included. With
`sonicmoe_ep_row_padding: highwater` and its 4096-token granule, training rows land on multiples
of 32768 (4096 x 8), roughly 0.7M to 1.0M, which a threshold covers and a list of exact sizes
would miss. List these entries before the `allow` entries: the first matching entry for a call
decides, except that a pin with no config valid for the call passes it to the next match.

```yaml
autotune_profiles:
  - kernel: gemm_gated_tuned          # winner at 512k and 1M rows
    backend: quack
    gpu: B200
    match: { "postact_out.dim0": { ge: 262144 } }
    configs:
      - { tile_m: 256, tile_n: 512, cluster_m: 2, cluster_n: 1, is_dynamic_persistent: true, use_tma_gather: true }
  - kernel: gemm_tuned                # down projection forward: 256x256 c2x1; 128x256 c1x1 also won at 1M rows
    backend: quack
    gpu: B200
    match: { "A.dim0": { ge: 262144 }, "B.shape": [32, 2560, 5120] }
    configs:
      - { tile_m: 256, tile_n: 256, cluster_m: 2, cluster_n: 1 }
      - { tile_m: 128, tile_n: 256, cluster_m: 1, cluster_n: 1 }
  - kernel: gemm_dgated_tuned
    backend: quack
    gpu: B200
    match: { "PreAct.dim0": { ge: 262144 } }
    configs:
      - { tile_m: 256, tile_n: 256, cluster_m: 2, cluster_n: 1, is_dynamic_persistent: true, use_tma_gather: true }
```

The `gemm_gated_tuned` and `gemm_dgated_tuned` configs give every field that varies on SM100
except `swap_ab`, which validity pruning removes for these gathered GEMMs, so each resolves to one
candidate and is not benchmarked at all. The `gemm_tuned` entry leaves `is_dynamic_persistent`
open, so that kernel runs a 4-config search. `gemm_tuned` also runs other GEMMs, and the pin
leaves them alone on purpose, since its winners were measured on the down-projection forward:
- the `B.shape` match keeps it off other weights;
- the weight-gradient calls have `A` as `[5120, tokens]`, so `A.dim0` is 5120 and the threshold
  does not catch them.

Those keep the `allow` filter and their own search until their winners are measured. A dW pin
would match on `"A.dim1": { ge: 32768 }`.

## quack key buckets and the no-sweep mode (SonicMoE)

quack keys each autotune winner on every tensor's exact shape, so every new routed row count of a
grouped GEMM starts a full sweep (40-76 candidates). With `use_sonicmoe`, the variable-length dims
of grouped calls are rounded up to the next power of two in the key quack looks up, stores and
writes to its disk cache, so one sweep serves the bucket:

- with `cu_seqlens_m`: the total-M rows (dim 0 of `A`, `out`, `C`, `preact_out`, `postact_out`,
  `PreAct`, `dx_out`, `colvec_scale`) and the length of `A_idx` (with `A_idx`, `A`'s rows are the
  gather source, which is bucketed too);
- with `cu_seqlens_k`: the total-K dim (`A`'s last dim, `B`'s first) and the length of `A_idx`.

`quack_autotune_key_buckets: shape` replaces those dims with `-1` instead, so each kernel sweeps once
per weight shape and every token count, sequence length or EP size reuses that winner (keys end
with `axolotl_bucket=shape`). The first call of a shape is the one benchmarked.

Weight and feature dims, the expert count and blockscaled calls keep their exact keys. Bucketed
keys end with `axolotl_bucket=pow2`. Before a winner is reused for another shape in its bucket,
quack's validity pruning (plus any matching `autotune_profiles`) runs for the actual call; a winner
it rejects sends that call to a sweep of its own exact key (`bucket_fallback: true` in its record).
Records of bucketed keys carry `bucket_reuses`, the number of other shapes that reused the winner;
`call_args` keeps the shapes of the call that was tuned.

Each quack record also carries `swept_configs_sha256`, a fingerprint of the candidates that key
was swept over after `autotune_profiles` and quack's pruning. A winner seeded from the file is
checked at its first lookup in the run: when this run's profiles resolve to a different candidate
set, the key (or bucket) is swept again, since the old winner was chosen among other configs. A
seeded bucket winner that is not among the call's candidates at all also re-tunes the bucket
instead of falling back to exact keys. Either way the new sweep overwrites the stale record at the
next save (a record swept over a different candidate set always replaces the one on file), so a
profile change heals the file once instead of turning the bucket back into per-token-count keys
every run. Records written before the fingerprint existed skip the first check. With
`quack_autotune_key_buckets: none` the check still runs for seeded keys that carry a fingerprint.

`quack_autotune: false` never benchmarks: each key gets one config by a fixed rule, so a new shape
costs at most a compile of that config (`source: heuristic` in the records). Matching
`autotune_profiles` still decide the candidates; the rule then picks among them.

| kernel | rule (B200) |
|---|---|
| `gemm_gated_tuned` | 256x512, cluster 2x1, persistent |
| `gemm_dgated_tuned` | 256x256, cluster 2x1, persistent |
| `gemm_tuned`, N <= 128 | 128x64, cluster 2x1 with `cu_seqlens_k` else 1x1, persistent |
| `gemm_tuned`, K <= 128 < N | 128x256, cluster 1x2, not persistent |
| `gemm_tuned` otherwise, other kernels | 256x256, cluster 2x1, persistent |

All use `use_tma_gather` exactly when `A_idx` is passed, no pingpong and no `swap_ab`. When the
rule's config is not a valid candidate (another GPU generation, or pruned for the call) the first
valid candidate is used, logged once per kernel.

```yaml
use_sonicmoe: true
quack_autotune_key_buckets: pow2   # default with use_sonicmoe; shape = no token dims; none = exact shapes
quack_autotune: false              # default true (sweep)
```

With `use_sonicmoe` (or any quack `autotune_profiles`), the ~200 ms GPU warm-up matmul quack runs
before each new key's sweep is skipped.
