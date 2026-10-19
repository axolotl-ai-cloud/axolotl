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
disk-cache hit). A re-run that only seeds or pins a key keeps the measured record on file.

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
510 s per rank). None of their measured winners use `tile_n` 64 or 128, and a 2x2 cluster only
ever won with `tile_m` 256. This filter keeps every measured winner:

- `gemm_gated_tuned`: 128x256 c2x1, 256x512 c2x1
- `gemm_tuned`: 128x256 c2x1 and c1x1, 128x224 c2x1, 256x256 c2x1, 128x160 c1x2, 256x192 c2x2,
  128x192 c1x1
- `gemm_dgated_tuned`: 256x256 c2x1

It cuts each first sweep from 76 to 52 candidates for `gemm_gated_tuned` and `gemm_dgated_tuned`
and from 74 to 42 for `gemm_tuned`:

```yaml
autotune_profiles:
  - kernel: gemm_gated_tuned
    backend: quack
    gpu: B200
    allow:
      tile_n: [160, 192, 224, 256, 512]
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
expensive one, a range `match` pins the measured winners for every routed-row count from 262144
up, while smaller shapes (eval, short batches) keep the filtered search. With
`sonicmoe_ep_row_padding: highwater`, training sees a few shapes that are not powers of two,
which is exactly what a threshold covers and a list of exact sizes would miss. List these entries
before the `allow` entries: the first matching entry for a call decides.

```yaml
autotune_profiles:
  - kernel: gemm_gated_tuned          # winner at 512k and 1M rows
    backend: quack
    gpu: B200
    match: { "postact_out.dim0": { ge: 262144 } }
    configs:
      - { tile_m: 256, tile_n: 512, cluster_m: 2, cluster_n: 1, is_dynamic_persistent: true, use_tma_gather: true }
  - kernel: gemm_tuned                # 256x256 c2x1; 128x256 c1x1 also won at 1M rows
    backend: quack
    gpu: B200
    match: { "A.dim0": { ge: 262144 } }
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
candidate and is not benchmarked at all. The `gemm_tuned` entries leave `is_dynamic_persistent`
open, so that kernel runs a 4-config search.
