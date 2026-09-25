# Context Parallel (ringmaster)

Long-context attention via sequence parallelism, backed by the standalone
`ringmaster` package (`pip install axolotl[ringmaster]`). Ulysses / Ring / USP that wrap
existing HF attention kernels (FA2/FA3/FA4, sdpa, flex) — **no `flash_attn` pypi
dependency**. This is a built-in plugin: configure `context_parallel_size` or
`context_parallel` without adding a `plugins:` entry.

Requires **torch ≥ 2.13**. The integration targets the pinned upstream releases
**Transformers 5.17.0**, **Accelerate 1.15.0**, and **axolotl-ringmaster ≥0.2.1**.
No custom Transformers or Accelerate branch is required. Ringmaster 0.2.1 includes
the packed attention and recurrent context-parallel adapters.

## Usage

```yaml
flash_attention: true            # the kernel Ulysses wraps (FA2 here)

context_parallel:
  size: 8                        # total CP degree
  backend: auto                  # auto | ulysses | ring | usp
  # ulysses_size / ring_size: auto-selected from KV-head count + topology
  load_balance: auto             # head_tail for eligible pure Ring; none otherwise
  ring_impl: auto                # auto -> hf_kernels (FA2/3/4) for ring, else torch_native
```

The **auto-selector** picks the `ulysses_size × ring_size` split: pure Ulysses when
the CP degree divides the KV-head count and fits a node; Ring when KV heads are
scarce (MQA); USP otherwise (Ulysses intra-node × Ring inter-node).

## Training support

Ulysses can wrap SDPA or Flash Attention. Ring/USP training requires a Flash
Attention kernel (`ring_impl: hf_kernels` or automatic selection); Ringmaster's
`torch_native` block kernel is forward-only. Sample packing and batch flattening are supported with CP; automatic load balancing selects contiguous shards for packed inputs. GLM DSA kernels remain incompatible with packed CP.
SFT uses the model's causal LM loss and requires loss-kwargs support; custom loss
functions and label smoothing are rejected. GRPO/EBFT retain their output-gathering
path.

The adapter preserves Trainer's supervised-token count across the entire gradient
accumulation window, including unequal microbatches and partial final windows.

## Composition with FSDP2 / ND parallelism

Accelerate owns the `cp` mesh dimension, replicated batches within that group,
and FSDP2 gradient reduction. Ringmaster owns attention and sequence sharding.
The plugin bypasses native torch CP only on its own Trainer/Accelerator instances,
so it does not install DeepSpeed's Ulysses adapter or globally disable native CP.

For USP, the plugin derives `cp_ring` and `cp_ulysses` subgroups inside each CP
group while preserving Accelerate's original mesh for data loading and FSDP2.

Two-GPU FSDP2 regression tests compare Ulysses/SDPA, Ulysses/Flash Attention 2,
and Ring/Flash Attention losses and gradients with an unsharded tiny Llama. A separate eight-process CPU
test checks USP subgroup isolation across two data-parallel groups. Ringmaster
also tests USP attention outputs and q/k/v gradients on four CPU ranks; GPU USP
coverage remains hardware-dependent.

An opt-in 16-process CPU test (`pytest -m slow tests/integrations/test_context_parallel_mesh.py::test_ringmaster_nd_cpu_parity`)
uses Torch 2.13+, Accelerate mesh construction, real tensor-parallel linear layers,
and FSDP2/HSDP. It compares dense and packed attention outputs, parameter gradients,
and an SGD update against an unsharded reference, with different batches per DP
group and non-contiguous CP groups. The layouts are DP=2 × CP=4 × TP=2
(with Ring=2 × Ulysses=2) and DP-replicate=2 × DP-shard=2 × CP=2 × TP=2.
This exercises CPU math attention for USP and SDPA for Ulysses, not CUDA kernels or a complete
Accelerator/Trainer launch: Accelerate does not accept CPU ND execution through
its normal Accelerator validation.

## Recurrent models and optional kernels

Install `axolotl[fla,ringmaster]` for native FLA context parallelism and TileLang.
Both Docker UV images install these extras. GDN and KDA mixers require importable
FLA kernels exposing `cp_context`, a single batch row after flattening (`micro_batch_size: 1` or `batch_flattening: true`), contiguous shards, and
`use_cache: false`. Missing or incompatible FLA fails during setup, including for
CP degrees greater than two. The four-rank forward/backward parity test is marked
`slow`; the regular e2e suite has one small optional kernel smoke test.

`load_balance: auto` selects a compatible layout. Explicit incompatible options
raise instead of being silently ignored. `head_tail` and `distflash` require pure
Ring; their P2P schedules do not accept `rotate_method`. `per_document` and `ptrr`
are not implemented and are rejected.

Axolotl core owns packing and document-boundary metadata for Mamba, GDN, and KDA.
Ringmaster owns their distributed state propagation and convolution halos. Its
public `wire_recurrent_layers(model)` API installs instance-local adapters and
returns a wiring object with a `restore()` callback. Axolotl uses this same API
after model kernelization, so Mamba adapters preserve the selected Hub kernel's
normalization semantics.
Packed CP preserves global document boundaries for attention, FLA state passing, and Mamba convolution/scan resets. Ringmaster 0.2.1 is the minimum version for these packed CP paths.

## Architecture capabilities

Ringmaster uses generic attention and recurrent-layer detection. A model does
not need an architecture allowlist entry. ModelSupport descriptors may declare
`context_parallel` as `Supported` (verified coverage), `Experimental` (warn), or
`Unsupported` (fail with a reason). An absent descriptor or capability leaves the
generic path enabled; it is not a claim of verified accuracy. Runtime checks still
apply to the selected kernels, recurrent implementation, and shard layout.
