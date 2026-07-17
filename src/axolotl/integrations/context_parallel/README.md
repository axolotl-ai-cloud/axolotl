# Context Parallel (ringmaster)

Long-context attention via sequence parallelism, backed by the standalone
`ringmaster` package (`pip install axolotl-ringmaster`). Ulysses / Ring / USP that wrap
existing HF attention kernels (FA2/FA3/FA4, sdpa, flex) — **no `flash_attn` pypi
dependency**. Opt-in and independent of the legacy `context_parallel_size`
ring-flash-attn path.

Requires **torch ≥ 2.11**.

## Usage

```yaml
plugins:
  - axolotl.integrations.context_parallel.ContextParallelPlugin

flash_attention: true            # the kernel Ulysses wraps (FA2 here)

context_parallel:
  size: 8                        # total CP degree
  backend: auto                  # auto | ulysses | ring | usp
  # ulysses_size / ring_size: auto-selected from KV-head count + topology
  rotate_method: allgather       # ring KV movement
  load_balance: head_tail        # ring-only (Ulysses is inherently balanced)
  ring_impl: auto                # auto -> hf_kernels (FA2/3/4) for ring, else torch_native
```

The **auto-selector** picks the `ulysses_size × ring_size` split: pure Ulysses when
the CP degree divides the KV-head count and fits a node; Ring when KV heads are
scarce (MQA); USP otherwise (Ulysses intra-node × Ring inter-node).

## Status

Phase 1 (Ulysses) is implemented. Ring (Phase 2), USP forward (Phase 3),
Mamba/linear state-passing (Phase 4), and ALST memory toggles (Phase 5) are
staged in ringmaster. Packing/varlen is v2.

## Composition with FSDP2 / ND parallelism

The plugin resolves the `cp` group from accelerate's device mesh when present, so
it composes with FSDP2/TP. For a pure-CP run with no other parallelism it builds a
standalone CP mesh from the world group.
