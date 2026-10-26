# tree-recurrence

Prefix-shared ("tree") packing for recurrent and linear-attention layers.

RL rollouts share long prefixes: every completion of a prompt repeats it, and a
multi-turn conversation repeats every earlier turn. A tree-packed row (TRL's
`TreeRow`) forwards a shared prefix once, as a trie whose segments are laid out
depth-first, and attention reads each segment's root-to-segment path. A recurrent
layer scans its row in order, so it would carry state from the end of one branch into
its sibling. This package makes those layers follow the trie: each segment starts
from the state its parent ends in, and each short causal conv reads the last inputs
of its own path.

It depends on `torch` and `numpy` only. It does not depend on axolotl, and TRL can
use it directly.

## Use with transformers

```python
import tree_recurrence

support = tree_recurrence.install(model)  # after loading (and kernelizing) the model
if support.mode == "none":
    raise ValueError(support.reason)  # some layer would mix packed samples
shares_prefixes = support.mode == "tree"  # "chains": one chain per sample instead
```

`install` hooks the modeling module of each recurrent mixer at seams transformers
already has, so the mixers' own forwards run unchanged:

| Family | Models | Seam |
| --- | --- | --- |
| Gated DeltaNet | Qwen3.5, Qwen3.5-MoE, Qwen3-Next, OLMo-Hybrid | `causal_conv1d_fn`, `torch_chunk_gated_delta_rule` (transformers >= 5.15) |
| Mamba2 (SSD) | Mamba2, Nemotron-H, Bamba, Granite-MoE-Hybrid | `mamba2_split_conv1d_scan_combined`, `causal_conv1d_fn`, `mamba2_chunk_scan` |
| Gated short conv | LFM2, LFM2-MoE | the decoder layer's forward |

Each hook reads the tree from the `tree_*` kwargs the forward passes down, which
gradient checkpointing saves and replays, and passes everything else through. It
calls the function it replaced when it runs, so a kernel bound later (by
`kernelize`, say) is used too. `uninstall(model)` restores the modules.

Both TRL tree formats are read: TRL 1.16's `tree_kv_index` /
`tree_cu_seqlens_q` / `tree_cu_seqlens_k`, and huggingface/trl#7686's, which adds
`tree_segment_parents` and `tree_segment_depths`. A row of whole samples is a trie
without branches, so chains take the same path.

## How a tree runs

`TreePlan` is built once per forward on the host and kept on the
`tree_cu_seqlens_q` tensor. The driver then runs a family's kernels on it:

- `tree_conv` puts each segment's path history in front of it, in one call; the
  conv needs no boundaries.
- `tree_scan` either forks, scanning the trie level by level and starting each
  segment from its parent's final state, or rescans every root-to-segment path in
  one call and keeps each segment's tail. Forking needs a kernel that takes
  per-sequence initial states with varlen boundaries (FLA's chunked kernels); it is
  chosen when the prefix tokens it saves outweigh one more kernel call per level
  (`Policy(fork_min_saved_tokens=...)`).
- Mamba2's kernels reset at `seq_idx` but take initial states per batch row only, so
  Mamba2 rescans paths; the fused `mamba_split_conv1d_scan_combined` runs whole over
  the gathered paths.
- Without a varlen kernel (transformers' PyTorch references), every path becomes a
  row of one right-padded batch.

`tree_recurrence.reference` has PyTorch versions of every kernel with the kernels'
APIs, so all of this is tested on CPU against samples run alone.

## With TRL

TRL 1.16's `AsyncGRPOTrainer` refuses models with non-attention `layer_types`.
After `install`, build the trainer under
`tree_recurrence.integrations.trl.hidden_layer_types(model)`. For `chains` support,
`ChainRow`, `ChainTokenBudgetBatcher` and `ChainFixedCountBatcher` make every sample
its own chain. `tree_kwargs_from_batch` turns a collated batch of either format into
forward kwargs.

huggingface/trl#7686 adds TRL's own Gated DeltaNet tree kernel, `patch_tree_gdn`.
Each modeling module has one owner: a module TRL patched first is left to TRL, and a
module patched here is recorded in TRL's patched set, so TRL does not wrap it again.
`install(model, backends={"gated_delta_net": "trl"})` hands Gated DeltaNet to TRL when
TRL has it.

## Other models

A model that follows trees in its own code (an in-tree Kimi-Linear, say) builds a plan
with `tree_plan(kwargs)`, runs its kernels with `tree_conv` / `tree_scan` and the
adapters in `tree_recurrence.components` (`kda_scan`, `short_convolution`, ...),
marks its mixer's forward with `mark_tree_forward`, and registers a `TreeFamily`
naming its model types and mixer classes.

## Extras

- `fla`: flash-linear-attention, the varlen Gated DeltaNet and KDA kernels
- `mamba`: mamba-ssm and causal-conv1d
- `kernels`: hub kernels, e.g. `integrations.hub_kernels.use_hub_causal_conv1d`
- `transformers`, `trl`

## Not yet

- Forking Mamba2 state: per-depth batches with `initial_states`, padded with `dt`
  masked to zero after its softplus so the padding leaves the state alone.
- Falcon-H1 and Zamba2, whose mixers share Mamba2's seams, and transformers' own
  Kimi-Linear and Bailing KDA modules.
