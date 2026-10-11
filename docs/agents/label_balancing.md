# Label Balancing — Agent Reference

Implementation details for `balance_labels`. User-facing guidance lives in
[multipack.qmd](../multipack.qmd#balancing-supervised-tokens).

## Integration status

The sampler paths below run the within-optimizer-step refinement with
`window_steps=1`. The helper also supports wider windows, but the experimental
32-step then within-step pipeline is not wired into the trainer.
`rank_balance.order_batches_by_rank` is the final ordering pass in both map-style
training samplers when `dp_count > 1`. It preserves optimizer-step membership.
Rank permutations are exhaustive up to four ranks; larger jobs use at most 24
deterministic cyclic/reversed permutations per layout. The 32-step refinement
experiment remains separate from this production rank-ordering pass.

## Balancing supervised tokens

For datasets with masked prompts, similarly sized packed batches can contain very
different numbers of supervised tokens. Enable label balancing with:

```yaml
sample_packing: true
balance_labels: true
```

The sampler first packs by context length, then balances the number of unmasked
labels across full batches in windows of 64 batches. It groups complementary
packed rows, attempts capacity-safe sample swaps, and tries local repacking when
swaps cannot improve the balance. Search is limited to four passes per window.
Only improvements are retained: no extra rows are allocated, no retained samples
are added or removed, and the incomplete final batch is left unchanged. Aggregate
packing utilization is preserved; individual batches can have different token
occupancy. Trainer and streaming integration also respect the collator's padding
multiple, rejecting changes that would increase the total padded tensor size.
Exact label equality is not guaranteed for indivisible samples.

This works with both flattened batches and `multipack_real_batches: true`. These
are local microbatch layouts. With packing-aware attention, the default is one
packed row with capacity `micro_batch_size * sequence_len` per rank. With
`multipack_real_batches: true`, each rank instead receives `micro_batch_size`
packed rows, each with capacity `sequence_len`. Under the default Accelerate
`split_batches: false`, data parallelism shards whole microbatches across ranks.
Four ranks with microbatch size one therefore process four rows globally; this
is distinct from balancing four rows jointly inside one rank's microbatch.
Tokenized label counts are read once per sampler construction and reused across
epochs. For causal loss, counts exclude the first label of each packed row;
precomputed `shift_labels` are counted directly. Exchanges also preserve the total
number of loss-bearing labels. The resulting full batches are shuffled within
each window using the training seed and epoch.

For streaming SFT and pretraining, balancing runs independently within each
`streaming_multipack_buffer_size` chunk, after tokenization and length filtering.
It uses the existing flattened capacity of `micro_batch_size * sequence_len`.
When pretraining examples omit labels, counts match the labels created from
`input_ids` during collation. No label metadata or samples are retained across
chunks. Fully supervised pretraining already has label counts close to context
lengths, so it may benefit less than data with masked prompts.

Label balancing requires causal LM data with fixed tokenized labels (or pretraining
labels derived from input IDs) and right padding. It is incompatible with
sequential/curriculum sampling, reward models, and diffusion or RL training. It
applies to training only, including packed training. For map-style data, an additional accumulation-aware ordering pass smooths
update totals using the configured gradient accumulation and rank count. This
does not replace token-weighted loss normalization. Collators that dynamically
change label masking are not supported.

`generate_batches()` exposes `sampler.label_metrics` with `before` and `after`
summaries when label metadata is available. Each summary contains `batches`,
`packed_rows`, `mean_packed_length` (without padding), `mean_label_count`,
`std_label_count` (population standard deviation), and `total_label_count`.
Label counts are per microbatch, summed over its packed rows. Passing
`set_stats=True` also logs the summaries, including when batches were previously
cached. These describe the generated plan after `drop_last`, before distributed
sharding or minimum-length truncation; they are not measured per-rank training
or gradient-accumulation statistics.

With balancing enabled, random sampling and balancing use the configured seed
(`data_seed`, falling back to `seed`) plus epoch, independently of rank-local RNG state. Identical input datasets,
metadata, and packing settings therefore produce identical plans before rank
sharding. Existing distributed length synchronization and sharding remain in
place. Custom input samplers must themselves yield the same order on each rank
when used with this shared-plan sharding scheme.

Checkpoint resume reconstructs the balanced plan from its seed and epoch; the
Trainer skips the microbatches already consumed in that epoch. Axolotl forwards
the restored epoch to both balanced samplers before Accelerate adds its resume
wrappers, clearing any cached plan from another epoch. New checkpoints save `balanced_sampler.json` with sampler settings, metadata
hashes, dataset fingerprint, epoch length, and consumed batches. Resume validates
the schedule fields before skipping; fingerprint mismatches only warn. Legacy checkpoints reconstruct the offset from
`TrainerState.epoch` and require an integer optimizer-step boundary. Packing is
deterministic for a given seed and epoch, but different epochs can pack to
different lengths.
The samplers do not keep
a second consumed-batch cursor, which would conflict with Trainer skipping and
DataLoader prefetching. Exact continuation requires the same dataset order,
seed, packing settings, microbatch size, data-parallel size and accumulation
grouping, with `ignore_data_skip: false`. This applies to map-style training;
streaming still relies on the iterable dataset's existing resume behavior.

## Balancing fixed-count padded or flattened batches

Label balancing also works with ordinary padded attention, without sample
packing or batch flattening:

```yaml
sample_packing: false
batch_flattening: false
balance_labels: true
micro_batch_size: 4
```

`LabelBalancedRandomSampler` preserves the number of original samples per
microbatch. In padded mode, batch cost is the sample count multiplied by the
longest sequence length, rounded to the collator's padding multiple. Within each
window of at most 64 microbatches, it tries length grouping and four passes of
sample swaps. Changes must not increase label-count variance, padded-cost
variance, total padding, or peak batch cost. These objectives can conflict, in
which case the sampler leaves some imbalance instead of adding padding.

The padded path requires tokenized labels and the standard Axolotl or Transformers
seq2seq collator, with right padding, longest-sequence padding, and ignored padding
labels (`-100`). `pad_to_sequence_len` remains supported through the collator's
padding multiple. Counts exclude each sample's first label when the model uses a
shifted causal loss. Custom collators that rewrite labels are not supported.

For ordinary batch flattening, keep sample packing disabled:

```yaml
sample_packing: false
batch_flattening: true
balance_labels: true
micro_batch_size: 4
```

`LabelBalancedRandomSampler` yields ordinary scalar dataset indices. Each full
microbatch still contains exactly four original samples in this example; the
flattening collator concatenates them into one variable-length row. The sampler
balances both the sum of sequence lengths and the count of supervised labels
across microbatches. Counts account for the collator masking the first label of
every source sample, independently of the causal loss shift.

Within shuffled windows of at most 64 full microbatches, it tries greedy
regrouping followed by four passes of paired sample swaps. Candidates are accepted
only if neither length variance nor label-count variance increases, at least one
improves, and the window's peak flattened token length does not increase. This
preserves coverage, fixed sample counts and the original distributed drop-last
tail. It can leave unavoidable imbalance when the two objectives conflict.

The sampler uses the data seed (falling back to the training seed) and epoch,
independently of rank-local RNG. Existing DataLoader batching and distributed
sharding apply. Both fixed-count modes require a map-style tokenized dataset and
`accelerator_config.split_batches: false`. Flattened mode requires the standard
flattening collator with `separator_id=-100`. Neither mode can be combined with streaming
pretraining, curriculum sampling or length-grouped sampling. Streaming sample
packing continues to use its separate chunked balancing path.

`sampler.label_metrics` reports unpadded length mean, standard deviation and
maximum (`mean_unpadded_length`, `std_unpadded_length`, `max_unpadded_length`),
plus label mean, standard deviation and total. Batch-cost metrics include
`mean_batch_cost`, `std_batch_cost`, `max_batch_cost`, `total_batch_cost`, and
`padding_tokens`. Cost is the unpadded token sum in flattened mode and the padded
token slots in padded mode. These describe the
unsharded plan including the final partial batch. The accumulation-aware pass described below further smooths label totals
without changing any microbatch or its cost.


## Accumulation-aware ordering

For map-style packed, padded, and flattened training, `balance_labels`
receives a normalized `batches_per_optimizer_step` from the trainer: data-parallel
replicas multiplied by `gradient_accumulation_steps`. Both `MultipackBatchSampler`
and `LabelBalancedRandomSampler` use this same grouping value. With accumulation
steps times rank count greater than one, a second pass reorders whole microbatches across up to
16 complete optimizer updates. With `W` ranks and `G` accumulation steps, each
update spans `W * G` batches of the common plan; rank `r` consumes every `W`th
batch. This requires the standard whole-batch sharding (`split_batches: false`).

Greedy assignment balances the total labels across all ranks and microsteps in
each optimizer update. A candidate is accepted only if global-update label
variance decreases. Per-rank totals are not an objective. These
comparisons are against the plan after microbatch balancing, not the original
random plan. No microbatch contents, padding, lengths or individual label counts
change in this pass. The incomplete final accumulation window stays in place.
The pass can leave unavoidable imbalance; individual ranks' totals and temporal
variances are not constrained.

A final bounded swap pass refines microbatches within each completed optimizer
update, preserving that update's samples, total tokens, and supervised targets.
It preserves row cardinality and packed capacity, never increases peak cost,
padding cost, or within-update label/token variance, and uses token variance to
break cost ties. Padded cost is batch size times the rounded maximum sequence
length; flattened cost is total tokens. Packed cost includes row padding.
Unpacked microbatch size one and incomplete updates are left unchanged. With
fixed-count padded batches, the longest sample still determines the update's
peak padded size; token balancing does not necessarily lower that peak. These
costs approximate activation-memory demand, not measured GPU memory or runtime.

Metrics include `before`, `before_accumulation`, `before_rank` (before final rank
ordering), `before_microbatch` (after
optimizer-step grouping), and `after` summaries, with
`mean_global_update_labels`,
`std_global_update_labels`, `updates`, and `excluded_tail_microbatches`.
Update metrics exclude partial accumulation windows and partial microbatches;
ordinary microbatch statistics retain their existing tail semantics. These are
plan statistics, not measurements of runtime loss normalization or arbitrary
custom sharding. Streaming retains chunk-local microbatch balancing: this pass
is not enabled there because accumulation windows can cross chunk boundaries.

The rank-ordering beam search minimizes worst-rank cumulative squared size
changes, then cumulative workload imbalance, simultaneous rank spread, and total
squared size changes. Each step protects both simultaneous squared spread and
the sum of microstep maximum costs. These are tensor-size proxies, not measured
compute time. Tail batches remain in their original positions. Streaming uses
a content-derived chunk seed and does not apply cross-chunk rank ordering.

### Cross-step refinement window

`label_balance_window_optim_steps` defaults to 1 and must be a positive integer.
For values greater than 1, both map-style samplers run `balance_microbatches`
with that window, then with window 1, before rank ordering. The number of
rank-local microbatches is window × normalized `batches_per_optimizer_step`
(GAS × DP, excluding CP/TP). The helper protects global-step token and label
variance, capacity, and padding constraints. Incomplete windows skip cross-step
swaps but still receive within-step refinement. This setting is recorded in
checkpoint sampler settings when non-default; omitting the default preserves
compatibility with earlier manifests. Changing it on resume rejects exact replay.
Wider windows require `balance_labels` and non-streaming training; preprocessing
chunks cannot establish the eventual training optimizer-step boundaries.

### Replay validation and save failures

Dataset fingerprint mismatches warn; sampler settings and metadata hashes remain
strict. Equal hashes do not certify identical token contents. Invalid checkpoint
position bookkeeping writes a `replay_error` manifest marker while allowing the
normal checkpoint save to proceed. The marker must be rejected on exact resume,
not treated as a missing legacy manifest. `ignore_data_skip: true` bypasses data
replay validation and skips restoration of the previous data position. Packing
capacity uses `_train_batch_size` in training and `args.eval_batch_size` in eval,
including when label balancing is disabled.
