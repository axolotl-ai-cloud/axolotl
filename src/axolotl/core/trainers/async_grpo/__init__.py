"""Async GRPO on TRL's experimental ``AsyncGRPOTrainer``.

TRL generates, scores and computes advantages continuously in a CUDA-free child
process that streams requests to a vLLM server, while the trainer consumes scored
samples from a queue (dropping those older than ``max_staleness`` policy versions),
tree-packs them by token budget and syncs weights every ``weight_sync_steps``.

On top of TRL, this trainer:

- takes the model axolotl loaded (quantized, patched, LoRA-wrapped) instead of a
  path, and plugs in axolotl's optimizer, scheduler and distributed mixins;
- drops samples that carry no gradient (``skip_zero_advantage_batches``,
  ``mask_truncated_completions``) before they are packed and forwarded;
- scores rewards in a process pool, several groups at a time
  (``reward_num_workers``);
- honors ``reward_weights``, ``multi_objective_aggregation`` and ``scale_rewards``;
- offers ``loss_type: cispo``, which weights each token by its policy/sampler ratio,
  capped at ``vllm_importance_sampling_cap``, instead of clipping that ratio, and,
  experimentally on TRL main, ``loss_type: score_centering``, TRL's score centering;
- syncs merged LoRA weights without merging into the trained model, bitwise equal to
  the ``lora_merge_aware`` forward, and measures how much of the update the merge
  keeps (``vllm_merge_check_interval``);
- with adapter-only sync, keeps each adapter it retires loadable by vLLM, which may
  still hold requests for it, rather than deleting it;
- runs a custom rollout loop (``rollout_loop``), e.g. one that delegates whole
  rollouts to an agent server, and serves the synced adapter under a fixed name
  for clients that cannot follow TRL's versioned one (``vllm_lora_alias``).
"""
