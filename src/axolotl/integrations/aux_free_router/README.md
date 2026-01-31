# Aux-Loss-Free MoE Router Plugin

This integration adds an aux-loss-free (AFB) gating option to compatible MoE architectures without forking model code.

Summary
- Bias only affects expert selection (top-k); mixture weights come from unbiased logits.
- Per-expert token loads are accumulated on device and reduced across DP or EP groups.
- Bias is updated post-optimizer step outside autograd using EMA-smoothed loads.
- Existing aux loss is disabled when aux-free is enabled to avoid double signals.

Enable
- Add the plugin to your YAML, then set the aux-free toggle:

  plugins:
    - axolotl.integrations.aux_free_router.plugin.AuxFreeMoEPlugin

  moe_balance_type: noaux_tc
  moe_update_rate: 0.01        # default if unset
  moe_update_momentum: 0.9     # default if unset
  moe_bias_cap: 2.0            # default if unset
  moe_afb_warmup_steps: 100    # optional
  moe_bias_sync_group: world   # or 'ep' if expert_parallel_size > 1
  expert_parallel_size: 1      # set to your EP width when using moe_bias_sync_group: ep

Config keys
- moe_balance_type: gshard (auxiliary loss) | noaux_tc (aux-free). Default: model native.
- moe_update_rate: bias update rate (gamma). Default: 0.01.
- moe_update_momentum: EMA momentum for load smoothing. Default: 0.9.
- moe_bias_cap: absolute clamp for bias. Default: 2.0.
- moe_afb_warmup_steps: delay before applying updates. Default: 0.
- moe_bias_sync_group: reduction group for counts, 'world' (DP) or 'ep' (expert-parallel). Default: world.
- expert_parallel_size: number of ranks per expert-parallel group when using `moe_bias_sync_group: ep`. Defaults to 1 (world).

Compatibility
- Targeted families: Mixtral, Qwen3-MoE, Bailing/Ring 2.0, and Llama 4 text MoE layers.
- Pass-through: Models with native aux-free routing (e.g., DeepSeek-V3) are left unmodified; only telemetry may be added in future.

Notes
- If you also enable Liger’s aux-loss paths, the plugin neutralizes aux loss when aux-free is on.
- Telemetry: logs per-layer min/mean/max token loads, `|bias| max`, and bias sign flip fraction using the Trainer’s `logging_steps` cadence.
- Sample packing: packed batches are compatible with aux-free routing. Because load counts are accumulated on-device per expert before reduction, packing tends to smooth token histograms and reduce bias oscillation. Keep `pad_to_sequence_len: true` when packing to preserve the target token budget per expert.

Telemetry metrics
- `moe_afb/l{idx}_load_min|mean|max`: token frequency per expert after reduction (0–1 range, sums to 1).
- `moe_afb/l{idx}_bias_abs_max`: absolute maximum of the learned bias for the layer.
- `moe_afb/l{idx}_bias_sign_flip_frac`: fraction of experts whose bias sign changed since the previous step (simple oscillation indicator).

Usage tips
- Increase `logging_steps` if router telemetry becomes noisy for large jobs—the plugin follows the Trainer’s logging cadence.
- Compare aux-free vs. aux-loss load metrics by plotting the `load_*` series; aux-free typically tightens min/max spread without the auxiliary loss term.
