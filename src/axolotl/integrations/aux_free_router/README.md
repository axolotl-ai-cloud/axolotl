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
- Telemetry: future updates will log per-expert loads and bias magnitudes.
