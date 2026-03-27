"""EBFT (Energy-Based Fine-Tuning) Strategy for training

Two modes:
- structured: For QA data with prompt/completion splits. Uses GRPOTrainer + vLLM.
- strided: For unstructured text (raw code, prose). Uses strided block-parallel generation.
"""

from typing import Any

from axolotl.core.trainers.ebft.args import (
    AxolotlAsyncEBFTConfig,
    AxolotlEBFTConfig,
    AxolotlStridedEBFTConfig,
)
from axolotl.utils.dict import DictDefault


def _get_ebft_mode(cfg: DictDefault) -> str:
    """Determine EBFT mode from config."""
    if cfg.ebft and hasattr(cfg.ebft, "mode") and cfg.ebft.mode:
        return cfg.ebft.mode
    return "structured"


class EBFTStrategy:
    """Strategy for EBFT training — dispatches between structured and strided modes."""

    @classmethod
    def get_trainer_class(cls, cfg: DictDefault | None = None):
        mode = _get_ebft_mode(cfg) if cfg else "structured"
        if mode == "strided":
            from axolotl.core.trainers.ebft.strided import AxolotlStridedEBFTTrainer

            return AxolotlStridedEBFTTrainer

        # Structured mode: async or sync
        # use_data_producer also triggers async trainer (needed for LoRA sync
        # without async_prefetch, since sync trainer lacks LoRA sync support)
        use_async = (
            cfg
            and cfg.trl
            and (
                getattr(cfg.trl, "async_prefetch", False)
                or getattr(cfg.trl, "use_data_producer", False)
            )
        )
        if use_async:
            from axolotl.core.trainers.ebft.trainer import AxolotlAsyncEBFTTrainer

            return AxolotlAsyncEBFTTrainer
        from axolotl.core.trainers.ebft.trainer import AxolotlEBFTTrainer

        return AxolotlEBFTTrainer

    @classmethod
    def get_training_args_class(cls, cfg: DictDefault | None = None):
        mode = _get_ebft_mode(cfg) if cfg else "structured"
        if mode == "strided":
            return AxolotlStridedEBFTConfig

        # Structured mode: async or sync config
        use_async = (
            cfg
            and cfg.trl
            and (
                getattr(cfg.trl, "async_prefetch", False)
                or getattr(cfg.trl, "use_data_producer", False)
            )
        )
        if use_async:
            return AxolotlAsyncEBFTConfig
        return AxolotlEBFTConfig

    @classmethod
    def is_strided(cls, cfg: DictDefault) -> bool:
        return _get_ebft_mode(cfg) == "strided"

    @classmethod
    def set_training_args_kwargs(cls, cfg: DictDefault) -> dict[str, Any]:
        """Map axolotl YAML config fields to training args kwargs."""
        kwargs: dict[str, Any] = {}
        mode = _get_ebft_mode(cfg)

        # Common EBFT fields
        ebft = cfg.ebft
        if ebft:
            if ebft.feature_layers is not None:
                kwargs["ebft_feature_layers"] = ebft.feature_layers
            if ebft.embed_method is not None:
                kwargs["ebft_embed_method"] = ebft.embed_method
            if ebft.use_whitening is not None:
                kwargs["ebft_use_whitening"] = ebft.use_whitening
            if ebft.alignment_coef is not None:
                kwargs["ebft_alignment_coef"] = ebft.alignment_coef
            if ebft.diversity_coef is not None:
                kwargs["ebft_diversity_coef"] = ebft.diversity_coef
            if ebft.ce_coef is not None:
                kwargs["ebft_ce_coef"] = ebft.ce_coef
            if getattr(ebft, "adaptive_max_tokens", None) is not None:
                kwargs["ebft_adaptive_max_tokens"] = ebft.adaptive_max_tokens
            if getattr(ebft, "gt_length_multiplier", None) is not None:
                kwargs["ebft_gt_length_multiplier"] = ebft.gt_length_multiplier

        if mode == "strided":
            # Strided-specific fields
            if ebft:
                if ebft.stride is not None:
                    kwargs["ebft_stride"] = ebft.stride
                if ebft.context_length is not None:
                    kwargs["ebft_context_length"] = ebft.context_length
                if ebft.generate_max_len is not None:
                    kwargs["ebft_generate_max_len"] = ebft.generate_max_len
                if ebft.n_samples_per_prompt is not None:
                    kwargs["ebft_n_samples_per_prompt"] = ebft.n_samples_per_prompt
                if ebft.temperature is not None:
                    kwargs["ebft_temperature"] = ebft.temperature
                if ebft.top_p is not None:
                    kwargs["ebft_top_p"] = ebft.top_p
                if ebft.rl_coef is not None:
                    kwargs["ebft_rl_coef"] = ebft.rl_coef
                if ebft.advantage_estimator is not None:
                    kwargs["ebft_advantage_estimator"] = ebft.advantage_estimator
                if ebft.min_completion_prefix is not None:
                    kwargs["ebft_min_completion_prefix"] = ebft.min_completion_prefix
        else:
            # Structured mode: map TRL config fields
            trl = cfg.trl
            if trl:
                if trl.use_vllm:
                    kwargs["use_vllm"] = trl.use_vllm
                    if trl.vllm_mode:
                        kwargs["vllm_mode"] = trl.vllm_mode
                    if trl.vllm_mode == "colocate":
                        kwargs["vllm_enable_sleep_mode"] = trl.vllm_enable_sleep_mode
                        vllm_cfg = cfg.vllm
                        if vllm_cfg:
                            kwargs["vllm_gpu_memory_utilization"] = (
                                vllm_cfg.gpu_memory_utilization
                            )
                            kwargs["vllm_tensor_parallel_size"] = (
                                vllm_cfg.tensor_parallel_size
                            )
                    kwargs["vllm_server_host"] = trl.vllm_server_host or (
                        trl.vllm.host if trl.vllm else None
                    )
                    kwargs["vllm_server_port"] = trl.vllm_server_port or (
                        trl.vllm.port if trl.vllm else None
                    )
                    if trl.vllm_server_timeout:
                        kwargs["vllm_server_timeout"] = trl.vllm_server_timeout

                if trl.num_generations:
                    kwargs["num_generations"] = trl.num_generations
                if trl.max_completion_length is not None:
                    kwargs["max_completion_length"] = trl.max_completion_length
                if trl.temperature is not None:
                    kwargs["temperature"] = trl.temperature
                if trl.top_p is not None:
                    kwargs["top_p"] = trl.top_p
                if trl.top_k is not None:
                    kwargs["top_k"] = trl.top_k
                if trl.min_p is not None:
                    kwargs["min_p"] = trl.min_p
                if trl.num_iterations is not None:
                    kwargs["num_iterations"] = trl.num_iterations
                if trl.epsilon is not None:
                    kwargs["epsilon"] = trl.epsilon
                if trl.epsilon_high is not None:
                    kwargs["epsilon_high"] = trl.epsilon_high
                if trl.scale_rewards is not None:
                    kwargs["scale_rewards"] = trl.scale_rewards
                if trl.loss_type is not None:
                    kwargs["loss_type"] = trl.loss_type
                if trl.mask_truncated_completions is not None:
                    kwargs["mask_truncated_completions"] = (
                        trl.mask_truncated_completions
                    )
                if trl.log_completions is not None:
                    kwargs["log_completions"] = trl.log_completions
                if trl.num_completions_to_print is not None:
                    kwargs["num_completions_to_print"] = trl.num_completions_to_print
                if trl.sync_ref_model:
                    kwargs["sync_ref_model"] = trl.sync_ref_model
                if trl.repetition_penalty is not None:
                    kwargs["repetition_penalty"] = trl.repetition_penalty
                if trl.generation_kwargs is not None:
                    kwargs["generation_kwargs"] = trl.generation_kwargs
                if trl.chat_template_kwargs is not None:
                    kwargs["chat_template_kwargs"] = trl.chat_template_kwargs

                # Async prefetch fields (only pass when enabled — sync config doesn't have these)
                if getattr(trl, "async_prefetch", False):
                    kwargs["async_prefetch"] = trl.async_prefetch
                    if getattr(trl, "vllm_sync_interval", None) is not None:
                        kwargs["vllm_sync_interval"] = trl.vllm_sync_interval
                if getattr(trl, "vllm_lora_sync", False):
                    kwargs["vllm_lora_sync"] = trl.vllm_lora_sync

        return kwargs

    @classmethod
    def set_trainer_args(cls, cfg: DictDefault) -> list[Any]:
        return []

    @classmethod
    def set_trainer_kwargs(cls, cfg: DictDefault) -> dict[str, Any]:
        return {}

    @classmethod
    def get_blocklist_args_kwargs(cls, cfg: DictDefault | None = None) -> list[str]:
        mode = _get_ebft_mode(cfg) if cfg else "structured"
        if mode == "strided":
            return [
                "dataset_num_proc",
                "max_length",
                "max_prompt_length",
                "include_tokens_per_second",
                "beta",
            ]
        return [
            "dataset_num_proc",
            "max_length",
            "include_tokens_per_second",
            "max_prompt_length",
        ]

    @classmethod
    def get_collator(cls, *args, **kwargs):
        return None
