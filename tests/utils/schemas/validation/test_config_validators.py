"""
Tests for new config validators added to AxolotlInputConfig.

Covers:
  - save_strategy: 'best' requires metric_for_best_model
  - streaming=True with val_set_size > 0 is rejected
  - lora_target_modules with invalid regex patterns is rejected
  - GRPO: generation batch size must be divisible by num_generations,
    num_generations >= 2, and effective_gbs >= num_generations * world_size
  - context_parallel_size > 1 requires kernel-backed attention
  - use_dft / use_eaft are SFT-only and reject each other, RL, reward models, CP and fused CE
  - vllm_mode: colocate rejects CP > 1, vllm_lora_sync, and async GRPO with an adapter
"""

from unittest.mock import patch

import pytest

from axolotl.utils.config import validate_config
from axolotl.utils.dict import DictDefault
from axolotl.utils.schemas.config import AxolotlInputConfig


class TestSaveStrategyBestValidator:
    """save_strategy: 'best' must be accompanied by metric_for_best_model."""

    def test_save_strategy_best_without_metric_raises(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(save_strategy="best")
        with pytest.raises(ValueError, match="metric_for_best_model"):
            validate_config(cfg)

    def test_save_strategy_best_with_metric_passes(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            save_strategy="best",
            metric_for_best_model="eval_loss",
        )
        validated = validate_config(cfg)
        assert validated.save_strategy == "best"
        assert validated.metric_for_best_model == "eval_loss"

    def test_save_strategy_epoch_without_metric_passes(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(save_strategy="epoch")
        validated = validate_config(cfg)
        assert validated.save_strategy == "epoch"

    def test_save_strategy_no_without_metric_passes(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(save_strategy="no")
        validated = validate_config(cfg)
        assert validated.save_strategy == "no"

    def test_save_strategy_unset_without_metric_passes(self, min_base_cfg):
        """The default (None / not set) should not require metric_for_best_model."""
        validated = validate_config(min_base_cfg)
        assert validated.save_strategy is None


class TestStreamingWithValSetSizeValidator:
    """streaming=True is incompatible with val_set_size > 0."""

    def test_streaming_with_val_set_size_raises(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            streaming=True, val_set_size=0.1, max_steps=100
        )
        with pytest.raises(ValueError, match="val_set_size"):
            validate_config(cfg)

    def test_streaming_with_val_set_size_zero_passes(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            streaming=True, val_set_size=0.0, max_steps=100
        )
        validated = validate_config(cfg)
        assert validated.streaming is True

    def test_streaming_false_with_val_set_size_passes(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(streaming=False, val_set_size=0.1)
        validated = validate_config(cfg)
        assert validated.val_set_size == pytest.approx(0.1)

    def test_streaming_unset_with_val_set_size_passes(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(val_set_size=0.2)
        validated = validate_config(cfg)
        assert validated.val_set_size == pytest.approx(0.2)


class TestLoraTargetModulesRegexValidator:
    """lora_target_modules entries must be valid Python regex patterns."""

    def test_invalid_regex_raises(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            adapter="lora",
            lora_target_modules=["q_proj", "[invalid_regex"],
        )
        with pytest.raises(ValueError, match="invalid regex pattern"):
            validate_config(cfg)

    def test_valid_regex_passes(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            adapter="lora",
            lora_target_modules=["q_proj", "v_proj", r".*_proj"],
        )
        validated = validate_config(cfg)
        assert "q_proj" in validated.lora_target_modules

    def test_plain_module_names_pass(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            adapter="lora",
            lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        validated = validate_config(cfg)
        assert len(validated.lora_target_modules) == 4

    def test_lora_target_linear_string_not_validated(self, min_base_cfg):
        """When lora_target_modules is a string (e.g. 'all-linear'), skip regex check."""
        cfg = min_base_cfg | DictDefault(
            adapter="lora",
            lora_target_modules="all-linear",
        )
        # Should not raise
        validate_config(cfg)

    def test_multiple_invalid_patterns_reported(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            adapter="lora",
            lora_target_modules=["[bad1", "[bad2"],
        )
        with pytest.raises(ValueError, match="invalid regex pattern"):
            validate_config(cfg)


class TestGRPOBatchSizeValidator:
    """GRPO requires (mb*GA*world_size) % num_generations == 0 and num_generations >= 2.

    These call the @model_validator(mode="before") classmethod directly on a
    plain dict — same input shape it receives during full Pydantic validation,
    just without dragging in unrelated fields (datasets / model loading / etc.)
    that aren't relevant to what's under test. The validator is registered on
    ``RLValidationMixin`` (which ``AxolotlInputConfig`` inherits) so this is the
    same code path ``axolotl train`` exercises.
    """

    @staticmethod
    def _check(data):
        from axolotl.utils.schemas.validation import RLValidationMixin

        return RLValidationMixin.check_grpo_batch_size_divisibility(data)

    def test_divisible_passes(self):
        data = {
            "rl": "grpo",
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "trl": {"num_generations": 4},
        }
        # Should return data unchanged (no exception)
        out = self._check(data)
        assert out["trl"]["num_generations"] == 4

    def test_non_divisible_raises(self):
        data = {
            "rl": "grpo",
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 2,
            "trl": {"num_generations": 4},
        }
        with pytest.raises(ValueError, match="num_generations"):
            self._check(data)

    def test_non_divisible_error_includes_fix_hint(self):
        data = {
            "rl": "grpo",
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 3,
            "trl": {"num_generations": 4},
        }
        with pytest.raises(ValueError, match="gradient_accumulation_steps: 4"):
            self._check(data)

    def test_num_generations_one_raises(self):
        data = {
            "rl": "grpo",
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "trl": {"num_generations": 1},
        }
        with pytest.raises(ValueError, match=r"num_generations >= 2"):
            self._check(data)

    def test_explicit_generation_batch_size_divisible_passes(self):
        data = {
            "rl": "grpo",
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "trl": {"num_generations": 4, "generation_batch_size": 8},
        }
        out = self._check(data)
        assert out["trl"]["generation_batch_size"] == 8

    def test_explicit_generation_batch_size_non_divisible_raises(self):
        data = {
            "rl": "grpo",
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "trl": {"num_generations": 4, "generation_batch_size": 6},
        }
        with pytest.raises(ValueError, match="trl.generation_batch_size"):
            self._check(data)

    def test_non_grpo_skips_check(self):
        # Anything other than rl=grpo should pass through untouched, even
        # with non-divisible batch sizes — they're irrelevant to other RL
        # methods that don't use group-relative advantages.
        data = {
            "rl": "dpo",
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 3,
            "trl": {"num_generations": 4},
        }
        assert self._check(data) is data

    def test_no_rl_set_skips_check(self):
        data = {
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 3,
        }
        assert self._check(data) is data

    def test_grpo_without_num_generations_skips_check(self):
        # If num_generations isn't set, TRL uses its own default — we don't
        # have enough info to validate, so the validator must short-circuit
        # rather than guess.
        data = {
            "rl": "grpo",
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 3,
            "trl": {},
        }
        out = self._check(data)
        assert out["rl"] == "grpo"

    def test_multi_rank_group_size_check(self):
        data = {
            "rl": "grpo",
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 2,  # gbs = 1*2*2 = 4
            "world_size": 2,  # need gbs >= 4*2 = 8
            "trl": {"num_generations": 4},
        }
        with pytest.raises(ValueError, match=r"world_size=2"):
            self._check(data)

    def test_multi_rank_derived_gbs_includes_world_size(self):
        data = {
            "rl": "grpo",
            "micro_batch_size": 4,
            "gradient_accumulation_steps": 1,  # gbs = 4*1*2 = 8 >= 4*2
            "world_size": 2,
            "trl": {"num_generations": 4},
        }
        out = self._check(data)
        assert out["gradient_accumulation_steps"] == 1

    def test_multi_rank_divisibility_uses_world_size(self):
        data = {
            "rl": "grpo",
            "micro_batch_size": 3,
            "gradient_accumulation_steps": 1,  # gbs = 3*1*2 = 6
            "world_size": 2,
            "trl": {"num_generations": 4},
        }
        with pytest.raises(ValueError, match="gradient_accumulation_steps: 2"):
            self._check(data)

    def test_multi_rank_explicit_generation_batch_size_is_global(self):
        data = {
            "rl": "grpo",
            "micro_batch_size": 4,
            "gradient_accumulation_steps": 1,
            "world_size": 2,
            "trl": {"num_generations": 4, "generation_batch_size": 4},
        }
        with pytest.raises(ValueError, match=r"world_size=2"):
            self._check(data)


class TestBatchSizeFieldsValidator:
    """``micro_batch_size`` and ``gradient_accumulation_steps`` both default to
    1, so a config that sets only one of them (or neither) is still fully
    specified and must not be rejected by ``check_batch_size_fields``.

    ``check_batch_size_fields`` runs in ``mode="before"`` (before field defaults
    are applied), so these guard against a regression where relying on the
    documented defaults raised "At least two of ... must be set".
    """

    @staticmethod
    def _base():
        """``min_base_cfg`` sets both batch fields explicitly, which is exactly
        the case that already worked; build a config without them instead."""
        return DictDefault(
            base_model="HuggingFaceTB/SmolLM2-135M",
            learning_rate=1e-3,
            datasets=[
                {
                    "path": "mhenrichsen/alpaca_2k_test",
                    "type": "alpaca",
                },
            ],
        )

    def test_only_gradient_accumulation_steps_passes(self):
        cfg = self._base() | DictDefault(gradient_accumulation_steps=4)
        validated = validate_config(cfg)
        assert validated.gradient_accumulation_steps == 4
        assert validated.micro_batch_size == 1

    def test_only_micro_batch_size_passes(self):
        cfg = self._base() | DictDefault(micro_batch_size=4)
        validated = validate_config(cfg)
        assert validated.micro_batch_size == 4
        assert validated.gradient_accumulation_steps == 1

    def test_neither_set_relies_on_defaults(self):
        validated = validate_config(self._base())
        assert validated.micro_batch_size == 1
        assert validated.gradient_accumulation_steps == 1


class TestContextParallelAttnImplValidator:
    """CP requires a kernel-backed attention implementation."""

    @pytest.mark.parametrize(
        "impl",
        [
            "flash_attention_2",
            "flash_attention_3",
            "flash_attention_4",
            "kernels-community/flash-attn2",
            "kernels-community/flash-attn3",
            "sdpa",
            "flex_attention",
        ],
    )
    def test_kernel_backends_allowed(self, min_base_cfg, impl, monkeypatch):
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        cfg = min_base_cfg | DictDefault(
            context_parallel_size=2,
            attn_implementation=impl,
        )
        validated = validate_config(cfg)
        assert validated.context_parallel_size == 2
        assert validated.attn_implementation == impl

    def test_eager_rejected(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            context_parallel_size=2,
            attn_implementation="eager",
        )
        with pytest.raises(ValueError, match="kernel-backed attention"):
            validate_config(cfg)


class TestStagedNF4ConstraintTable:
    """check_staged_nf4 raises from a table, so an empty or messageless table would pass silently."""

    def test_every_constraint_carries_a_message(self):
        from axolotl.utils.schemas.validation import STAGED_NF4_CONSTRAINTS

        assert STAGED_NF4_CONSTRAINTS
        for predicate, message in STAGED_NF4_CONSTRAINTS:
            assert callable(predicate)
            assert isinstance(message, str) and message


class TestValueDependentInitOnExpertTargets:
    """PEFT's value-dependent inits read base_layer.weight, which fused experts lack;
    the failure must surface at validation, not after the model has loaded."""

    @pytest.mark.parametrize("init", ["pissa", "olora", "loftq", "corda", "eva"])
    def test_value_dependent_init_with_target_parameters_is_rejected(
        self, min_base_cfg, init
    ):
        cfg = min_base_cfg | DictDefault(
            adapter="lora",
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            lora_target_parameters=["mlp.experts.gate_up_proj"],
            peft_init_lora_weights=init,
        )
        with pytest.raises(ValueError, match="lora_target_parameters"):
            validate_config(cfg)

    @pytest.mark.parametrize("init", [None, True, "gaussian"])
    def test_default_init_with_target_parameters_passes(self, min_base_cfg, init):
        cfg = min_base_cfg | DictDefault(
            adapter="lora",
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            lora_target_parameters=["mlp.experts.gate_up_proj"],
        )
        if init is not None:
            cfg["peft_init_lora_weights"] = init
        validate_config(cfg)


class TestBnbBlocksizeValidator:
    """bitsandbytes 4-bit loading always quantizes at blocksize 64, so a different
    request must fail rather than silently train at 64 and merge at the requested size."""

    def _cfg(self, min_base_cfg, **kwargs):
        return min_base_cfg | DictDefault(
            adapter="qlora",
            load_in_4bit=True,
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            lora_target_linear=True,
            **kwargs,
        )

    def test_non_default_blocksize_rejected(self, min_base_cfg):
        cfg = self._cfg(min_base_cfg, bnb_config_kwargs={"blocksize": 128})
        with pytest.raises(ValueError, match="blocksize"):
            validate_config(cfg)

    def test_default_blocksize_passes(self, min_base_cfg):
        cfg = self._cfg(min_base_cfg, bnb_config_kwargs={"blocksize": 64})
        validate_config(cfg)

    def test_unset_blocksize_passes(self, min_base_cfg):
        validate_config(self._cfg(min_base_cfg))

    def test_blocksize_without_4bit_ignored(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(bnb_config_kwargs={"blocksize": 128})
        validate_config(cfg)


class TestCustomLossConflictsValidator:
    """use_dft / use_eaft need full logits and unsharded labels in the SFT trainer."""

    @staticmethod
    def _check(data):
        from axolotl.utils.schemas.validation import OptimizationValidationMixin

        return OptimizationValidationMixin.check_custom_loss_conflicts(data)

    def test_dft_and_eaft_together_raises(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(use_dft=True, use_eaft=True)
        with pytest.raises(ValueError, match="cannot be enabled together"):
            validate_config(cfg)

    def test_dft_alone_passes(self, min_base_cfg):
        validated = validate_config(min_base_cfg | DictDefault(use_dft=True))
        assert validated.use_dft is True

    @pytest.mark.parametrize("loss", ["use_dft", "use_eaft"])
    @pytest.mark.parametrize(
        "conflict",
        [
            {"rl": "dpo"},
            {"reward_model": True},
            {"process_reward_model": True},
            {"cut_cross_entropy": True},
            {"liger_fused_linear_cross_entropy": True},
            {"chunked_cross_entropy": True},
            {"context_parallel_size": 2},
            {"sequence_parallel_degree": 2},
        ],
    )
    def test_conflict_raises(self, loss, conflict):
        with pytest.raises(ValueError, match="SFT-only"):
            self._check({loss: True, **conflict})


class TestVllmColocateCompat:
    """vllm_mode: colocate rejects settings that only work with a vLLM server."""

    @pytest.mark.parametrize(
        "extra,trl,match",
        [
            ({"context_parallel_size": 2}, {}, "colocate` is not supported with"),
            ({}, {"vllm_lora_sync": True}, "requires `vllm_mode: server`"),
            (
                {"adapter": "lora"},
                {"async_prefetch": True},
                "vllm_mode: colocate` is not supported; use `server`",
            ),
        ],
    )
    def test_incompatible_settings_raise(self, min_base_cfg, extra, trl, match):
        cfg = min_base_cfg | DictDefault(
            rl="grpo",
            trl={"use_vllm": True, "vllm_mode": "colocate", **trl},
            vllm={"max_model_len": 1024},
            **extra,
        )
        with pytest.raises(ValueError, match=match):
            validate_config(cfg)


class TestAsyncGRPOCompat:
    """async_prefetch with rl: grpo runs on TRL's AsyncGRPOTrainer."""

    @staticmethod
    def _cfg(min_base_cfg, trl=None, **extra):
        return min_base_cfg | DictDefault(
            rl="grpo",
            max_steps=10,
            trl={"use_vllm": True, "async_prefetch": True, **(trl or {})},
            **extra,
        )

    def test_supported_config_passes(self, min_base_cfg):
        validated = validate_config(
            self._cfg(
                min_base_cfg,
                trl={
                    "max_staleness": 2,
                    "reward_num_workers": 4,
                    "epsilon": 0.2,
                    "epsilon_high": 0.28,
                    "scale_rewards": "none",
                },
            )
        )
        assert validated.trl.max_staleness == 2

    def test_cispo_takes_the_importance_sampling_options(self, min_base_cfg):
        validated = validate_config(
            self._cfg(
                min_base_cfg,
                trl={
                    "loss_type": "cispo",
                    "vllm_importance_sampling_correction": True,
                    "vllm_importance_sampling_mode": "token_mask",
                    "vllm_importance_sampling_cap": 5.0,
                },
            )
        )
        assert validated.trl.loss_type == "cispo"

    @pytest.mark.parametrize(
        "trl,extra,match",
        [
            ({"beta": 0.04}, {}, "no reference model"),
            ({"loss_type": "grpo"}, {}, "`loss_type` must be `dapo`"),
            (
                {"vllm_importance_sampling_cap": 5.0},
                {},
                "ignores `vllm_importance_sampling_cap`; .* set `loss_type: cispo`",
            ),
            (
                {"loss_type": "dapo", "vllm_importance_sampling_correction": True},
                {},
                "ignores `vllm_importance_sampling_correction`",
            ),
            (
                {"loss_type": "cispo", "epsilon_high": 0.28},
                {},
                "ignores `epsilon_high`; .*`vllm_importance_sampling_cap`",
            ),
            (
                {"loss_type": "cispo", "vllm_importance_sampling_correction": False},
                {},
                "correction: false",
            ),
            ({"num_iterations": 2}, {}, "num_iterations"),
            ({"scale_rewards": "batch"}, {}, "scale_rewards: batch"),
            ({"prefetch_depth": 2}, {}, "max_staleness"),
            ({"streaming_partial_batch": True}, {}, "data producer trainer"),
            ({"replay_buffer_size": 64}, {}, "replay_buffer_size"),
            (
                {
                    "loss_type": "cispo",
                    "vllm_importance_sampling_mode": "sequence_mask",
                },
                {},
                "token_truncate",
            ),
            ({"use_vllm": False}, {}, "use_vllm: true"),
            ({}, {"load_in_4bit": True, "adapter": "qlora"}, "vllm_lora_sync"),
            (
                {"vllm_importance_sampling_clip_min": 0.5},
                {},
                "bound `loss_type: score_centering` only",
            ),
            (
                {"loss_type": "cispo", "num_sampler_logprobs": 8},
                {},
                "`num_sampler_logprobs` is for `loss_type: score_centering` only",
            ),
        ],
    )
    def test_unsupported_settings_raise(self, min_base_cfg, trl, extra, match):
        with pytest.raises(ValueError, match=match):
            validate_config(self._cfg(min_base_cfg, trl=trl, **extra))

    def test_generation_groups_need_not_fill_a_step(self, min_base_cfg):
        """Async GRPO packs trained samples by token budget, apart from the groups
        the rollout worker generates, so the synchronous batch shape does not apply."""
        validated = validate_config(
            self._cfg(
                min_base_cfg,
                trl={"num_generations": 8, "token_budget": 4096},
                micro_batch_size=1,
                gradient_accumulation_steps=1,
            )
        )
        assert validated.trl.num_generations == 8

    def test_groups_still_need_two_generations(self, min_base_cfg):
        with pytest.raises(ValueError, match="num_generations >= 2"):
            validate_config(self._cfg(min_base_cfg, trl={"num_generations": 1}))

    @pytest.fixture
    def score_centering(self, monkeypatch):
        from axolotl.core.trainers.grpo import trl_compat

        monkeypatch.setattr(trl_compat, "SCORE_CENTERING", True)

    @pytest.mark.usefixtures("score_centering")
    def test_score_centering_takes_its_bounds(self, min_base_cfg):
        validated = validate_config(
            self._cfg(
                min_base_cfg,
                trl={
                    "loss_type": "score_centering",
                    "vllm_importance_sampling_clip_min": 0.25,
                    "vllm_importance_sampling_clip_max": 4.0,
                    "num_sampler_logprobs": 16,
                },
            )
        )
        assert validated.trl.loss_type == "score_centering"
        assert validated.trl.num_sampler_logprobs == 16

    @pytest.mark.usefixtures("score_centering")
    @pytest.mark.parametrize(
        "trl,match",
        [
            ({"epsilon_high": 0.28}, "ignores `epsilon_high`"),
            (
                {"vllm_importance_sampling_cap": 3.0},
                "ignores `vllm_importance_sampling_cap`",
            ),
        ],
    )
    def test_score_centering_rejects_other_losses_bounds(
        self, min_base_cfg, trl, match
    ):
        with pytest.raises(ValueError, match=match):
            validate_config(
                self._cfg(min_base_cfg, trl={"loss_type": "score_centering", **trl})
            )

    def test_score_centering_needs_trl_main(self, min_base_cfg, monkeypatch):
        from axolotl.core.trainers.grpo import trl_compat

        monkeypatch.setattr(trl_compat, "SCORE_CENTERING", False)
        with pytest.raises(ValueError, match="only TRL main has"):
            validate_config(
                self._cfg(min_base_cfg, trl={"loss_type": "score_centering"})
            )

    def test_lists_every_problem(self, min_base_cfg):
        with pytest.raises(ValueError) as error:
            validate_config(
                self._cfg(
                    min_base_cfg, trl={"beta": 0.04, "off_policy_mask_threshold": 0.5}
                )
            )
        assert "no reference model" in str(error.value)
        assert "off_policy_mask_threshold` applies to synchronous GRPO" in str(
            error.value
        )

    def test_decaying_schedule_without_max_steps_warns(self, min_base_cfg):
        cfg = self._cfg(min_base_cfg, lr_scheduler="cosine")
        cfg.max_steps = None
        with patch("axolotl.utils.schemas.validation.LOG.warning") as warning:
            validate_config(cfg)
        assert any("lr_scheduler: cosine" in c.args[0] for c in warning.call_args_list)

    @pytest.mark.parametrize(
        "trl,adapter,warns",
        [
            ({}, "lora", True),
            (
                {"loss_type": "dapo", "max_staleness": 3, "vllm_lora_sync": True},
                "qlora",
                True,
            ),
            ({"max_staleness": 2, "vllm_sync_interval": 2}, "lora", True),
            ({"max_staleness": 2}, "lora", False),
            ({"loss_type": "cispo"}, "lora", False),
            ({}, None, False),
        ],
    )
    def test_lora_with_stale_rollouts_recommends_cispo(
        self, min_base_cfg, trl, adapter, warns
    ):
        extra = {}
        if adapter:
            extra = {
                "adapter": adapter,
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_target_linear": True,
            }
            if adapter == "qlora":
                extra["load_in_4bit"] = True
        cfg = self._cfg(min_base_cfg, trl=trl, **extra)
        with patch("axolotl.utils.schemas.validation.LOG.warning") as warning:
            validate_config(cfg)
        recommended = [
            c for c in warning.call_args_list if "loss_type: cispo" in c.args[0]
        ]
        assert bool(recommended) == warns


class TestDataProducerRemoval:
    """The data producer trainer is gone; its options fail with their replacement."""

    def test_use_data_producer_maps_to_async_grpo(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            rl="grpo", max_steps=10, trl={"use_vllm": True, "use_data_producer": True}
        )
        with patch("axolotl.utils.schemas.validation.LOG.warning") as warning:
            validated = validate_config(cfg)
        assert validated.trl.async_prefetch and validated.trl.max_staleness == 0
        assert not validated.trl.use_data_producer
        assert any(
            "use_data_producer` is deprecated" in c.args[0]
            for c in warning.call_args_list
        )

    def test_use_data_producer_keeps_an_explicit_async_setup(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            rl="grpo",
            max_steps=10,
            trl={
                "use_vllm": True,
                "use_data_producer": True,
                "async_prefetch": True,
                "max_staleness": 2,
            },
        )
        assert validate_config(cfg).trl.max_staleness == 2

    def test_removed_options_name_their_replacement(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            rl="grpo",
            batch_flattening=True,
            trl={
                "use_vllm": True,
                "prefetch_depth": 2,
                "replay_buffer_size": 0,
                "reroll_start_fraction": 0.5,
            },
        )
        with pytest.raises(ValueError) as error:
            validate_config(cfg)
        message = str(error.value)
        assert "data producer trainer has been removed" in message
        assert "`trl.prefetch_depth`: `trl.max_staleness`" in message
        assert "`trl.replay_buffer_size`" in message
        assert "`trl.reroll_start_fraction`" in message
        assert "`batch_flattening`: `trl.token_budget`" in message
        assert "docs/grpo.qmd#sec-data-producer-migration" in message

    def test_batch_flattening_still_applies_outside_grpo(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            batch_flattening=True, micro_batch_size=2, flash_attention=True
        )
        assert validate_config(cfg).batch_flattening


class TestTrainableTokenWarning:
    def test_mapping_values_satisfy_fixed_token_check(self):
        data = {
            "fix_untrained_tokens": [200, 201],
            "peft_trainable_token_indices": {
                "encoder.embed_tokens": [200],
                "other_embedding": [201],
            },
        }
        with patch("axolotl.utils.schemas.config.LOG.warning_once") as warning:
            assert (
                AxolotlInputConfig.warn_peft_trainable_token_to_fix_untrained(data)
                is data
            )
        warning.assert_not_called()

    def test_missing_fixed_token_warns_for_mapping(self):
        data = {
            "fix_untrained_tokens": [200, 201],
            "peft_trainable_token_indices": {"encoder.embed_tokens": [200]},
        }
        with patch("axolotl.utils.schemas.config.LOG.warning_once") as warning:
            AxolotlInputConfig.warn_peft_trainable_token_to_fix_untrained(data)
        warning.assert_called_once()
        assert "Token 201" in warning.call_args.args[0]


class TestCheckpointMirror:
    """checkpoint_mirror copies saves from a local output_dir to a shared directory."""

    def test_defaults(self, min_base_cfg, tmp_path):
        cfg = min_base_cfg | {
            "output_dir": str(tmp_path / "nvme"),
            "checkpoint_mirror": {"dir": str(tmp_path / "nfs")},
        }
        mirror = validate_config(cfg).checkpoint_mirror
        assert mirror["delete"] == "mirror"
        assert mirror["save_total_limit"] is None
        assert mirror["max_pending"] == 2
        assert mirror["resume"] is True

    @pytest.mark.parametrize("inside", ["", "mirror"])
    def test_rejects_a_mirror_inside_output_dir(self, min_base_cfg, tmp_path, inside):
        out = tmp_path / "nvme"
        cfg = min_base_cfg | {
            "output_dir": str(out),
            "checkpoint_mirror": {"dir": str(out / inside)},
        }
        with pytest.raises(ValueError, match="outside output_dir"):
            validate_config(cfg)

    def test_mirror_limit_requires_keep(self, min_base_cfg, tmp_path):
        cfg = min_base_cfg | {
            "output_dir": str(tmp_path / "nvme"),
            "checkpoint_mirror": {
                "dir": str(tmp_path / "nfs"),
                "delete": "mirror",
                "save_total_limit": 3,
            },
        }
        with pytest.raises(ValueError, match="only applies with `delete: keep`"):
            validate_config(cfg)

        cfg["checkpoint_mirror"]["delete"] = "keep"
        assert validate_config(cfg).checkpoint_mirror["save_total_limit"] == 3


class TestCheckpointSaveRetry:
    def test_defaults(self, min_base_cfg):
        cfg = min_base_cfg | {"checkpoint_save_retry": {}}
        retry = validate_config(cfg).checkpoint_save_retry
        assert (retry["wait_seconds"], retry["retries"]) == (300, 1)

    def test_warns_when_the_wait_reaches_ddp_timeout(self, min_base_cfg, caplog):
        cfg = min_base_cfg | {
            "checkpoint_save_retry": {"wait_seconds": 600, "retries": 3},
            "ddp_timeout": 1800,
        }
        with caplog.at_level("WARNING"):
            validate_config(cfg)
        assert "Lower wait_seconds or raise ddp_timeout" in caplog.text

    def test_rejects_negative_wait(self, min_base_cfg):
        cfg = min_base_cfg | {"checkpoint_save_retry": {"wait_seconds": -1}}
        with pytest.raises(ValueError):
            validate_config(cfg)
