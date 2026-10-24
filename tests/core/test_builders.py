"""Unit tests for axolotl.core.builders SFT and reward-model trainer builders."""

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch._dynamo.config as _dynamo_cfg
import torch._inductor.config as _inductor_cfg
from datasets import Dataset

from axolotl.core.builders import HFCausalTrainerBuilder, HFRLTrainerBuilder
from axolotl.core.builders.base import TrainerBuilderBase
from axolotl.core.trainers import AxolotlKTOTrainer, AxolotlTrainer
from axolotl.model_support import ModelStrategies
from axolotl.utils.collators import (
    BatchSamplerDataCollatorForSeq2Seq,
    DataCollatorForSeq2Seq,
    V2BatchSamplerDataCollatorForSeq2Seq,
)
from axolotl.utils.dict import DictDefault
from axolotl.utils.schemas.enums import INDUCTOR_COMPILE_OPTIONS_ALLOWLIST


def _gradient_checkpointing_kwargs(cfg):
    training_args_kwargs = {}
    TrainerBuilderBase._configure_gradient_checkpointing(
        SimpleNamespace(cfg=cfg), training_args_kwargs
    )
    return training_args_kwargs


def _save_and_eval_kwargs(cfg, *, eval_dataset=True, initial_kwargs=None):
    builder = SimpleNamespace(cfg=cfg, eval_dataset=eval_dataset)
    training_args_kwargs = dict(initial_kwargs or {})
    TrainerBuilderBase._configure_save_and_eval_strategy(builder, training_args_kwargs)
    return training_args_kwargs


class TestSaveAndEvalStrategy:
    def test_explicit_no_disables_initial_eval_even_with_eval_steps(self):
        kwargs = _save_and_eval_kwargs(
            SimpleNamespace(
                save_steps=None,
                save_strategy=None,
                save_total_limit=None,
                val_set_size=0,
                eval_strategy="no",
                eval_steps=5,
            ),
            initial_kwargs={"eval_on_start": True},
        )

        assert kwargs["eval_strategy"] == "no"
        assert kwargs["eval_on_start"] is False
        assert "eval_steps" not in kwargs

    @pytest.mark.parametrize(
        ("eval_strategy", "eval_steps", "expected_strategy"),
        [("epoch", None, "epoch"), ("steps", 5, "steps")],
    )
    def test_enabled_eval_keeps_initial_eval_behavior(
        self, eval_strategy, eval_steps, expected_strategy
    ):
        kwargs = _save_and_eval_kwargs(
            SimpleNamespace(
                save_steps=None,
                save_strategy=None,
                save_total_limit=None,
                val_set_size=0,
                eval_strategy=eval_strategy,
                eval_steps=eval_steps,
            )
        )

        assert kwargs["eval_strategy"] == expected_strategy
        assert kwargs["eval_on_start"] is True
        if eval_steps:
            assert kwargs["eval_steps"] == eval_steps

    @pytest.mark.parametrize(
        ("eval_strategy", "eval_steps"), [("epoch", None), ("steps", 5)]
    )
    def test_explicit_eval_on_start_false_skips_initial_eval(
        self, eval_strategy, eval_steps
    ):
        kwargs = _save_and_eval_kwargs(
            SimpleNamespace(
                save_steps=None,
                save_strategy=None,
                save_total_limit=None,
                val_set_size=0,
                eval_strategy=eval_strategy,
                eval_steps=eval_steps,
                eval_on_start=False,
            )
        )

        assert kwargs["eval_on_start"] is False

    def test_explicit_eval_on_start_true_is_kept(self):
        kwargs = _save_and_eval_kwargs(
            SimpleNamespace(
                save_steps=None,
                save_strategy=None,
                save_total_limit=None,
                val_set_size=0,
                eval_strategy=None,
                eval_steps=5,
                eval_on_start=True,
            )
        )

        assert kwargs["eval_on_start"] is True


class TestGradientCheckpointingConfig:
    def test_hidden_states_offload_uses_non_reentrant_trainer_path(self):
        training_args_kwargs = _gradient_checkpointing_kwargs(
            SimpleNamespace(
                layer_offloading=False,
                activation_offloading="hidden_states",
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        )

        assert training_args_kwargs["gradient_checkpointing"] is True
        assert training_args_kwargs["gradient_checkpointing_kwargs"] == {
            "use_reentrant": False
        }
        assert training_args_kwargs["activation_offloading"] == "hidden_states"

    def test_hidden_states_offload_with_reentrant_stays_on_model_loader_path(self):
        training_args_kwargs = _gradient_checkpointing_kwargs(
            SimpleNamespace(
                layer_offloading=False,
                activation_offloading="hidden_states",
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": True},
            )
        )

        assert training_args_kwargs["gradient_checkpointing"] is True
        assert training_args_kwargs["gradient_checkpointing_kwargs"] == {
            "use_reentrant": True
        }
        assert "activation_offloading" not in training_args_kwargs


def _reward_dataset():
    return Dataset.from_list(
        [
            {
                "chosen_ids": [1, 2, 3],
                "rejected_ids": [1, 4],
            }
        ]
    )


def _prm_dataset():
    return Dataset.from_list(
        [
            {
                "input_ids": [1, 2, 3],
                "labels": [-100, -100, 1],
            }
        ]
    )


class TestHFCausalTrainerBuilder:
    """
    TestCase class for SFT trainer builder
    """

    def test_training_arguments(self, sft_cfg, model, tokenizer):
        builder = HFCausalTrainerBuilder(sft_cfg, model, tokenizer)
        trainer = builder.build(100)
        training_arguments = trainer.args

        # Test common arguments
        assert training_arguments.per_device_train_batch_size == 2
        assert training_arguments.gradient_accumulation_steps == 1
        assert training_arguments.max_steps == 100

        assert training_arguments.learning_rate == 0.00005
        assert training_arguments.weight_decay == 0.01
        assert training_arguments.adam_beta1 == 0.91
        assert training_arguments.adam_beta2 == 0.99
        assert training_arguments.adam_epsilon == 0.00001
        assert training_arguments.max_grad_norm == 1.0

        assert training_arguments.lr_scheduler_type == "cosine"
        assert training_arguments.warmup_steps == 10
        assert training_arguments.cosine_min_lr_ratio == 0.1

        assert training_arguments.dataloader_num_workers == 1
        assert training_arguments.dataloader_pin_memory is True
        assert training_arguments.gradient_checkpointing is False

        # SFT specific
        assert training_arguments.sample_packing is False
        assert training_arguments.eval_sample_packing is False

    def test_checkpoint_save_retry_reaches_training_arguments(
        self, sft_cfg, model, tokenizer
    ):
        args = HFCausalTrainerBuilder(sft_cfg, model, tokenizer).build(100).args
        assert args.checkpoint_save_retry_wait is None

        sft_cfg["checkpoint_save_retry"] = {"wait_seconds": 120, "retries": 2}
        args = HFCausalTrainerBuilder(sft_cfg, model, tokenizer).build(100).args
        assert (args.checkpoint_save_retry_wait, args.checkpoint_save_retries) == (
            120,
            2,
        )

    def test_checkpoint_mirror_adds_callback(self, sft_cfg, model, tokenizer, tmp_path):
        from axolotl.core.checkpoint_mirror import CheckpointMirrorCallback

        sft_cfg["checkpoint_mirror"] = {
            "dir": str(tmp_path / "nfs"),
            "delete": "keep",
            "save_total_limit": 3,
            "max_pending": 1,
            "copy_threads": 2,
        }
        trainer = HFCausalTrainerBuilder(sft_cfg, model, tokenizer).build(100)
        (callback,) = [
            cb
            for cb in trainer.callback_handler.callbacks
            if isinstance(cb, CheckpointMirrorCallback)
        ]
        assert callback.mirror_dir == str(tmp_path / "nfs")
        assert (callback.delete, callback.save_total_limit) == ("keep", 3)
        assert (callback.max_pending, callback.copy_threads) == (1, 2)

    @pytest.mark.parametrize(
        "cfg_string",
        [
            "sft_cfg",
            "rm_cfg",
            "prm_cfg",
        ],
    )
    def test_builder_w_rm_trainers(self, request, cfg_string, model, tokenizer):
        cfg = request.getfixturevalue(cfg_string)
        builder = HFCausalTrainerBuilder(cfg, model, tokenizer)
        cfg["optimizer"] = "muon"

        if cfg_string == "rm_cfg":
            builder.train_dataset = _reward_dataset()
        elif cfg_string == "prm_cfg":
            builder.train_dataset = _prm_dataset()

        trainer = builder.build(100)

        assert trainer.optimizer_cls_and_kwargs is not None

        from axolotl.contribs.mit.muon import MuonOptimizerFactory
        from axolotl.contribs.mit.muon.muon import Muon

        optimizer_cls, optimizer_kwargs = trainer.optimizer_cls_and_kwargs
        assert optimizer_cls is MuonOptimizerFactory
        assert optimizer_kwargs["lr"] == 0.00005
        assert optimizer_kwargs["weight_decay"] == 0.01
        assert optimizer_kwargs["betas"] == (0.91, 0.99)
        assert optimizer_kwargs["eps"] == 0.00001

        # Ensure optimizer is created with correct class
        optim = trainer.create_optimizer()
        assert isinstance(optim, Muon)

    def test_scao_import_from_top_level_with_optim_args(
        self, sft_cfg, model, tokenizer
    ):
        cfg = sft_cfg.copy()
        cfg["optimizer"] = "scao"
        cfg["optim_args"] = {"precond_freq": 10, "k_max": 64}

        builder = HFCausalTrainerBuilder(cfg, model, tokenizer)
        training_args_kwargs = {
            "learning_rate": 0.00005,
            "weight_decay": 0.01,
            "adam_beta1": 0.998,
            "adam_beta2": 0.9,
            "adam_epsilon": 0.00001,
        }
        trainer_kwargs = {}

        class FakeSCAO:
            pass

        scao_module = types.ModuleType("scao")
        scao_module.SCAO = FakeSCAO

        with patch.dict(sys.modules, {"scao": scao_module}, clear=False):
            builder._configure_optimizer(training_args_kwargs, trainer_kwargs)

        optimizer_cls, optimizer_kwargs = trainer_kwargs["optimizer_cls_and_kwargs"]
        assert optimizer_cls is FakeSCAO
        assert optimizer_kwargs["lr"] == 0.00005
        assert optimizer_kwargs["weight_decay"] == 0.01
        assert optimizer_kwargs["betas"] == (0.998, 0.9)
        assert optimizer_kwargs["eps"] == 0.00001
        assert optimizer_kwargs["precond_freq"] == 10
        assert optimizer_kwargs["k_max"] == 64

    def test_polora_optimizer(self, sft_cfg, model, tokenizer):
        cfg = sft_cfg.copy()
        cfg["optimizer"] = "polora"
        cfg["optim_args"] = {"beta1": 0.95, "ns_steps": 4}

        builder = HFCausalTrainerBuilder(cfg, model, tokenizer)
        trainer = builder.build(100)

        from axolotl.utils.optimizers.polora import PoloraOptimizerFactory

        optimizer_cls, optimizer_kwargs = trainer.optimizer_cls_and_kwargs
        assert optimizer_cls is PoloraOptimizerFactory
        assert optimizer_kwargs["lr"] == 0.00005
        assert optimizer_kwargs["beta1"] == 0.95
        assert optimizer_kwargs["ns_steps"] == 4
        # polora takes no Adam betas/eps; the builder branch must not merge them in
        assert "betas" not in optimizer_kwargs
        assert "eps" not in optimizer_kwargs

    @pytest.mark.parametrize("weight_decay", [0.0, 0.05])
    def test_loraplus_optimizer_weight_decay(
        self, sft_cfg, model, tokenizer, weight_decay
    ):
        cfg = sft_cfg.copy()
        cfg["loraplus_lr_ratio"] = 16
        cfg["weight_decay"] = weight_decay

        builder = HFCausalTrainerBuilder(cfg, model, tokenizer)
        trainer = builder.build(100)
        assert trainer.args.weight_decay == weight_decay

        optim = trainer.create_optimizer()

        # loraplus zeroes its no-decay group regardless; every other group tracks the config
        decays = {
            group["weight_decay"] for group in optim.param_groups if group["params"]
        }
        assert decays <= {weight_decay, 0.0}
        assert weight_decay in decays

    def test_label_keep_prob(self, sft_cfg, model, tokenizer):
        cfg = sft_cfg.copy()
        cfg["label_keep_prob"] = {1: 0.25, 2: 0.0}
        cfg["label_keep_prob_normalize"] = "original"
        trainer = HFCausalTrainerBuilder(cfg, model, tokenizer).build(100)
        assert trainer.args.label_keep_prob == {1: 0.25, 2: 0.0}
        assert trainer.args.label_keep_prob_normalize == "original"
        assert trainer._label_dropout.token_ids.tolist() == [1, 2]

        cfg["label_keep_prob"] = {len(tokenizer): 0.5}
        with pytest.raises(ValueError, match="outside the tokenizer vocabulary"):
            HFCausalTrainerBuilder(cfg, model, tokenizer).build(100)

    def test_sinkgd_optimizer(self, sft_cfg, model, tokenizer):
        cfg = sft_cfg.copy()
        cfg["optimizer"] = "sinkgd"
        cfg["optim_args"] = {"sinkhorn_iters": 5, "sinkgd_lr_scale": 0.05}

        builder = HFCausalTrainerBuilder(cfg, model, tokenizer)
        trainer = builder.build(100)

        assert trainer.optimizer_cls_and_kwargs is not None

        from axolotl.utils.optimizers.sinkgd import SinkGD, SinkGDOptimizerFactory

        optimizer_cls, optimizer_kwargs = trainer.optimizer_cls_and_kwargs
        assert optimizer_cls is SinkGDOptimizerFactory
        assert optimizer_kwargs["lr"] == 0.00005
        assert optimizer_kwargs["weight_decay"] == 0.01
        assert optimizer_kwargs["sinkhorn_iters"] == 5
        assert optimizer_kwargs["sinkgd_lr_scale"] == 0.05

        # OptimizerMixin must resolve the factory into a concrete SinkGD instance
        optim = trainer.create_optimizer()
        assert isinstance(optim, SinkGD)
        # linear weight matrices must be in the stateless SR-Sinkhorn group
        assert any(group.get("use_sinkgd") for group in optim.param_groups)

    def test_sinkgd_optimizer_feature_a_optim_args(self, sft_cfg, model, tokenizer):
        """The Feature-A optim_args (spectral norm + width-aware 1/d_in) plumb through and
        construct a live SinkGD with the flags set."""
        cfg = sft_cfg.copy()
        cfg["optimizer"] = "sinkgd"
        cfg["optim_args"] = {
            "sinkhorn_iters": 5,
            "sinkgd_lr_scale": 0.05,
            "sinkgd_spectral_norm": True,
            "sinkgd_spectral_norm_iters": 2,
            "sinkgd_spectral_target": "unit",
            "sinkgd_base_width": 256,
        }

        from axolotl.utils.optimizers.sinkgd import SinkGD

        builder = HFCausalTrainerBuilder(cfg, model, tokenizer)
        trainer = builder.build(100)
        optim = trainer.create_optimizer()
        assert isinstance(optim, SinkGD)
        assert optim.sinkgd_spectral_norm is True
        assert optim.sinkgd_spectral_norm_iters == 2
        assert optim.sinkgd_spectral_target == "unit"
        assert optim.sinkgd_base_width == 256

    def test_sinkgd_optimizer_width_double_count_rejected(
        self, sft_cfg, model, tokenizer
    ):
        """base_width + spectral_target='muon' double-count width and must be rejected."""
        cfg = sft_cfg.copy()
        cfg["optimizer"] = "sinkgd"
        cfg["optim_args"] = {
            "sinkgd_spectral_norm": True,
            "sinkgd_spectral_target": "muon",
            "sinkgd_base_width": 256,
        }
        builder = HFCausalTrainerBuilder(cfg, model, tokenizer)
        trainer = builder.build(100)
        with pytest.raises(ValueError):
            trainer.create_optimizer()

    def test_sinkgd_md_sphere_selects_subclass(self, sft_cfg, model, tokenizer):
        """sinkgd_md_sphere=true builds the SinkGDMD (A+B) subclass; default builds SinkGD."""
        from axolotl.utils.optimizers.sinkgd import SinkGD, SinkGDMD

        cfg = sft_cfg.copy()
        cfg["optimizer"] = "sinkgd"
        cfg["optim_args"] = {"sinkgd_md_sphere": True}
        trainer = HFCausalTrainerBuilder(cfg, model, tokenizer).build(100)
        optim = trainer.create_optimizer()
        assert isinstance(optim, SinkGDMD)

        cfg2 = sft_cfg.copy()
        cfg2["optimizer"] = "sinkgd"
        cfg2["optim_args"] = {"sinkgd_lr_scale": 0.05}
        trainer2 = HFCausalTrainerBuilder(cfg2, model, tokenizer).build(100)
        optim2 = trainer2.create_optimizer()
        assert isinstance(optim2, SinkGD) and not isinstance(optim2, SinkGDMD)

    def test_sinkgd_fused_kernel_flag(self, sft_cfg, model, tokenizer):
        """sinkgd_fused_kernel plumbs through optim_args into the optimizer."""
        from axolotl.utils.optimizers.sinkgd import SinkGD

        cfg = sft_cfg.copy()
        cfg["optimizer"] = "sinkgd"
        cfg["optim_args"] = {"sinkgd_fused_kernel": True}
        trainer = HFCausalTrainerBuilder(cfg, model, tokenizer).build(100)
        optim = trainer.create_optimizer()
        assert isinstance(optim, SinkGD)
        assert optim.sinkgd_fused_kernel is True


class TestTrainerClsPlugin:
    """
    TestCase class for trainer builder with plugin
    """

    def test_trainer_cls_is_not_none_with_plugin(self, kto_cfg, model, tokenizer):
        """
        Test that the trainer cls is not none when a plugin returns no trainer cls

        Fixes #2693
        """
        plugin_manager = MagicMock()
        plugin_manager.get_trainer_cls.return_value = None
        with patch(
            "axolotl.core.builders.rl.PluginManager.get_instance",
            return_value=plugin_manager,
        ):
            builder = HFRLTrainerBuilder(kto_cfg, model, tokenizer)
            trainer_cls, trainer_cls_args = builder._get_trainer_cls({})

        assert trainer_cls is AxolotlKTOTrainer
        assert trainer_cls_args[0] is model


@pytest.fixture(name="inductor_config_snapshot")
def fixture_inductor_config_snapshot():
    keys = sorted(INDUCTOR_COMPILE_OPTIONS_ALLOWLIST)
    saved = {k: getattr(_inductor_cfg, k) for k in keys}
    try:
        yield saved
    finally:
        for k, v in saved.items():
            setattr(_inductor_cfg, k, v)


class TestApplyTorchCompileOptions:
    # Runtime apply path; schema-layer validation is tested in TestTorchCompileValidation.

    def test_allowlisted_flag_is_applied(self, inductor_config_snapshot):
        assert _inductor_cfg.coordinate_descent_tuning is False
        TrainerBuilderBase._apply_torch_compile_options(
            {"coordinate_descent_tuning": True}
        )
        assert _inductor_cfg.coordinate_descent_tuning is True

    def test_all_allowlisted_flags_are_attributes_on_inductor_config(
        self, inductor_config_snapshot
    ):
        for key in INDUCTOR_COMPILE_OPTIONS_ALLOWLIST:
            assert hasattr(_inductor_cfg, key), (
                f"torch._inductor.config has no attribute {key!r}; "
                f"INDUCTOR_COMPILE_OPTIONS_ALLOWLIST needs updating."
            )

    def test_dotted_flag_is_applied(self, inductor_config_snapshot):
        # ConfigModule supports dotted setattr natively (no path-walk needed).
        assert _inductor_cfg.triton.cudagraphs is False
        TrainerBuilderBase._apply_torch_compile_options({"triton.cudagraphs": True})
        assert _inductor_cfg.triton.cudagraphs is True

    def test_multiple_flags_are_applied(self, inductor_config_snapshot):
        TrainerBuilderBase._apply_torch_compile_options(
            {
                "coordinate_descent_tuning": True,
                "shape_padding": False,
                "epilogue_fusion": False,
            }
        )
        assert _inductor_cfg.coordinate_descent_tuning is True
        assert _inductor_cfg.shape_padding is False
        assert _inductor_cfg.epilogue_fusion is False

    def test_empty_dict_does_not_mutate_state(self, inductor_config_snapshot):
        TrainerBuilderBase._apply_torch_compile_options({})
        for k, v in inductor_config_snapshot.items():
            assert getattr(_inductor_cfg, k) == v

    @pytest.mark.parametrize(
        ("accelerator_config", "pin_memory", "expected"),
        [
            (None, None, True),
            (None, True, True),
            (None, False, False),
            ({"split_batches": False}, None, True),
            ({"non_blocking": False}, None, False),
            ({"non_blocking": True}, False, True),
        ],
    )
    def test_configure_accelerator_config_non_blocking(
        self, accelerator_config, pin_memory, expected
    ):
        builder = MagicMock(spec=TrainerBuilderBase)
        builder.cfg = DictDefault(
            accelerator_config=accelerator_config, dataloader_pin_memory=pin_memory
        )
        training_args_kwargs: dict = {}

        TrainerBuilderBase._configure_accelerator_config(builder, training_args_kwargs)

        config = training_args_kwargs["accelerator_config"]
        assert config.non_blocking is expected
        if accelerator_config:
            for key, value in accelerator_config.items():
                assert getattr(config, key) is value

    def test_configure_torch_compile_disables_dynamo_lru_under_checkpointing(self):
        builder = MagicMock(spec=TrainerBuilderBase)
        builder.cfg = DictDefault(torch_compile=True, gradient_checkpointing=True)
        with patch("torch._C._dynamo.eval_frame._set_lru_cache") as set_lru:
            TrainerBuilderBase._configure_torch_compile(builder, {})
        set_lru.assert_called_once_with(False)

    def test_configure_torch_compile_keeps_dynamo_lru_without_checkpointing(self):
        builder = MagicMock(spec=TrainerBuilderBase)
        builder.cfg = DictDefault(torch_compile=True, gradient_checkpointing=False)
        with patch("torch._C._dynamo.eval_frame._set_lru_cache") as set_lru:
            TrainerBuilderBase._configure_torch_compile(builder, {})
        set_lru.assert_not_called()

    def test_configure_torch_compile_raises_recompile_limit_for_regional(self):
        builder = MagicMock(spec=TrainerBuilderBase)
        builder.cfg = DictDefault(torch_compile=True, torch_compile_regional=True)
        before = _dynamo_cfg.recompile_limit
        try:
            TrainerBuilderBase._configure_torch_compile(builder, {})
            assert _dynamo_cfg.recompile_limit >= 64
        finally:
            _dynamo_cfg.recompile_limit = before

    def test_configure_torch_compile_invokes_apply_when_options_set(self):
        builder = MagicMock(spec=TrainerBuilderBase)
        builder.cfg = MagicMock()
        builder.cfg.torch_compile = True
        builder.cfg.torch_compile_backend = None
        builder.cfg.torch_compile_mode = None
        builder.cfg.torch_compile_options = {"coordinate_descent_tuning": True}
        training_args_kwargs: dict = {}

        TrainerBuilderBase._configure_torch_compile(builder, training_args_kwargs)

        builder._apply_torch_compile_options.assert_called_once_with(
            {"coordinate_descent_tuning": True}
        )

    def test_configure_torch_compile_skips_apply_when_options_unset(self):
        builder = MagicMock(spec=TrainerBuilderBase)
        builder.cfg = MagicMock()
        builder.cfg.torch_compile = True
        builder.cfg.torch_compile_backend = None
        builder.cfg.torch_compile_mode = None
        builder.cfg.torch_compile_options = None
        training_args_kwargs: dict = {}

        TrainerBuilderBase._configure_torch_compile(builder, training_args_kwargs)

        builder._apply_torch_compile_options.assert_not_called()


@pytest.mark.parametrize("is_eval", [False, True])
@pytest.mark.parametrize(
    ("attn_implementation", "expected"),
    [
        ("flash_attention_2", True),
        ("flash_attention_3", True),
        ("flash_attention_4", True),
        ("flash_attention_torch", True),
        ("kernels-community/flash-attn2", True),
        ("kernels-community/flash-attn3@v2:flash_attn_func", True),
        ("kernels-community/vllm-flash-attn3", True),
        ("kernels-community/flash-attn4", True),
        ("sdpa", False),
        ("eager", False),
        ("flex_attention", False),
        ("kernels-community/sage-attention", False),
        (None, False),
    ],
)
def test_packed_collator_flash_attention_metadata(
    attn_implementation, expected, is_eval
):
    builder = HFCausalTrainerBuilder(
        DictDefault(
            attn_implementation=attn_implementation,
            model_config_type="llama",
            torch_compile=False,
        ),
        model=None,
        tokenizer=MagicMock(),
    )
    training_args = SimpleNamespace(
        pretraining=False,
        sample_packing=not is_eval,
        eval_sample_packing=is_eval,
    )

    collator = builder.build_collator(training_args, is_eval=is_eval)

    assert type(collator) is V2BatchSamplerDataCollatorForSeq2Seq
    assert collator.emit_fa_varlen_kwargs is expected
    builder.tokenizer.padding_side = "right"
    builder.tokenizer.pad.side_effect = lambda features, **kwargs: {
        key: torch.tensor([feature[key] for feature in features]) for key in features[0]
    }
    batch = collator(
        [
            {"input_ids": [10, 11], "position_ids": [0, 1]},
            {"input_ids": [12], "position_ids": [0]},
        ]
    )
    assert ("cu_seq_lens_q" in batch) is expected
    if expected:
        assert batch["cu_seq_lens_q"].tolist() == [0, 2, 3]
        assert batch["max_length_q"] == 2


@pytest.mark.parametrize("is_eval", [False, True])
def test_packed_collator_flash_attention_metadata_streaming_without_dispatch(is_eval):
    builder = HFCausalTrainerBuilder(
        DictDefault(
            attn_implementation="flash_attention_2",
            model_config_type="llama",
            streaming=True,
            accelerator_config={"dispatch_batches": False},
        ),
        model=None,
        tokenizer=MagicMock(),
    )
    training_args = SimpleNamespace(
        pretraining=False, sample_packing=not is_eval, eval_sample_packing=is_eval
    )

    collator = builder.build_collator(training_args, is_eval=is_eval)

    assert collator.emit_fa_varlen_kwargs is True


@pytest.mark.parametrize("is_eval", [False, True])
@pytest.mark.parametrize(
    ("fa_varlen_kwargs_in_collator", "packing", "model_type", "collator_cls", "extra"),
    [
        (False, True, "llama", V2BatchSamplerDataCollatorForSeq2Seq, {}),
        (True, False, "llama", DataCollatorForSeq2Seq, {}),
        (True, True, "gpt2", BatchSamplerDataCollatorForSeq2Seq, {}),
        (
            True,
            True,
            "llama",
            V2BatchSamplerDataCollatorForSeq2Seq,
            {"accelerator_config": {"dispatch_batches": True}},
        ),
        # DeepSpeed Ulysses slices every batch tensor as [batch, seq]
        (
            True,
            True,
            "llama",
            V2BatchSamplerDataCollatorForSeq2Seq,
            {
                "deepspeed": "deepspeed_configs/zero3_bf16.json",
                "context_parallel_size": 2,
            },
        ),
        # accelerate dispatches iterable-dataset batches by slicing every tensor on dim 0
        (
            True,
            True,
            "llama",
            V2BatchSamplerDataCollatorForSeq2Seq,
            {"streaming": True},
        ),
    ],
)
def test_collator_flash_attention_metadata_requires_supported_packed(
    fa_varlen_kwargs_in_collator, packing, model_type, collator_cls, extra, is_eval
):
    builder = HFCausalTrainerBuilder(
        DictDefault(
            attn_implementation="flash_attention_2",
            model_config_type=model_type,
            fa_varlen_kwargs_in_collator=fa_varlen_kwargs_in_collator,
            **extra,
        ),
        model=None,
        tokenizer=MagicMock(),
    )
    training_args = SimpleNamespace(
        pretraining=False,
        sample_packing=not packing if is_eval else packing,
        eval_sample_packing=packing if is_eval else not packing,
    )

    collator = builder.build_collator(training_args, is_eval=is_eval)

    assert type(collator) is collator_cls
    assert collator.emit_fa_varlen_kwargs is False


class _ProfileTrainer(AxolotlTrainer):
    post_set_calls = 0

    @classmethod
    def requires_all_columns(cls, cfg):
        return True

    def post_set_axolotl_cfg(self):
        type(self).post_set_calls += 1


def _patch_profile(strategies):
    return patch(
        "axolotl.core.builders.causal.resolve_model_support",
        return_value=SimpleNamespace(strategies=strategies),
    )


def _patch_plugins(trainer_cls=None, collator=None):
    plugin_manager = MagicMock()
    plugin_manager.get_trainer_cls.return_value = trainer_cls
    plugin_manager.get_collator_cls_and_kwargs.return_value = collator
    return patch(
        "axolotl.core.builders.causal.PluginManager.get_instance",
        return_value=plugin_manager,
    )


class TestProfileStrategyFactories:
    def _builder(self, cfg=None):
        return HFCausalTrainerBuilder(
            cfg or DictDefault(model_config_type="llama"),
            model=None,
            tokenizer=MagicMock(),
        )

    def test_profile_trainer_factory_receives_cfg(self):
        builder = self._builder()
        factory = MagicMock(return_value=_ProfileTrainer)
        with _patch_profile(ModelStrategies(trainer_cls=factory)):
            assert builder._get_trainer_cls() is _ProfileTrainer
        factory.assert_called_once_with(builder.cfg)

    def test_profile_trainer_factory_none_falls_back(self):
        builder = self._builder()
        with _patch_profile(ModelStrategies(trainer_cls=lambda cfg: None)):
            assert builder._get_trainer_cls() is AxolotlTrainer

    def test_plugin_trainer_wins_over_profile(self):
        class PluginTrainer(AxolotlTrainer):
            pass

        builder = self._builder()
        factory = MagicMock(return_value=_ProfileTrainer)
        with (
            _patch_plugins(trainer_cls=PluginTrainer),
            _patch_profile(ModelStrategies(trainer_cls=factory)),
        ):
            assert builder._get_trainer_cls() is PluginTrainer
        factory.assert_not_called()

    @pytest.mark.parametrize("is_eval", [False, True])
    def test_profile_collator_factory_is_used(self, is_eval):
        builder = self._builder()
        sentinel = object()
        factory = MagicMock(return_value=sentinel)
        training_args = SimpleNamespace(
            pretraining=False, sample_packing=False, eval_sample_packing=False
        )
        with _patch_profile(ModelStrategies(collator_factory=factory)):
            collator = builder.build_collator(training_args, is_eval=is_eval)
        assert collator is sentinel
        factory.assert_called_once_with(builder.cfg, builder.tokenizer, is_eval)

    def test_profile_collator_factory_none_falls_back(self):
        builder = self._builder()
        training_args = SimpleNamespace(
            pretraining=False, sample_packing=False, eval_sample_packing=False
        )
        with _patch_profile(
            ModelStrategies(collator_factory=lambda cfg, tokenizer, is_eval: None)
        ):
            collator = builder.build_collator(training_args)
        assert type(collator) is DataCollatorForSeq2Seq

    def test_plugin_collator_wins_over_profile(self):
        class PluginCollator:
            def __init__(self, *args, **kwargs):
                pass

        builder = self._builder()
        factory = MagicMock(return_value=object())
        training_args = SimpleNamespace(
            pretraining=False, sample_packing=False, eval_sample_packing=False
        )
        with (
            _patch_plugins(collator=(PluginCollator, {})),
            _patch_profile(ModelStrategies(collator_factory=factory)),
        ):
            collator = builder.build_collator(training_args)
        assert isinstance(collator, PluginCollator)
        factory.assert_not_called()

    @pytest.mark.parametrize(
        ("user_value", "expected"), [(None, False), (True, True), (False, False)]
    )
    def test_trainer_requires_all_columns(
        self, sft_cfg, model, tokenizer, user_value, expected
    ):
        cfg = sft_cfg.copy()
        cfg.remove_unused_columns = user_value
        _ProfileTrainer.post_set_calls = 0
        with _patch_profile(ModelStrategies(trainer_cls=lambda cfg: _ProfileTrainer)):
            trainer = HFCausalTrainerBuilder(cfg, model, tokenizer).build(100)
        assert type(trainer) is _ProfileTrainer
        assert trainer.args.remove_unused_columns is expected
        assert _ProfileTrainer.post_set_calls == 1
