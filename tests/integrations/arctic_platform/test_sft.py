# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI

"""Unit tests for the Arctic Platform SFT integration.

Config / soft-dependency tests run without ``arctic_platform`` installed.
Tests that build ``ArcticSFTClientConfig`` or exercise the client path skip
when the package is missing.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from pydantic import ValidationError

from axolotl.integrations.arctic_platform.sft.args import ArcticSFTArgs, ArcticSFTConfig
from axolotl.integrations.arctic_platform.sft.deps import (
    require_arctic_platform,
    require_arctic_sft_client,
)
from axolotl.integrations.arctic_platform.sft.plugin import ArcticSFTPlugin
from axolotl.utils.dict import DictDefault

ARCTIC_PLATFORM_INSTALLED = importlib.util.find_spec("arctic_platform") is not None


# ---------------------------------------------------------------------------
# Soft dependency (always runs)
# ---------------------------------------------------------------------------


class TestSoftDependency:
    def test_require_arctic_platform_hint(self, monkeypatch):
        """Missing package → ImportError that tells the user how to pip install."""
        monkeypatch.setitem(sys.modules, "arctic_platform", None)
        with pytest.raises(ImportError, match=r"pip install arctic_platform"):
            require_arctic_platform()

    def test_require_arctic_sft_client_hint(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "arctic_platform", None)
        with pytest.raises(ImportError, match=r"pip install arctic_platform"):
            require_arctic_sft_client()


# ---------------------------------------------------------------------------
# Config schema (always runs — pure pydantic, no arctic_platform)
# ---------------------------------------------------------------------------


class TestArcticSFTConfig:
    def test_requires_training_gpus(self):
        with pytest.raises(ValidationError):
            ArcticSFTConfig()  # training_gpus is required

    def test_training_gpus_must_be_positive(self):
        with pytest.raises(ValidationError):
            ArcticSFTConfig(training_gpus=0)

    def test_defaults(self):
        cfg = ArcticSFTConfig(training_gpus=2)
        assert cfg.comm_protocol == "http"
        assert cfg.loss_fn == "sft"
        assert cfg.host == "localhost"
        assert cfg.port == 8000
        assert cfg.launch_local_server is False

    def test_loss_fn_literal(self):
        assert ArcticSFTConfig(training_gpus=1, loss_fn="sft_ce").loss_fn == "sft_ce"
        with pytest.raises(ValidationError):
            ArcticSFTConfig(training_gpus=1, loss_fn="grpo")

    def test_logits_optimization_defaults_and_literal(self):
        cfg = ArcticSFTConfig(training_gpus=1)
        assert cfg.logits_optimization == "none"
        assert cfg.logits_optimization_peak_mem_size_in_gib == 4
        assert ArcticSFTConfig(training_gpus=1, logits_optimization="memory").logits_optimization == "memory"
        with pytest.raises(ValidationError):
            ArcticSFTConfig(training_gpus=1, logits_optimization="tiled")

    def test_args_mixin(self):
        args = ArcticSFTArgs(arctic_sft=ArcticSFTConfig(training_gpus=2, port=8765))
        assert args.arctic_sft.port == 8765


# ---------------------------------------------------------------------------
# Plugin registration (always runs)
# ---------------------------------------------------------------------------


class TestArcticSFTPluginHooks:
    def test_get_input_args(self):
        assert (
            ArcticSFTPlugin().get_input_args()
            == "axolotl.integrations.arctic_platform.sft.args.ArcticSFTArgs"
        )

    def test_register_sets_remove_unused_columns(self):
        cfg: dict = {}
        ArcticSFTPlugin().register(cfg)
        assert cfg["remove_unused_columns"] is False

    def test_register_preserves_explicit_value(self):
        cfg = {"remove_unused_columns": True}
        ArcticSFTPlugin().register(cfg)
        assert cfg["remove_unused_columns"] is True

    def test_get_trainer_cls(self):
        from axolotl.integrations.arctic_platform.sft.trainer import ArcticSFTTrainer

        assert ArcticSFTPlugin().get_trainer_cls(DictDefault()) is ArcticSFTTrainer


# ---------------------------------------------------------------------------
# Client-config mapping + trainer wire batch (needs arctic_platform)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not ARCTIC_PLATFORM_INSTALLED, reason="arctic_platform package not installed"
)
class TestBuildClientConfig:
    def _cfg(self, **arctic_overrides):
        arctic = {
            "training_gpus": 2,
            "host": "localhost",
            "port": 8765,
            "launch_local_server": True,
            "server_cuda_visible_devices": "0,1",
            "checkpoint_path": "/tmp/ckpt",
            "loss_fn": "sft",
        }
        arctic.update(arctic_overrides)
        return DictDefault(
            {
                "base_model": "NousResearch/Llama-3.2-1B",
                "learning_rate": 1e-5,
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "adam_epsilon": 1e-8,
                "weight_decay": 0.0,
                "max_grad_norm": 1.0,
                "warmup_ratio": 0.0,
                "sequence_len": 256,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "num_epochs": 1,
                "seed": 7,
                "arctic_sft": ArcticSFTConfig(**arctic),
            }
        )

    def test_maps_top_level_knobs(self):
        cfg = self._cfg()
        client_cfg = ArcticSFTPlugin._build_client_config(cfg, cfg.arctic_sft)
        assert client_cfg.model_name == "NousResearch/Llama-3.2-1B"
        assert client_cfg.training_gpus == 2
        assert client_cfg.host == "localhost"
        assert client_cfg.port == 8765
        assert client_cfg.launch_local_server is True
        assert client_cfg.server_cuda_visible_devices == "0,1"
        assert client_cfg.seed == 7
        # Optimizer + clipping are folded into ds_config (DeepSpeed config-json);
        # there is no separate training_config on the client anymore.
        assert not hasattr(client_cfg, "training_config")
        assert client_cfg.ds_config["optimizer"]["type"] == "AdamW"
        assert client_cfg.ds_config["optimizer"]["params"]["lr"] == 1e-5
        # No torch_adam → DeepSpeed builds its default FusedAdam.
        assert "torch_adam" not in client_cfg.ds_config["optimizer"]["params"]
        assert client_cfg.ds_config["gradient_clipping"] == 1.0
        assert client_cfg.ds_config["optimizer"]["params"]["eps"] == 1e-8
        # Per-GPU micro batch = micro_batch_size / training_gpus (2 / 2 = 1).
        assert client_cfg.ds_config["train_micro_batch_size_per_gpu"] == 1
        # Global effective batch = per_gpu * gpus * gas == micro_bs * gas.
        assert client_cfg.ds_config["train_batch_size"] == 2
        assert client_cfg.ds_config["gradient_accumulation_steps"] == 1
        # DeepSpeed bf16 mixed precision on by default (mirrors axolotl bf16).
        assert client_cfg.ds_config["bf16"] == {"enabled": True}
        # H5: default attention matches DeepSpeedWorker (FA2, not sdpa).
        assert client_cfg.ds_worker_config["attn_implementation"] == "flash_attention_2"

    def test_attn_implementation_from_top_level(self):
        cfg = self._cfg()
        cfg.attn_implementation = "sdpa"
        client_cfg = ArcticSFTPlugin._build_client_config(cfg, cfg.arctic_sft)
        assert client_cfg.ds_worker_config["attn_implementation"] == "sdpa"

    def test_default_seed_when_missing(self):
        cfg = self._cfg()
        cfg.seed = None
        client_cfg = ArcticSFTPlugin._build_client_config(cfg, cfg.arctic_sft)
        assert client_cfg.seed == 42

    def test_model_name_override(self):
        cfg = self._cfg(model_name="other/model")
        client_cfg = ArcticSFTPlugin._build_client_config(cfg, cfg.arctic_sft)
        assert client_cfg.model_name == "other/model"

    def test_micro_batch_must_divide_across_gpus(self):
        cfg = self._cfg()
        cfg.micro_batch_size = 1  # not a multiple of training_gpus=2
        with pytest.raises(ValueError, match=r"micro_batch_size"):
            ArcticSFTPlugin._build_client_config(cfg, cfg.arctic_sft)

    def test_ds_micro_batch_is_per_gpu_after_shard(self):
        """Fixed: ``train_micro_batch_size_per_gpu`` == micro_batch_size / training_gpus.

        After HTTP sharding each rank sees ``micro_batch_size / training_gpus``
        samples per microbatch, and DeepSpeed's per-gpu knob must match that.
        """
        cfg = self._cfg()  # micro=2, gpus=2, gas=1
        client_cfg = ArcticSFTPlugin._build_client_config(cfg, cfg.arctic_sft)
        per_gpu = cfg.micro_batch_size // cfg.arctic_sft.training_gpus
        assert per_gpu == 1
        assert client_cfg.ds_config["train_micro_batch_size_per_gpu"] == per_gpu
        assert (
            client_cfg.ds_config["train_batch_size"]
            == per_gpu * cfg.arctic_sft.training_gpus * cfg.gradient_accumulation_steps
        )

    def test_constant_schedule_has_no_scheduler(self):
        # Default (constant LR, no warmup): DeepSpeed runs on the raw LR, so no
        # ``scheduler`` block is emitted.
        cfg = self._cfg()
        client_cfg = ArcticSFTPlugin._build_client_config(cfg, cfg.arctic_sft)
        assert "scheduler" not in client_cfg.ds_config

    def test_cosine_schedule_uses_max_steps_horizon(self):
        cfg = self._cfg()
        cfg.lr_scheduler = "cosine"
        cfg.warmup_ratio = 0.1
        cfg.num_epochs = 3
        cfg.max_steps = 10
        client_cfg = ArcticSFTPlugin._build_client_config(cfg, cfg.arctic_sft)
        sched = client_cfg.ds_config["scheduler"]
        assert sched["type"] == "WarmupCosineLR"
        # Explicit max_steps wins as the horizon; warmup = ratio * horizon.
        assert sched["params"]["total_num_steps"] == 10
        assert sched["params"]["warmup_num_steps"] == 1.0

    def test_cosine_schedule_deferred_without_max_steps(self):
        # Without max_steps the horizon is unknown at build time, so cosine
        # cannot be baked yet; the trainer re-applies it with the resolved step
        # count via ``_apply_scheduler``.
        cfg = self._cfg()
        cfg.lr_scheduler = "cosine"
        cfg.warmup_ratio = 0.1
        cfg.max_steps = None
        client_cfg = ArcticSFTPlugin._build_client_config(cfg, cfg.arctic_sft)
        assert "scheduler" not in client_cfg.ds_config
        # Trainer patch path: resolved horizon → WarmupCosineLR baked in.
        ArcticSFTPlugin._apply_scheduler(client_cfg.ds_config, cfg, 20)
        sched = client_cfg.ds_config["scheduler"]
        assert sched["type"] == "WarmupCosineLR"
        assert sched["params"]["total_num_steps"] == 20
        assert sched["params"]["warmup_num_steps"] == 2.0

    def test_warmup_steps_absolute_uses_warmuplr(self):
        cfg = self._cfg()
        cfg.warmup_steps = 4
        client_cfg = ArcticSFTPlugin._build_client_config(cfg, cfg.arctic_sft)
        sched = client_cfg.ds_config["scheduler"]
        assert sched["type"] == "WarmupLR"
        assert sched["params"]["warmup_num_steps"] == 4.0
        assert sched["params"]["warmup_max_lr"] == 1e-5

    def test_checkpoint_path_defaults_from_output_dir(self):
        cfg = self._cfg(checkpoint_path=None)
        cfg.output_dir = "/data-fast/run1"
        client_cfg = ArcticSFTPlugin._build_client_config(cfg, cfg.arctic_sft)
        assert client_cfg.checkpoint_path == "/data-fast/run1/arctic_sft_ckpt"

    def test_post_trainer_create_attaches_config(self):
        from axolotl.integrations.arctic_platform.sft.trainer import ArcticSFTTrainer

        cfg = self._cfg(loss_fn="sft_ce")
        real = object.__new__(ArcticSFTTrainer)
        ArcticSFTPlugin().post_trainer_create(cfg, real)
        assert real._arctic_loss_fn == "sft_ce"
        assert real._arctic_learning_rate == 1e-5
        assert real._arctic_client_config.training_gpus == 2

    def test_post_trainer_create_maps_logits_optimization(self):
        from axolotl.integrations.arctic_platform.sft.trainer import ArcticSFTTrainer

        cfg = self._cfg(loss_fn="sft_ce")
        cfg.arctic_sft.logits_optimization = "memory"
        cfg.arctic_sft.logits_optimization_peak_mem_size_in_gib = 6
        real = object.__new__(ArcticSFTTrainer)
        ArcticSFTPlugin().post_trainer_create(cfg, real)
        assert real._arctic_logits_optimization == "memory"
        assert real._arctic_logits_optimization_peak_mem_gib == 6


# ---------------------------------------------------------------------------
# Trainer loop edge cases (mocked client — no live server)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not ARCTIC_PLATFORM_INSTALLED, reason="arctic_platform package not installed"
)
class TestTrainerGasFlush:
    def test_trailing_partial_gas_batches_are_dropped_with_warning(self, monkeypatch):
        """Trailing partial group is dropped (fixed-gas server contract) but warned.

        The server's DeepSpeed engine splits each wire batch into exactly
        ``gradient_accumulation_steps`` microbatches and errors on a short shard,
        so a partial group can't be flushed — it's dropped (drop_last semantics),
        but the trainer must log a warning so the drop is not silent.
        """
        import axolotl.integrations.arctic_platform.sft.trainer as trainer_mod
        from axolotl.integrations.arctic_platform.sft.trainer import ArcticSFTTrainer

        warnings: list[str] = []
        monkeypatch.setattr(
            trainer_mod.LOG, "warning", lambda *a, **k: warnings.append(a[0] if a else "")
        )

        trainer = object.__new__(ArcticSFTTrainer)
        trainer._arctic_client_config = MagicMock()
        trainer._arctic_loss_fn = "sft"
        trainer._arctic_learning_rate = 1e-5
        trainer._client = None
        trainer._pad_token_id = 0

        # 5 batches, gas=2 → 2 full steps + 1 trailing flush = 3 steps.
        batches = [
            {
                "input_ids": torch.ones(1, 4, dtype=torch.long),
                "attention_mask": torch.ones(1, 4, dtype=torch.long),
                "labels": torch.ones(1, 4, dtype=torch.long),
            }
            for _ in range(5)
        ]

        class FakeDL:
            def __len__(self):
                return 5

            def __iter__(self):
                return iter(batches)

        class FakeClient:
            def __init__(self):
                self.fwd_calls = 0
                self.step_calls = 0

            def fwd_bwd(self, wire):
                self.fwd_calls += 1
                return {"metrics": {"loss": 1.0}, "avg_loss": 1.0}

            def step(self, learning_rate=None):
                self.step_calls += 1
                return {"metrics": {"grad_norm": 0.1}}

            def save_checkpoint(self, path=None, **kwargs):
                return {}

            def shutdown(self):
                pass

        client = FakeClient()
        trainer._client = client
        trainer.get_train_dataloader = lambda: FakeDL()
        trainer.args = SimpleNamespace(
            gradient_accumulation_steps=2,
            num_train_epochs=1,
            max_steps=-1,
            logging_steps=1,
            output_dir=".",
            save_total_limit=None,
            load_best_model_at_end=False,
        )
        trainer.state = SimpleNamespace()
        trainer.control = SimpleNamespace(
            should_training_stop=False,
            should_epoch_stop=False,
            should_evaluate=False,
            should_save=False,
            should_log=False,
        )
        trainer.callback_handler = MagicMock()
        trainer.callback_handler.on_train_begin.return_value = trainer.control
        trainer.callback_handler.on_epoch_begin.return_value = trainer.control
        trainer.callback_handler.on_step_begin.return_value = trainer.control
        trainer.callback_handler.on_step_end.return_value = trainer.control
        trainer.callback_handler.on_epoch_end.return_value = trainer.control
        trainer.callback_handler.on_train_end.return_value = trainer.control
        trainer.log = MagicMock()

        out = trainer.train()
        # 5 batches / gas=2 → 2 full steps; the trailing batch is dropped.
        assert client.fwd_calls == 2
        assert client.step_calls == 2
        assert out.global_step == 2
        # ...but not silently: a drop warning must have been logged.
        assert any("dropping the trailing" in w for w in warnings)


class _Control:
    """Stand-in for transformers ``TrainerControl`` with the flags the loop reads."""

    def __init__(self):
        self.should_training_stop = False
        self.should_epoch_stop = False
        self.should_evaluate = False
        self.should_save = False
        self.should_log = False


class _FlowCallbackHandler:
    """Mimics HF ``DefaultFlowCallback``: sets ``should_save`` / ``should_evaluate``
    on chosen global steps in ``on_step_end`` so we can drive the trainer's
    checkpoint / eval honoring without a live HF callback stack."""

    def __init__(self, control, *, save_on=(), eval_on=()):
        self.control = control
        self.save_on = set(save_on)
        self.eval_on = set(eval_on)
        self.on_save_calls = 0
        self.on_evaluate_calls = 0

    def on_train_begin(self, *a, **k):
        return self.control

    def on_epoch_begin(self, *a, **k):
        return self.control

    def on_step_begin(self, *a, **k):
        return self.control

    def on_step_end(self, args, state, control):
        self.control.should_save = state.global_step in self.save_on
        self.control.should_evaluate = state.global_step in self.eval_on
        return self.control

    def on_epoch_end(self, *a, **k):
        return self.control

    def on_train_end(self, *a, **k):
        return self.control

    def on_save(self, *a, **k):
        self.on_save_calls += 1
        return self.control

    def on_evaluate(self, *a, **k):
        self.on_evaluate_calls += 1
        return self.control


class _RecordingClient:
    def __init__(self, *, load_step: int = 0):
        self.fwd_calls = 0
        self.step_calls = 0
        self.save_calls = 0
        self.no_grad_calls = 0
        self.load_calls: list = []
        self.last_save_kwargs: dict = {}
        self._load_step = load_step

    def fwd_bwd(self, wire):
        self.fwd_calls += 1
        return {"metrics": {"loss": 1.0}, "avg_loss": 1.0}

    def step(self, learning_rate=None):
        self.step_calls += 1
        return {"metrics": {"grad_norm": 0.1}}

    def fwd_no_grad(self, wire):
        self.no_grad_calls += 1
        return {"metrics": {"loss": 2.0}, "avg_loss": 2.0}

    def save_checkpoint(self, path=None, **kwargs):
        self.save_calls += 1
        self.last_save_kwargs = {"path": path, **kwargs}
        return {}

    def load_checkpoint(self, path=None, **kwargs):
        self.load_calls.append({"path": path, **kwargs})
        return {"global_step": self._load_step}

    def shutdown(self):
        pass


@pytest.mark.skipif(
    not ARCTIC_PLATFORM_INSTALLED, reason="arctic_platform package not installed"
)
class TestTrainerCheckpointEvalResume:
    """Checkpoint / eval / resume honoring in the remote loop (mocked client)."""

    def _batches(self, n: int):
        return [
            {
                "input_ids": torch.ones(1, 4, dtype=torch.long),
                "attention_mask": torch.ones(1, 4, dtype=torch.long),
                "labels": torch.tensor([[-100, 5, 6, 7]]),  # 3 valid targets
            }
            for _ in range(n)
        ]

    def _make_trainer(self, tmp_path, client, handler, *, num_batches=4, grad_accum=1, num_epochs=1):
        from axolotl.integrations.arctic_platform.sft.trainer import ArcticSFTTrainer

        batches = self._batches(num_batches)

        class FakeDL:
            def __len__(self):
                return num_batches

            def __iter__(self):
                return iter(batches)

        trainer = object.__new__(ArcticSFTTrainer)
        trainer._arctic_client_config = MagicMock()
        trainer._arctic_loss_fn = "sft"
        trainer._arctic_logits_optimization = "none"
        trainer._arctic_logits_optimization_peak_mem_gib = 4
        trainer._arctic_learning_rate = 1e-5
        trainer._arctic_export_hf = False
        trainer._client = client
        trainer._pad_token_id = 0
        trainer.processing_class = SimpleNamespace(pad_token_id=0, eos_token_id=0)
        trainer.eval_dataset = None
        trainer.get_train_dataloader = lambda: FakeDL()
        trainer.args = SimpleNamespace(
            gradient_accumulation_steps=grad_accum,
            num_train_epochs=num_epochs,
            max_steps=-1,
            logging_steps=1,
            output_dir=str(tmp_path),
            save_total_limit=None,
            load_best_model_at_end=False,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
        from transformers.trainer_callback import TrainerState

        trainer.state = TrainerState()
        trainer.control = handler.control
        trainer.callback_handler = handler
        trainer.log = MagicMock()
        return trainer

    def test_periodic_save_fires_on_save_and_writes_state(self, tmp_path):
        control = _Control()
        handler = _FlowCallbackHandler(control, save_on={2})
        client = _RecordingClient()
        trainer = self._make_trainer(tmp_path, client, handler, num_batches=3)

        out = trainer.train()

        assert out.global_step == 3
        # Periodic save at step 2 + the end-of-run remote save => 2 remote saves.
        assert client.save_calls == 2
        assert handler.on_save_calls == 1
        # HF-layout local state written for the periodic checkpoint.
        assert os.path.isfile(os.path.join(tmp_path, "checkpoint-2", "trainer_state.json"))

    def test_eval_runs_forward_only_and_fires_on_evaluate(self, tmp_path):
        control = _Control()
        handler = _FlowCallbackHandler(control, eval_on={2})
        client = _RecordingClient()
        trainer = self._make_trainer(tmp_path, client, handler, num_batches=3)
        eval_batches = self._batches(2)
        trainer.eval_dataset = object()
        trainer.get_eval_dataloader = lambda eval_dataset=None: eval_batches

        trainer.train()

        assert client.no_grad_calls == 2  # one fwd_no_grad per eval batch
        assert client.fwd_calls == 3  # training steps unaffected
        assert handler.on_evaluate_calls == 1
        eval_logs = [c.args[0] for c in trainer.log.call_args_list if "eval_loss" in c.args[0]]
        assert eval_logs and eval_logs[0]["eval_loss"] == pytest.approx(2.0)

    def test_eval_skipped_without_eval_dataset(self, tmp_path):
        control = _Control()
        handler = _FlowCallbackHandler(control, eval_on={1, 2, 3})
        client = _RecordingClient()
        trainer = self._make_trainer(tmp_path, client, handler, num_batches=3)
        # eval_dataset stays None → evaluate() must never run.

        trainer.train()

        assert client.no_grad_calls == 0
        assert handler.on_evaluate_calls == 0

    def test_resume_restores_step_and_skips_consumed_groups(self, tmp_path):
        control = _Control()
        handler = _FlowCallbackHandler(control)
        client = _RecordingClient(load_step=2)
        trainer = self._make_trainer(tmp_path, client, handler, num_batches=5)

        out = trainer.train(resume_from_checkpoint="/ckpt/dir")

        assert client.load_calls == [{"path": "/ckpt/dir", "step": None}]
        # Resumed at step 2; only the remaining 3 groups train.
        assert client.fwd_calls == 3
        assert out.global_step == 5

    def test_resume_with_no_server_checkpoint_does_not_skip(self, tmp_path):
        # Server found nothing (global_step=0) → must NOT skip any groups even
        # though resume was requested, or we'd desync a fresh engine.
        control = _Control()
        handler = _FlowCallbackHandler(control)
        client = _RecordingClient(load_step=0)
        trainer = self._make_trainer(tmp_path, client, handler, num_batches=3)

        out = trainer.train(resume_from_checkpoint="/ckpt/dir")

        assert client.load_calls == [{"path": "/ckpt/dir", "step": None}]
        assert client.fwd_calls == 3  # nothing skipped
        assert out.global_step == 3

    def test_eval_handles_zero_token_batches(self, tmp_path):
        control = _Control()
        handler = _FlowCallbackHandler(control, eval_on={1})
        client = _RecordingClient()
        trainer = self._make_trainer(tmp_path, client, handler, num_batches=1)
        masked = [
            {
                "input_ids": torch.ones(1, 4, dtype=torch.long),
                "attention_mask": torch.ones(1, 4, dtype=torch.long),
                "labels": torch.full((1, 4), -100),  # no valid targets
            }
        ]
        trainer.eval_dataset = object()
        trainer.get_eval_dataloader = lambda eval_dataset=None: masked

        trainer.train()

        assert client.no_grad_calls == 1
        eval_logs = [c.args[0] for c in trainer.log.call_args_list if "eval_loss" in c.args[0]]
        assert eval_logs and eval_logs[0]["eval_loss"] == 0.0

    def test_no_resume_does_not_load(self, tmp_path):
        control = _Control()
        handler = _FlowCallbackHandler(control)
        client = _RecordingClient(load_step=2)
        trainer = self._make_trainer(tmp_path, client, handler, num_batches=3)

        out = trainer.train()

        assert client.load_calls == []  # no resume requested
        assert client.fwd_calls == 3
        assert out.global_step == 3

    def test_resume_passes_explicit_path(self, tmp_path):
        control = _Control()
        handler = _FlowCallbackHandler(control)
        client = _RecordingClient(load_step=1)
        trainer = self._make_trainer(tmp_path, client, handler, num_batches=4)

        trainer.train(resume_from_checkpoint="/ckpt/dir")

        assert client.load_calls == [{"path": "/ckpt/dir", "step": None}]
        assert client.fwd_calls == 3  # 4 batches, skip 1 consumed group


@pytest.mark.skipif(
    not ARCTIC_PLATFORM_INSTALLED, reason="arctic_platform package not installed"
)
class TestWireBatch:
    def test_gas_list_keeps_per_microbatch_lengths(self):
        """H3: GAS microbatches stay as a list — no cross-GAS re-pad/concat."""
        from axolotl.integrations.arctic_platform.sft.trainer import ArcticSFTTrainer

        trainer = object.__new__(ArcticSFTTrainer)
        trainer._arctic_loss_fn = "sft"
        trainer._pad_token_id = 0
        trainer.processing_class = SimpleNamespace(pad_token_id=0, eos_token_id=0)

        b0 = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "labels": torch.tensor([[-100, 2, 3]]),
        }
        b1 = {
            "input_ids": torch.tensor([[4, 5]]),
            "attention_mask": torch.tensor([[1, 1]]),
            "labels": torch.tensor([[-100, 5]]),
        }
        wire = trainer._build_wire_batch([b0, b1])
        assert wire["processing"] == {"loss_fn": "sft"}
        assert wire["meta"]["pad_token_id"] == 0
        assert wire["meta"]["gas_microbatches"] is True
        assert isinstance(wire["batch"], list)
        assert len(wire["batch"]) == 2
        assert wire["batch"][0]["input_ids"].shape == (1, 3)
        assert wire["batch"][1]["input_ids"].shape == (1, 2)  # not padded to 3
        assert torch.equal(wire["batch"][1]["labels"], torch.tensor([[-100, 5]]))
        assert not wire["batch"][0]["input_ids"].is_cuda

    def test_packed_forwards_position_ids_no_synthetic_mask(self):
        from axolotl.integrations.arctic_platform.sft.trainer import ArcticSFTTrainer

        trainer = object.__new__(ArcticSFTTrainer)
        trainer._arctic_loss_fn = "sft"
        trainer._pad_token_id = 0
        trainer.processing_class = SimpleNamespace(pad_token_id=0, eos_token_id=0)

        # Axolotl FA2 packing drops attention_mask and resets position_ids.
        b0 = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "labels": torch.tensor([[-100, -100, 3, 4]]),
            "position_ids": torch.tensor([[0, 1, 0, 1]]),
        }
        b1 = {
            "input_ids": torch.tensor([[5, 6]]),
            "labels": torch.tensor([[-100, 6]]),
            "position_ids": torch.tensor([[0, 1]]),
        }
        wire = trainer._build_wire_batch([b0, b1])
        assert wire["meta"]["sample_packing"] is True
        assert wire["meta"]["gas_microbatches"] is True
        assert isinstance(wire["batch"], list)
        assert "attention_mask" not in wire["batch"][0]
        assert "attention_mask" not in wire["batch"][1]
        assert torch.equal(wire["batch"][0]["position_ids"], torch.tensor([[0, 1, 0, 1]]))
        # Shorter mb keeps its own length (no continuing-arange cross-GAS pad).
        assert torch.equal(wire["batch"][1]["position_ids"], torch.tensor([[0, 1]]))
        assert wire["batch"][1]["input_ids"].shape == (1, 2)

    def test_compute_loss_is_disabled(self):
        from axolotl.integrations.arctic_platform.sft.trainer import ArcticSFTTrainer

        trainer = object.__new__(ArcticSFTTrainer)
        with pytest.raises(NotImplementedError, match="remote server"):
            trainer.compute_loss(None, {})

    def _wire_trainer(self, *, loss_fn="sft_ce", logits_optimization="none", peak_mem=4):
        from axolotl.integrations.arctic_platform.sft.trainer import ArcticSFTTrainer

        trainer = object.__new__(ArcticSFTTrainer)
        trainer._arctic_loss_fn = loss_fn
        trainer._arctic_logits_optimization = logits_optimization
        trainer._arctic_logits_optimization_peak_mem_gib = peak_mem
        trainer._pad_token_id = 0
        trainer.processing_class = SimpleNamespace(pad_token_id=0, eos_token_id=0)
        return trainer

    def _one_batch(self):
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "labels": torch.tensor([[-100, 2, 3]]),
        }

    def test_logits_optimization_absent_when_none(self):
        trainer = self._wire_trainer(logits_optimization="none")
        wire = trainer._build_wire_batch([self._one_batch()])
        assert wire["processing"] == {"loss_fn": "sft_ce"}
        assert "config" not in wire["processing"]

    def test_logits_optimization_on_wire_when_set(self):
        trainer = self._wire_trainer(logits_optimization="memory", peak_mem=6)
        wire = trainer._build_wire_batch([self._one_batch()])
        cfg = wire["processing"]["config"]
        assert cfg["logits_optimization"] == "memory"
        assert cfg["logits_optimization_peak_mem_size_in_gib"] == 6

