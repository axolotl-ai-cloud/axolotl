"""Compatibility with upstream Trainer loss scaling and Accelerate CP setup."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from accelerate import Accelerator, ParallelismConfig
from transformers import Trainer

pytest.importorskip("ringmaster")

from axolotl.integrations.context_parallel.trainer import (  # noqa: E402
    TrainerContextParallelContextManager,
    configure_trainer,
)


def _trainer(average=True):
    return SimpleNamespace(
        accelerator=SimpleNamespace(
            parallelism_config=ParallelismConfig(cp_size=2),
            gather=lambda count: torch.stack([count, count]),
        ),
        model_accepts_loss_kwargs=True,
        compute_loss_func=None,
        label_smoother=None,
        _loss_shifts_labels=True,
        args=SimpleNamespace(average_tokens_across_devices=average),
    )


@pytest.mark.parametrize("average, expected", [(True, 5), (False, 2.5)])
def test_accumulation_window_token_count(average, expected):
    trainer = _trainer(average)
    restore = configure_trainer(trainer)
    batches = [
        {"labels": torch.tensor([[1, 2, -100, 4]])},
        {"labels": torch.tensor([[1, -100, 3, 4, 5]])},
    ]
    assert trainer._get_num_items_in_batch(batches, "cpu").item() == expected
    restore()


def test_explicit_shift_labels_and_empty_targets():
    trainer = _trainer()
    configure_trainer(trainer)
    batch = {"labels": torch.tensor([[1, 2]]), "shift_labels": torch.tensor([[2, 3]])}
    assert trainer._get_num_items_in_batch([batch], "cpu").item() == 2
    batch["shift_labels"].fill_(-100)
    assert trainer._get_num_items_in_batch([batch], "cpu").item() > 0


def test_overrides_are_local_and_restore_existing_methods():
    accelerator_method = Accelerator._prepare_cp
    trainer_method = Trainer._prepare_context_parallel_inputs
    trainer = _trainer()
    existing = Mock()
    trainer.accelerator._prepare_cp = existing
    restore = configure_trainer(trainer)
    inputs = {"input_ids": torch.ones(1, 4)}
    context, prepared = trainer._prepare_context_parallel_inputs(None, inputs)
    with context():
        assert prepared is inputs
    assert trainer.accelerator._prepare_cp("model") == ("model",)
    assert Accelerator._prepare_cp is accelerator_method
    assert Trainer._prepare_context_parallel_inputs is trainer_method
    restore()
    assert trainer.accelerator._prepare_cp is existing
    assert not hasattr(trainer, "_get_num_items_in_batch")
    assert not hasattr(trainer.accelerator, "_cp_context")


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_accepts_loss_kwargs", False),
        ("compute_loss_func", Mock()),
        ("label_smoother", Mock()),
    ],
)
def test_unsupported_loss_fails_before_overrides(field, value):
    trainer = _trainer()
    setattr(trainer, field, value)
    with pytest.raises(ValueError, match="causal LM loss"):
        configure_trainer(trainer)
    assert not hasattr(trainer.accelerator, "_prepare_cp")


def test_grpo_retains_its_token_count():
    trainer = _trainer()
    trainer._get_num_items_in_batch = Mock()
    original = trainer._get_num_items_in_batch
    configure_trainer(trainer, gather_outputs=True)
    assert trainer._get_num_items_in_batch is original


@pytest.mark.parametrize("training", [True, False])
def test_sharding_preserves_trainer_denominator(monkeypatch, training):
    from ringmaster import sp_context

    monkeypatch.setattr(sp_context, "broadcast_batch", lambda *args: None)
    ctx = TrainerContextParallelContextManager([], None)
    ctx.cp_size = 2
    hook = ctx._make_pre_hook(["input_ids", "labels"])
    labels = torch.tensor([[1, 2, 3, 4, 5]])
    _, kwargs = hook(
        SimpleNamespace(training=training),
        (),
        {
            "input_ids": labels,
            "labels": labels,
            "num_items_in_batch": torch.tensor(13.5),
        },
    )
    assert kwargs["input_ids"].shape == (1, 3)
    assert kwargs["shift_labels"].tolist() == [[2, 3, 4]]
    assert kwargs["num_items_in_batch"].item() == 13.5
    assert ctx._local_valid is None


@pytest.mark.parametrize("rl", [None, "grpo", "gdpo", "ebft"])
def test_recurrent_wiring_visits_reference_model(monkeypatch, rl):
    import ringmaster as rm

    from axolotl.integrations.context_parallel import (
        ContextParallelConfig,
        ContextParallelPlugin,
        trainer as adapter,
    )

    cfg = SimpleNamespace(
        context_parallel=ContextParallelConfig(size=2, backend="ulysses"),
        attn_implementation="sdpa",
        rl=rl,
    )
    models = [
        SimpleNamespace(
            config=SimpleNamespace(num_key_value_heads=4),
            modules=lambda: [],
            set_attn_implementation=Mock(),
        )
        for _ in range(2)
    ]
    trainer = SimpleNamespace(
        model=models[0], ref_model=models[1], accelerator=SimpleNamespace()
    )
    plugin = ContextParallelPlugin()
    from ringmaster import mamba

    wiring = Mock(return_value=SimpleNamespace(restore=lambda: None))
    monkeypatch.setattr(mamba, "mamba2_mixers", lambda model: [model])
    monkeypatch.setattr(rm, "wire_recurrent_layers", wiring)
    monkeypatch.setattr(plugin, "_install_hooks", Mock())
    configure = Mock(return_value=lambda: None)
    monkeypatch.setattr(adapter, "configure_trainer", configure)
    monkeypatch.setattr(
        rm,
        "setup",
        lambda config, **kwargs: SimpleNamespace(
            config=config, cp_group=None, attn_implementation="ringmaster_ulysses"
        ),
    )
    plugin.post_trainer_create(cfg, trainer)
    configure.assert_called_once_with(trainer, gather_outputs=rl is not None)
    assert [call.args[0] for call in wiring.call_args_list] == models
    plugin.post_train_unload(cfg)


def test_forward_only_ring_rejected_before_setup(monkeypatch):
    import ringmaster as rm

    from axolotl.integrations.context_parallel import (
        ContextParallelConfig,
        ContextParallelPlugin,
    )

    setup = Mock()
    monkeypatch.setattr(rm, "setup", setup)
    cfg = SimpleNamespace(
        context_parallel=ContextParallelConfig(size=2, backend="ring"),
        attn_implementation="sdpa",
    )
    trainer = SimpleNamespace(
        model=SimpleNamespace(
            config=SimpleNamespace(num_key_value_heads=1), modules=lambda: []
        ),
        accelerator=SimpleNamespace(),
    )
    with pytest.raises(ValueError, match="forward-only"):
        ContextParallelPlugin().post_trainer_create(cfg, trainer)
    setup.assert_not_called()


@pytest.mark.parametrize("average, expected", [(True, 5), (False, 2.5)])
def test_token_count_with_tensor_and_context_parallelism(average, expected):
    trainer = _trainer(average)
    trainer.accelerator.parallelism_config = ParallelismConfig(cp_size=2, tp_size=2)
    trainer.accelerator.gather = lambda count: torch.stack([count] * 4)
    configure_trainer(trainer)
    batches = [{"labels": torch.tensor([[1, 2, 3, 4, 5, 6]])}]
    assert trainer._get_num_items_in_batch(batches, "cpu").item() == expected


def test_ulysses_rejects_degree_exceeding_tp_local_heads(monkeypatch):
    import ringmaster as rm

    from axolotl.integrations.context_parallel import (
        ContextParallelConfig,
        ContextParallelPlugin,
    )

    setup = Mock()
    monkeypatch.setattr(rm, "setup", setup)
    cfg = SimpleNamespace(
        context_parallel=ContextParallelConfig(size=4, backend="ulysses"),
        attn_implementation="sdpa",
    )
    trainer = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(num_key_value_heads=8)),
        accelerator=SimpleNamespace(
            parallelism_config=ParallelismConfig(cp_size=4, tp_size=4)
        ),
    )
    with pytest.raises(ValueError, match="num_kv_heads"):
        ContextParallelPlugin().post_trainer_create(cfg, trainer)
    setup.assert_not_called()
