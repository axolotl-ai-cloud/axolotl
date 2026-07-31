"""CPU micro-loop: the ternary plugin path end to end, plus the plugin machinery."""

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from axolotl.integrations.base import PluginManager
from axolotl.integrations.ternary import TernaryPlugin
from axolotl.integrations.ternary.args import (
    TernaryDistillConfig,
    resolve_ternary_config,
)
from axolotl.integrations.ternary.callbacks import (
    LambdaScheduleCallback,
    TernaryMonitorCallback,
    WdAnnealCallback,
)
from axolotl.integrations.ternary.modules import TernaryLinear, iter_ternary_modules
from axolotl.integrations.ternary.quant import (
    absmean_scale,
    pack_codes,
    ternary_codes,
)
from axolotl.integrations.ternary.swap import MANIFEST_FILENAME
from axolotl.utils.dict import DictDefault

from tests.conftest import capture_axolotl_warnings

PLUGIN_PATH = "axolotl.integrations.ternary.TernaryPlugin"
STEPS = 10
WARMUP = 5
FAMILIES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}
MINIMAL_CONFIG = {
    "base_model": "HuggingFaceTB/SmolLM2-135M",
    "datasets": [{"path": "mhenrichsen/alpaca_2k_test", "type": "alpaca"}],
    "micro_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "learning_rate": 1e-4,
}


def _tiny_llama() -> LlamaForCausalLM:
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    return LlamaForCausalLM(config)


def _batch() -> dict[str, torch.Tensor]:
    ids = torch.randint(0, 128, (2, 16), generator=torch.Generator().manual_seed(1))
    return {"input_ids": ids, "labels": ids}


def _plugin_cfg(output_dir: Path, **ternary) -> DictDefault:
    return DictDefault(
        {
            "ternary": {
                "lambda_warmup_steps": WARMUP,
                "log_code_flip_every": 1,
                **ternary,
            },
            "output_dir": str(output_dir),
            "weight_decay": 0.1,
        }
    )


@pytest.fixture(name="trained", scope="module")
def fixture_trained(tmp_path_factory) -> SimpleNamespace:
    """Convert a tiny llama through the plugin and heal it for `STEPS` optimizer steps."""
    output_dir = tmp_path_factory.mktemp("ternary_micro_loop")
    model = _tiny_llama()
    cfg = _plugin_cfg(output_dir)
    plugin = TernaryPlugin()

    plugin.post_model_build(cfg, model)
    callbacks = plugin.add_callbacks_pre_trainer(cfg, model)
    lambda_cb = next(cb for cb in callbacks if isinstance(cb, LambdaScheduleCallback))
    wd_cb = next(cb for cb in callbacks if isinstance(cb, WdAnnealCallback))
    monitor_cb = next(
        cb
        for cb in plugin.add_callbacks_post_trainer(cfg, SimpleNamespace(model=model))
        if isinstance(cb, TernaryMonitorCallback)
    )

    latent_ids = {
        id(param)
        for _, module in iter_ternary_modules(model)
        for param in module.parameters()
    }
    latents = [p for p in model.parameters() if id(p) in latent_ids]
    others = [p for p in model.parameters() if id(p) not in latent_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": latents, "weight_decay": 0.1},
            {"params": others, "weight_decay": 0.1},
        ],
        lr=1e-3,
    )
    # decay to zero so the ternary codes are expected to settle within the window
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: max(0.0, 1.0 - step / STEPS)
    )
    args = TrainingArguments(output_dir=str(output_dir), report_to=[])
    state = TrainerState(global_step=0, max_steps=STEPS)
    control = TrainerControl()

    initial = {
        name: module.weight.detach().clone()
        for name, module in iter_ternary_modules(model)
    }
    batch = _batch()
    lambdas: list[float] = []
    losses: list[float] = []
    logs: list[dict[str, float]] = []

    lambda_cb.on_train_begin(args, state, control, model=model)
    for _ in range(STEPS):
        lambda_cb.on_step_begin(args, state, control, model=model)
        wd_cb.on_step_begin(args, state, control, optimizer=optimizer, model=model)
        lambdas.append(next(iter_ternary_modules(model))[1].lambda_)

        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        state.global_step += 1

        monitor_cb.on_step_end(args, state, control, model=model)
        step_logs = {"loss": loss.detach().item()}
        monitor_cb.on_log(args, state, control, logs=step_logs, model=model)
        losses.append(step_logs["loss"])
        logs.append(step_logs)

    model.eval()
    with torch.no_grad():
        logits = model(**batch).logits.clone()
    codes = {
        name: module.code_snapshot().clone()
        for name, module in iter_ternary_modules(model)
    }
    return SimpleNamespace(
        model=model,
        plugin=plugin,
        cfg=cfg,
        optimizer=optimizer,
        output_dir=output_dir,
        initial=initial,
        lambdas=lambdas,
        losses=losses,
        logs=logs,
        logits=logits,
        codes=codes,
        batch=batch,
    )


@pytest.fixture(name="baked", scope="module")
def fixture_baked(trained: SimpleNamespace) -> SimpleNamespace:
    """Run axolotl's post-training module hook, the way `train.py` does at save time."""
    for name, module in trained.model.named_modules():
        if hasattr(module, "_post_training"):
            module._post_training(trained.model, name)
    return trained


@pytest.fixture(name="plugin_manager")
def fixture_plugin_manager():
    """Isolate the `PluginManager` singleton so registration does not leak."""
    manager = PluginManager.get_instance()
    saved = dict(manager.plugins)
    manager.plugins.clear()
    yield manager
    manager.plugins.clear()
    manager.plugins.update(saved)


def test_swap_ran_through_the_plugin(trained):
    assert trained.plugin.manifest is not None
    assert len(trained.plugin.manifest.entries) == 2 * len(FAMILIES)
    assert len(list(iter_ternary_modules(trained.model))) == 2 * len(FAMILIES)
    assert isinstance(trained.model.lm_head, nn.Linear)
    assert not isinstance(trained.model.lm_head, TernaryLinear)


def test_lambda_warms_up_and_reaches_one(trained):
    assert trained.lambdas == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8] + [1.0] * 5)
    assert all(
        module.lambda_ == 1.0 for _, module in iter_ternary_modules(trained.model)
    )


def test_losses_are_finite_and_decrease(trained):
    assert len(trained.losses) == STEPS
    assert all(math.isfinite(loss) for loss in trained.losses)
    assert trained.losses[-1] < trained.losses[0]


def test_latent_weights_receive_gradients(trained):
    for name, module in iter_ternary_modules(trained.model):
        assert not torch.equal(module.weight.detach(), trained.initial[name]), name


def test_codes_flip_early_then_settle(trained):
    flips = [
        log["ternary/flip_rate"] for log in trained.logs if "ternary/flip_rate" in log
    ]

    # the first monitored step has no previous snapshot to compare against
    assert len(flips) == STEPS - 1
    assert all(0.0 <= flip <= 1.0 for flip in flips)
    assert flips[0] > 0.0
    assert flips[0] > flips[-1]
    half = len(flips) // 2
    assert sum(flips[:half]) > sum(flips[half:])


def test_monitor_metrics_reach_the_logs(trained):
    last = trained.logs[-1]

    assert {f"ternary/flip_rate/{family}" for family in FAMILIES} <= set(last)
    assert {f"ternary/zero_frac/{family}" for family in FAMILIES} <= set(last)
    for log in trained.logs:
        assert 0.2 < log["ternary/zero_frac"] < 0.45


def test_weight_decay_annealed_for_the_latent_group_only(trained):
    latent_group, other_group = trained.optimizer.param_groups

    assert latent_group["weight_decay"] == 0.0
    assert other_group["weight_decay"] == 0.1


def test_bake_makes_every_swapped_weight_exactly_ternary(baked):
    for name, module in iter_ternary_modules(baked.model):
        weight = module.weight.detach()
        scale = weight.abs().max()

        assert module.baked is True
        assert module.scale is None
        assert torch.equal(
            weight.abs().unique(), torch.stack([torch.zeros_like(scale), scale])
        ), name
        # the dequant scale is f16-rounded so every packer re-derives the same codes
        assert scale.item() == scale.to(torch.float16).to(scale.dtype).item(), name
        assert torch.equal(module.code_snapshot(), baked.codes[name]), name


def test_bake_preserves_the_forward(baked):
    with torch.no_grad():
        logits = baked.model(**baked.batch).logits

    assert torch.equal(logits, baked.logits)


def test_bake_is_idempotent(baked):
    before = {
        name: module.weight.detach().clone()
        for name, module in iter_ternary_modules(baked.model)
    }

    for name, module in baked.model.named_modules():
        if hasattr(module, "_post_training"):
            module._post_training(baked.model, name)

    for name, module in iter_ternary_modules(baked.model):
        assert torch.equal(module.weight.detach(), before[name]), name


def test_master_roundtrips_through_save_pretrained(baked, tmp_path):
    expected = {
        name: module.weight.detach().clone()
        for name, module in iter_ternary_modules(baked.model)
    }

    baked.model.save_pretrained(tmp_path)
    loaded = LlamaForCausalLM.from_pretrained(tmp_path)

    assert set(loaded.state_dict()) == set(baked.model.state_dict())
    for name, weight in expected.items():
        module = loaded.get_submodule(name)
        assert isinstance(module, nn.Linear)
        assert not isinstance(module, TernaryLinear)
        assert torch.equal(module.weight.detach(), weight), name
        codes = ternary_codes(
            module.weight.detach(), absmean_scale(module.weight.detach())
        )
        assert torch.equal(pack_codes(codes), baked.codes[name]), name


def test_swap_manifest_written_to_output_dir(trained):
    assert (trained.output_dir / MANIFEST_FILENAME).is_file()


class _LogSpy(TrainerCallback):
    """Records the trainer log dicts as the reporting integrations see them."""

    def __init__(self):
        self.logs: list[dict[str, float]] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.logs.append(dict(logs or {}))


class _TokenDataset(Dataset):
    """Fixed random token ids, labels equal to inputs."""

    def __init__(self, size: int = 16, length: int = 16):
        self.ids = torch.randint(
            0, 128, (size, length), generator=torch.Generator().manual_seed(3)
        )

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"input_ids": self.ids[index], "labels": self.ids[index]}


class _MetricStoreTrainer(Trainer):
    """Mirrors `AxolotlTrainer`'s log seam: stored metrics merge before dispatch."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stored: dict[str, float] = {}

    def store_metrics(self, metrics, train_eval="train"):
        self._stored.update(metrics)

    def log(self, logs, start_time=None):
        logs.update(self._stored)
        self._stored = {}
        return super().log(logs, start_time)


def test_hf_trainer_drives_the_plugin_callbacks(tmp_path):
    model = _tiny_llama()
    cfg = _plugin_cfg(tmp_path, log_code_flip_every=2)
    plugin = TernaryPlugin()
    plugin.post_model_build(cfg, model)
    # registered FIRST, the way transformers orders the reporting integrations ahead
    # of the plugin callbacks — the metrics must already be in `logs` by then
    spy = _LogSpy()

    trainer = _MetricStoreTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_path),
            max_steps=STEPS,
            per_device_train_batch_size=2,
            learning_rate=1e-3,
            weight_decay=0.1,
            lr_scheduler_type="linear",
            logging_steps=1,
            save_strategy="no",
            use_cpu=True,
            disable_tqdm=True,
            report_to=[],
        ),
        train_dataset=_TokenDataset(),
        callbacks=[spy, *plugin.add_callbacks_pre_trainer(cfg, model)],
    )
    for callback in plugin.add_callbacks_post_trainer(cfg, trainer):
        trainer.add_callback(callback)
    trainer.train()

    monitored = [log for log in spy.logs if "ternary/zero_frac" in log]
    assert len(monitored) == STEPS // 2
    assert monitored[-1]["ternary/flip_rate"] < monitored[1]["ternary/flip_rate"]
    assert {f"ternary/zero_frac/{family}" for family in FAMILIES} <= set(monitored[-1])
    # and they survive into the history the trackers replay from
    assert [log for log in trainer.state.log_history if "ternary/zero_frac" in log]
    assert all(
        math.isfinite(log["loss"]) for log in trainer.state.log_history if "loss" in log
    )
    assert all(module.lambda_ == 1.0 for _, module in iter_ternary_modules(model))
    assert all(group["weight_decay"] == 0.0 for group in trainer.optimizer.param_groups)


def test_plugin_manager_loads_the_plugin(plugin_manager):
    plugin_manager.register(PLUGIN_PATH)

    assert isinstance(plugin_manager.plugins[PLUGIN_PATH], TernaryPlugin)
    assert plugin_manager.get_input_args() == [
        "axolotl.integrations.ternary.args.TernaryArgs"
    ]
    assert plugin_manager.get_training_args_mixin() == [
        "axolotl.integrations.ternary.distill.TernaryDistillTrainingArgsMixin"
    ]


def test_validate_config_merges_the_ternary_block(plugin_manager):
    from axolotl.cli.config import prepare_plugins
    from axolotl.utils.config import validate_config

    cfg = DictDefault(
        {
            **MINIMAL_CONFIG,
            "plugins": [PLUGIN_PATH],
            "save_only_model": True,
            "ternary": {
                "lambda_warmup_steps": 0.1,
                "activation_bits": None,
                "export": {"formats": ["master_bf16"]},
            },
        }
    )

    prepare_plugins(cfg)
    validated = validate_config(cfg)
    ternary_cfg = resolve_ternary_config(validated)

    assert ternary_cfg.lambda_warmup_steps == 0.1
    assert ternary_cfg.activation_bits is None
    assert ternary_cfg.export.formats == ["master_bf16"]
    assert ternary_cfg.weight_scale == "absmean"


def test_validate_config_rejects_group_scale_with_packed_export(plugin_manager):
    import pydantic

    from axolotl.cli.config import prepare_plugins
    from axolotl.utils.config import validate_config

    cfg = DictDefault(
        {
            **MINIMAL_CONFIG,
            "plugins": [PLUGIN_PATH],
            "ternary": {
                "weight_scale": "group",
                "group_size": 128,
                "export": {"formats": ["hf_bitnet"]},
            },
        }
    )

    with pytest.raises(pydantic.ValidationError, match="cannot be represented"):
        prepare_plugins(cfg)
        validate_config(cfg)


@pytest.mark.parametrize(
    "override,match",
    [
        ({"adapter": "lora"}, "full-finetune only"),
        ({"use_onebitllms": True}, "cannot be combined"),
        ({"qat": {"activation_dtype": "int8"}}, "cannot be combined"),
    ],
)
def test_prepare_plugins_rejects_conflicting_config(plugin_manager, override, match):
    from axolotl.cli.config import prepare_plugins

    cfg = DictDefault(
        {**MINIMAL_CONFIG, "plugins": [PLUGIN_PATH], "ternary": {}, **override}
    )

    with pytest.raises(ValueError, match=match):
        prepare_plugins(cfg)


def test_the_removed_kd_plugin_mode_is_refused_at_config_time():
    """The plugin no longer defers to the KD integration; the schema says why."""
    cfg = DictDefault(
        {
            **MINIMAL_CONFIG,
            "plugins": [PLUGIN_PATH],
            "ternary": {"distill": {"mode": "kd_plugin"}},
        }
    )

    with pytest.raises(ValueError, match="has been removed"):
        TernaryPlugin().register(cfg)


def test_register_recommends_save_only_model(caplog):
    cfg = DictDefault({**MINIMAL_CONFIG, "ternary": {}})

    with capture_axolotl_warnings(caplog):
        TernaryPlugin().register(cfg)

    assert "save_only_model" in caplog.text


@pytest.mark.parametrize(
    "override,match",
    [
        ({"smoothing": True}, "smoothing"),
    ],
)
def test_post_model_build_rejects_unimplemented_options(tmp_path, override, match):
    model = _tiny_llama()
    cfg = _plugin_cfg(tmp_path, **override)

    with pytest.raises(NotImplementedError, match=match):
        TernaryPlugin().post_model_build(cfg, model)

    assert not list(iter_ternary_modules(model))


@pytest.mark.parametrize("init", ["svid"])
def test_post_model_build_rejects_an_unimplemented_init(tmp_path, init):
    """The init pass runs on the swapped modules, so this one fails after the surgery."""
    model = _tiny_llama()
    cfg = _plugin_cfg(tmp_path, init=init, weight_scale="learnable")

    with pytest.raises(NotImplementedError, match=init):
        TernaryPlugin().post_model_build(cfg, model)


def test_post_model_build_starts_at_lambda_one(tmp_path):
    """Eval/inference never runs the schedule callback, so λ must start quantized."""
    model = _tiny_llama()
    cfg = _plugin_cfg(tmp_path)
    plugin = TernaryPlugin()

    plugin.post_model_build(model=model, cfg=cfg)

    assert all(module.lambda_ == 1.0 for _, module in iter_ternary_modules(model))


def test_the_schedule_callback_takes_lambda_back_down_when_training_starts(tmp_path):
    model = _tiny_llama()
    cfg = _plugin_cfg(tmp_path)
    plugin = TernaryPlugin()
    plugin.post_model_build(model=model, cfg=cfg)
    lambda_cb = next(
        cb
        for cb in plugin.add_callbacks_pre_trainer(cfg, model)
        if isinstance(cb, LambdaScheduleCallback)
    )

    lambda_cb.on_train_begin(
        TrainingArguments(output_dir=str(tmp_path), report_to=[]),
        TrainerState(global_step=0, max_steps=STEPS),
        TrainerControl(),
    )

    assert all(module.lambda_ == 0.0 for _, module in iter_ternary_modules(model))


def test_evaluate_only_entry_points_see_the_quantized_model(tmp_path):
    """`axolotl evaluate`/`inference` build the model and never start a train loop."""
    model = _tiny_llama()
    batch = _batch()
    plugin = TernaryPlugin()
    plugin.post_model_build(model=model, cfg=_plugin_cfg(tmp_path))
    plugin.add_callbacks_pre_trainer(_plugin_cfg(tmp_path), model)

    model.eval()
    with torch.no_grad():
        quantized = model(**batch).logits.clone()
        for _, module in iter_ternary_modules(model):
            module.set_lambda(0.0)
        latent = model(**batch).logits

    assert not torch.allclose(quantized, latent, atol=1e-4)


def test_post_model_build_keeps_lambda_one_without_a_schedule(tmp_path):
    model = _tiny_llama()

    TernaryPlugin().post_model_build(
        model=model, cfg=_plugin_cfg(tmp_path, lambda_schedule="none")
    )

    assert all(module.lambda_ == 1.0 for _, module in iter_ternary_modules(model))


def test_post_model_build_enables_shared_act_quant(tmp_path):
    model = _tiny_llama()

    TernaryPlugin().post_model_build(
        model=model, cfg=_plugin_cfg(tmp_path, share_act_quant=True)
    )

    assert all(module.share_act_quant for _, module in iter_ternary_modules(model))


def test_callbacks_follow_the_config(tmp_path):
    model = _tiny_llama()
    cfg = _plugin_cfg(tmp_path, lambda_schedule="none", log_code_flip_every=0)
    cfg["weight_decay"] = 0.0
    plugin = TernaryPlugin()
    plugin.post_model_build(cfg, model)

    callbacks = plugin.add_callbacks_pre_trainer(cfg, model)

    # only the always-on zero-gradient watch survives every knob being turned off
    assert [type(cb).__name__ for cb in callbacks] == ["ZeroGradWarningCallback"]


def test_distillation_hooks_are_off_by_default(tmp_path):
    plugin = TernaryPlugin()
    cfg = _plugin_cfg(tmp_path)

    assert plugin.get_trainer_cls(cfg) is None
    assert plugin.get_training_args(cfg) == {}


def test_inprocess_distillation_selects_the_trainer_and_args():
    from axolotl.integrations.ternary.distill import TernaryDistillTrainer

    plugin = TernaryPlugin()
    cfg = DictDefault(
        {
            "base_model": "meta-llama/Llama-3.2-1B",
            "ternary": {"distill": {"mode": "inprocess", "hidden_weight": 0.5}},
        }
    )

    assert plugin.get_trainer_cls(cfg) is TernaryDistillTrainer
    assert plugin.get_training_args(cfg) == {
        "ternary_distill_teacher_model": "meta-llama/Llama-3.2-1B",
        # the schema default, not a constant this test should pin
        "ternary_distill_logits_weight": TernaryDistillConfig().logits_weight,
        "ternary_distill_logits_temperature": 2.0,
        "ternary_distill_hidden_weight": 0.5,
        "ternary_distill_hidden_loss": TernaryDistillConfig().hidden_loss,
        "ternary_distill_hidden_huber_delta": TernaryDistillConfig().hidden_huber_delta,
        "ternary_distill_prefetch_teacher": TernaryDistillConfig().prefetch_teacher,
        "ternary_distill_teacher_prefetch_depth": (
            TernaryDistillConfig().teacher_prefetch_depth
        ),
        "ternary_distill_teacher_endpoint": TernaryDistillConfig().teacher_endpoint,
        "ternary_distill_attn_relation_layer": None,
        "ternary_distill_teacher_device_map": None,
        "ternary_distill_schedule": "constant",
        "ternary_distill_anchor_start": TernaryDistillConfig().anchor_start,
    }


def test_post_train_unload_runs_the_export(monkeypatch, tmp_path):
    calls: list[tuple[DictDefault, list[str]]] = []

    def fake_run_export(cfg, formats=None, **kwargs):
        calls.append((cfg, list(formats)))
        return {"master_bf16": tmp_path}

    monkeypatch.setattr(
        "axolotl.integrations.ternary.export.run_export", fake_run_export
    )
    cfg = _plugin_cfg(tmp_path, export={"formats": ["master_bf16", "i2_s"]})

    TernaryPlugin().post_train_unload(cfg)

    assert calls[0][1] == ["master_bf16", "i2_s"]


def test_post_train_unload_without_formats_is_a_noop(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise AssertionError("run_export must not be called without export formats")

    monkeypatch.setattr("axolotl.integrations.ternary.export.run_export", fail)

    TernaryPlugin().post_train_unload(_plugin_cfg(tmp_path, export={"formats": []}))


def test_post_train_unload_passes_the_live_manifest(monkeypatch, tmp_path):
    """The λ the bake happened at only exists on the manifest the run built."""
    seen: list = []
    monkeypatch.setattr(
        "axolotl.integrations.ternary.export.run_export",
        lambda cfg, formats=None, manifest=None, **kwargs: seen.append(manifest) or {},
    )
    model = _tiny_llama()
    plugin = TernaryPlugin()
    cfg = _plugin_cfg(tmp_path)
    plugin.post_model_build(cfg, model)

    plugin.post_train_unload(cfg)

    assert seen == [plugin.manifest]


def test_post_train_unload_survives_a_torn_down_process_group(monkeypatch, tmp_path):
    """`train.py` calls `cleanup_distributed()` before the CLI unloads the plugins."""
    from axolotl.integrations.ternary import is_export_rank

    monkeypatch.setattr(
        "axolotl.utils.distributed.is_main_process",
        lambda: (_ for _ in ()).throw(ValueError("process group has been destroyed")),
    )
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    calls: list = []
    monkeypatch.setattr(
        "axolotl.integrations.ternary.export.run_export",
        lambda *args, **kwargs: calls.append(1) or {},
    )

    assert is_export_rank() is True
    TernaryPlugin().post_train_unload(_plugin_cfg(tmp_path))
    assert calls == [1]

    monkeypatch.setenv("RANK", "1")
    assert is_export_rank() is False
    TernaryPlugin().post_train_unload(_plugin_cfg(tmp_path))
    assert calls == [1]


def test_register_rejects_distillation_on_the_rl_path():
    cfg = DictDefault(
        {
            "base_model": "meta-llama/Llama-3.2-1B",
            "rl": "dpo",
            "save_only_model": True,
            "plugins": [PLUGIN_PATH],
            "ternary": {"distill": {"mode": "inprocess"}},
        }
    )

    with pytest.raises(ValueError, match="cannot run under `rl: dpo`"):
        TernaryPlugin().register(cfg)


def test_get_trainer_cls_declines_the_rl_path():
    cfg = DictDefault(
        {
            "base_model": "meta-llama/Llama-3.2-1B",
            "rl": "dpo",
            "ternary": {"distill": {"mode": "inprocess"}},
        }
    )

    assert TernaryPlugin().get_trainer_cls(cfg) is None


def test_register_allows_pure_qat_on_the_rl_path():
    cfg = DictDefault(
        {"base_model": "meta-llama/Llama-3.2-1B", "rl": "dpo", "save_only_model": True}
    )

    TernaryPlugin().register(cfg)
