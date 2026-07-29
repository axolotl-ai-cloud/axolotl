"""CPU-only tests for the zero-gradient-while-targeted warning.

The failure this callback exists for is invisible in training metrics: a module the
loss never reaches takes the full quantization damage and produces no gradient to heal
it, while the loss curve looks exactly like a healthy run's. So the tests drive real
backward passes and assert on what the warning names, not just that it fired.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    TrainerControl,
    TrainerState,
)

from axolotl.integrations.ternary import TernaryPlugin
from axolotl.integrations.ternary.callbacks import (
    ZERO_GRAD_MONITORED_STEPS,
    ZeroGradWarningCallback,
)
from axolotl.integrations.ternary.modules import iter_ternary_modules
from axolotl.integrations.ternary.swap import convert_model
from axolotl.utils.dict import DictDefault

from tests.conftest import capture_axolotl_warnings

ARGS = SimpleNamespace()
CONTROL = TrainerControl()
HIDDEN = 32
WINDOW = 4

PRESET_TARGETS = [
    r".*\.self_attn\.(q|k|v|o)_proj",
    r".*\.mlp\.(gate|up|down)_proj",
]


def _tiny_llama() -> LlamaForCausalLM:
    torch.manual_seed(0)
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=HIDDEN,
            intermediate_size=HIDDEN * 2,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            tie_word_embeddings=False,
        )
    )


def _attach_unreached(model: LlamaForCausalLM, path: str) -> None:
    """Add a Linear at `model.layers.0.<path>` that no forward ever calls."""
    parent = model.model.layers[0]
    branch, _, leaf = path.rpartition(".")
    if branch:
        child = nn.Module()
        parent.add_module(branch, child)
        parent = child
    parent.add_module(leaf, nn.Linear(HIDDEN, HIDDEN, bias=False))


def _converted(unreached: str | None = None) -> LlamaForCausalLM:
    model = _tiny_llama()
    targets = list(PRESET_TARGETS)
    if unreached is not None:
        _attach_unreached(model, unreached)
        targets.append(rf"model\.layers\.0\.{unreached.replace('.', chr(92) + '.')}")
    convert_model(model, DictDefault({"ternary": {"target_modules": targets}}))
    return model


def _batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 64, (2, 8), generator=generator)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids}


def _step(
    model: LlamaForCausalLM, callback: ZeroGradWarningCallback, step: int
) -> None:
    model.zero_grad(set_to_none=True)
    model(**_batch()).loss.backward()
    callback.on_pre_optimizer_step(ARGS, TrainerState(global_step=step), CONTROL)
    callback.on_step_end(ARGS, TrainerState(global_step=step + 1), CONTROL)


def _train(model: LlamaForCausalLM, callback: ZeroGradWarningCallback) -> None:
    for step in range(WINDOW):
        _step(model, callback, step)


def _modules(model: LlamaForCausalLM) -> dict[str, nn.Module]:
    return dict(iter_ternary_modules(model))


# ------------------------------------------------------------------- detection


def test_a_targeted_module_no_loss_reaches_is_named(caplog):
    model = _converted(unreached="mtp.up_proj")
    callback = ZeroGradWarningCallback(model, monitored_steps=WINDOW)

    caplog.clear()
    with capture_axolotl_warnings(caplog):
        _train(model, callback)

    assert callback.warned
    assert "model.layers.0.mtp.up_proj" in caplog.text
    assert "received no gradient" in caplog.text


def test_a_frozen_targeted_module_is_named(caplog):
    """A latent excluded from the optimizer is damaged and unhealable just the same."""
    model = _converted()
    _modules(model)["model.layers.0.mlp.down_proj"].weight.requires_grad_(False)
    callback = ZeroGradWarningCallback(model, monitored_steps=WINDOW)

    caplog.clear()
    with capture_axolotl_warnings(caplog):
        _train(model, callback)

    assert callback.warned
    assert "model.layers.0.mlp.down_proj" in caplog.text


def test_normal_training_stays_silent(caplog):
    model = _converted()
    callback = ZeroGradWarningCallback(model, monitored_steps=WINDOW)

    caplog.clear()
    with capture_axolotl_warnings(caplog):
        _train(model, callback)

    assert not callback.warned
    assert caplog.text == ""
    assert len(callback.seen_grad) == len(callback.ternary_modules)


def test_a_module_that_wakes_up_inside_the_window_is_not_reported(caplog):
    """A cold expert routed on a later step has been reached; the window is cumulative."""
    model = _converted()
    late = _modules(model)["model.layers.0.mlp.down_proj"]
    late.weight.requires_grad_(False)
    callback = ZeroGradWarningCallback(model, monitored_steps=WINDOW)

    caplog.clear()
    with capture_axolotl_warnings(caplog):
        _step(model, callback, 0)
        late.weight.requires_grad_(True)
        for step in range(1, WINDOW):
            _step(model, callback, step)

    assert not callback.warned
    assert caplog.text == ""


def test_a_partitioned_optimizer_is_not_read_as_an_unreachable_model(caplog):
    """DeepSpeed ZeRO-2/3 reduces into a flat partition and clears `param.grad` before
    the step, so the presence check would otherwise accuse every targeted module."""
    model = _converted()
    callback = ZeroGradWarningCallback(model, monitored_steps=WINDOW)

    caplog.clear()
    with capture_axolotl_warnings(caplog):
        for step in range(WINDOW):
            model.zero_grad(set_to_none=True)
            model(**_batch()).loss.backward()
            for module in _modules(model).values():
                module.weight.grad = None
            callback.on_pre_optimizer_step(
                ARGS, TrainerState(global_step=step), CONTROL
            )
            callback.on_step_end(ARGS, TrainerState(global_step=step + 1), CONTROL)

    assert not callback.warned
    assert callback.observed_steps == WINDOW
    assert caplog.text == ""


def test_a_deepspeed_run_stands_the_watch_down(caplog):
    """Even with a genuinely unreached module: under ZeRO the hook has no evidence."""
    model = _converted(unreached="mtp.up_proj")
    callback = ZeroGradWarningCallback(model, monitored_steps=WINDOW)
    args = SimpleNamespace(deepspeed="deepspeed_configs/zero2.json")

    caplog.clear()
    with capture_axolotl_warnings(caplog):
        for step in range(WINDOW):
            model.zero_grad(set_to_none=True)
            model(**_batch()).loss.backward()
            callback.on_pre_optimizer_step(
                args, TrainerState(global_step=step), CONTROL
            )
            callback.on_step_end(args, TrainerState(global_step=step + 1), CONTROL)

    assert not callback.monitoring
    assert not callback.warned
    assert callback.observed_steps == 0
    assert caplog.text == ""


def test_a_flat_parameter_view_stands_the_watch_down(caplog):
    """FSDP1 without `use_orig_params` leaves the module holding a plain view."""
    model = _converted(unreached="mtp.up_proj")
    for module in _modules(model).values():
        weight = module.weight.detach()
        del module._parameters["weight"]
        module.weight = weight
    callback = ZeroGradWarningCallback(model, monitored_steps=1)

    caplog.clear()
    with capture_axolotl_warnings(caplog):
        callback.on_pre_optimizer_step(ARGS, TrainerState(global_step=0), CONTROL)
        callback.on_step_end(ARGS, TrainerState(global_step=1), CONTROL)

    assert not callback.warned
    assert callback.observed_steps == 0
    assert caplog.text == ""


def test_a_gradient_allocated_and_left_at_zero_does_not_count(caplog):
    """Sharded runs hand an unused parameter a zero-filled gradient, not `None`."""
    model = _converted()
    modules = _modules(model)
    for name, module in modules.items():
        grad = torch.zeros_like(module.weight)
        if name != "model.layers.0.self_attn.o_proj":
            grad[0, 0] = 1.0
        module.weight.grad = grad
    callback = ZeroGradWarningCallback(model, monitored_steps=1)

    caplog.clear()
    with capture_axolotl_warnings(caplog):
        callback.on_pre_optimizer_step(ARGS, TrainerState(global_step=0), CONTROL)
        callback.on_step_end(ARGS, TrainerState(global_step=1), CONTROL)

    assert callback.warned
    assert "model.layers.0.self_attn.o_proj" in caplog.text


# -------------------------------------------------------------- the watch itself


def test_the_warning_fires_once(caplog):
    model = _converted(unreached="mtp.up_proj")
    callback = ZeroGradWarningCallback(model, monitored_steps=WINDOW)

    with capture_axolotl_warnings(caplog):
        _train(model, callback)
        caplog.clear()
        for step in range(WINDOW, WINDOW * 2):
            _step(model, callback, step)

    assert caplog.text == ""


def test_monitoring_stops_once_every_module_has_been_reached():
    model = _converted()
    callback = ZeroGradWarningCallback(model, monitored_steps=WINDOW)

    _step(model, callback, 0)

    assert not callback.monitoring
    assert callback.observed_steps == 1


def test_a_trainer_that_never_reports_a_step_is_not_accused(caplog):
    """Without `on_pre_optimizer_step` there is no evidence, so there is no verdict."""
    model = _converted()
    callback = ZeroGradWarningCallback(model, monitored_steps=WINDOW)

    caplog.clear()
    with capture_axolotl_warnings(caplog):
        callback.on_step_end(ARGS, TrainerState(global_step=WINDOW), CONTROL)

    assert not callback.warned
    assert caplog.text == ""


def test_a_run_shorter_than_the_window_still_reports_when_it_ends(caplog):
    """A short ablation or smoke heal is where a mis-wired target list is cheapest to
    catch, and it is the run `on_step_end` never reaches a verdict on."""
    model = _converted(unreached="mtp.up_proj")
    callback = ZeroGradWarningCallback(model, monitored_steps=ZERO_GRAD_MONITORED_STEPS)

    caplog.clear()
    with capture_axolotl_warnings(caplog):
        for step in range(3):
            _step(model, callback, step)
        callback.on_train_end(ARGS, TrainerState(global_step=3), CONTROL)

    assert callback.warned
    assert "model.layers.0.mtp.up_proj" in caplog.text
    assert "3 optimizer step" in caplog.text


def test_a_healthy_short_run_stays_silent_at_train_end(caplog):
    model = _converted()
    callback = ZeroGradWarningCallback(model, monitored_steps=ZERO_GRAD_MONITORED_STEPS)

    caplog.clear()
    with capture_axolotl_warnings(caplog):
        _step(model, callback, 0)
        callback.on_train_end(ARGS, TrainerState(global_step=1), CONTROL)

    assert not callback.warned
    assert caplog.text == ""


def test_train_end_does_not_repeat_a_warning_the_window_already_gave(caplog):
    model = _converted(unreached="mtp.up_proj")
    callback = ZeroGradWarningCallback(model, monitored_steps=WINDOW)

    with capture_axolotl_warnings(caplog):
        _train(model, callback)
        caplog.clear()
        callback.on_train_end(ARGS, TrainerState(global_step=WINDOW), CONTROL)

    assert caplog.text == ""


def test_a_run_that_never_stepped_is_not_accused_at_train_end(caplog):
    model = _converted()
    callback = ZeroGradWarningCallback(model, monitored_steps=WINDOW)

    caplog.clear()
    with capture_axolotl_warnings(caplog):
        callback.on_train_end(ARGS, TrainerState(global_step=0), CONTROL)

    assert not callback.warned
    assert caplog.text == ""


def test_the_window_defaults_to_the_module_constant():
    model = _converted()

    assert ZeroGradWarningCallback(model).monitored_steps == ZERO_GRAD_MONITORED_STEPS


# ----------------------------------------------------------------- the message


def test_the_message_names_the_family_and_its_cause():
    model = _converted()
    callback = ZeroGradWarningCallback(model)

    message = callback._message(["model.layers.0.mtp.up_proj"])

    assert "multi-token-prediction head" in message
    assert "auxiliary loss" in message
    assert "keep_fp_modules:" in message


def test_the_message_reports_unrecognized_modules_apart():
    model = _converted()
    callback = ZeroGradWarningCallback(model)

    message = callback._message(["model.layers.0.mlp.experts.7.up_proj"])

    assert "no auxiliary-module family" in message
    assert "cold MoE expert" in message
    assert "keep_fp_modules:" not in message


def test_the_message_truncates_a_long_list():
    model = _converted()
    callback = ZeroGradWarningCallback(model)
    names = [f"model.layers.{index}.mlp.experts.0.up_proj" for index in range(9)]

    message = callback._message(names)

    assert "(+4 more)" in message
    assert names[-1] not in message


# ------------------------------------------------------------ plugin wiring


@pytest.fixture(name="plugin_cfg")
def fixture_plugin_cfg(tmp_path):
    return DictDefault(
        {
            "output_dir": str(tmp_path),
            "ternary": {"target_modules": PRESET_TARGETS},
        }
    )


def test_the_plugin_installs_the_watch(plugin_cfg):
    model = _tiny_llama()
    plugin = TernaryPlugin()
    plugin.post_model_build(plugin_cfg, model)

    callbacks = plugin.add_callbacks_pre_trainer(plugin_cfg, model)

    watch = next(cb for cb in callbacks if isinstance(cb, ZeroGradWarningCallback))
    assert len(watch.ternary_modules) == len(list(iter_ternary_modules(model)))


def test_the_watch_is_installed_even_with_every_other_callback_off(plugin_cfg):
    plugin_cfg["ternary"]["lambda_schedule"] = "none"
    plugin_cfg["weight_decay"] = 0.0
    model = _tiny_llama()
    plugin = TernaryPlugin()
    plugin.post_model_build(plugin_cfg, model)

    callbacks = plugin.add_callbacks_pre_trainer(plugin_cfg, model)

    assert [type(cb).__name__ for cb in callbacks] == ["ZeroGradWarningCallback"]
