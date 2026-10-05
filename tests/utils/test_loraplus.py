"""Tests for reusable LoRA+ optimizer parameter groups."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

from axolotl.utils.optimizers.loraplus import apply_loraplus_lr_groups


def _params_to_groups(groups):
    return {id(param): group for group in groups for param in group["params"]}


def test_loraplus_groups_preserve_metadata_and_scheduler_ratio():
    base = torch.nn.Parameter(torch.randn(4, 4))
    lora_a = torch.nn.Parameter(torch.randn(2, 4))
    lora_b = torch.nn.Parameter(torch.randn(4, 2))
    unrelated = torch.nn.Parameter(torch.randn(4, 2))
    fallback_lora_b = torch.nn.Parameter(torch.randn(4, 2))
    groups = [
        {
            "params": [base, lora_a, lora_b, unrelated],
            "lr": 2e-3,
            "initial_lr": 2e-3,
            "weight_decay": 0.1,
            "family_flag": "projected",
        },
        {
            "params": [fallback_lora_b],
            "lr": 3e-3,
            "weight_decay": 0.0,
            "family_flag": "fallback",
        },
        {"params": [], "family_flag": "empty"},
    ]
    original_params = [list(group["params"]) for group in groups]
    named_parameters = [
        ("layer.weight", base),
        ("layer.lora_A.default.weight", lora_a),
        ("layer.lora_B.default.weight", lora_b),
        ("layer.some_lora_B_module.weight", unrelated),
        ("embed.lora_B.default.weight", fallback_lora_b),
    ]

    result = apply_loraplus_lr_groups(
        groups,
        named_parameters,
        default_lr=1e-3,
        loraplus_lr_ratio=8,
        eligible=lambda _name, _param, group: group["family_flag"] == "projected",
    )

    assert [group["params"] for group in groups] == original_params
    params_to_groups = _params_to_groups(result)
    assert set(params_to_groups) == {id(param) for _, param in named_parameters}
    assert len(params_to_groups) == sum(len(group["params"]) for group in result)
    assert params_to_groups[id(lora_a)]["lr"] == 2e-3
    assert params_to_groups[id(lora_b)]["lr"] == 16e-3
    assert params_to_groups[id(lora_b)]["initial_lr"] == 16e-3
    assert params_to_groups[id(lora_b)]["weight_decay"] == 0.1
    assert params_to_groups[id(lora_b)]["family_flag"] == "projected"
    assert params_to_groups[id(unrelated)]["lr"] == 2e-3
    assert params_to_groups[id(fallback_lora_b)]["lr"] == 3e-3
    assert params_to_groups[id(fallback_lora_b)]["family_flag"] == "fallback"
    assert any(group["family_flag"] == "empty" for group in result)

    optimizer = torch.optim.SGD(result, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 0.5)
    optimizer.step()
    scheduler.step()
    scheduled = _params_to_groups(optimizer.param_groups)
    assert scheduled[id(lora_b)]["lr"] == 8 * scheduled[id(lora_a)]["lr"]


def test_loraplus_groups_leave_non_2d_and_no_ratio_unchanged():
    lora_a_3d = torch.nn.Parameter(torch.randn(2, 3, 4))
    groups = [{"params": [lora_a_3d], "family_flag": "synthetic"}]
    named_parameters = [("layer.lora_A.default.weight", lora_a_3d)]

    split = apply_loraplus_lr_groups(
        groups,
        named_parameters,
        default_lr=1e-3,
        loraplus_lr_ratio=8,
        eligible=lambda *_: True,
    )
    unchanged = apply_loraplus_lr_groups(
        groups,
        named_parameters,
        default_lr=1e-3,
        loraplus_lr_ratio=None,
        eligible=lambda *_: True,
    )

    assert split == groups
    assert unchanged == groups
    assert split[0] is not groups[0]
    assert unchanged[0] is not groups[0]


def test_loraplus_routes_lora_b_bias_with_b_group():
    lora_b_weight = torch.nn.Parameter(torch.randn(4, 2))
    lora_b_bias = torch.nn.Parameter(torch.randn(4))
    lora_a_vector = torch.nn.Parameter(torch.randn(2))
    groups = [{"params": [lora_b_weight, lora_b_bias, lora_a_vector], "lr": 1e-3}]
    named_parameters = [
        ("layer.lora_B.default.weight", lora_b_weight),
        ("layer.lora_B.default.bias", lora_b_bias),
        ("layer.lora_A.default.scale", lora_a_vector),
    ]

    split = apply_loraplus_lr_groups(
        groups,
        named_parameters,
        default_lr=1e-3,
        loraplus_lr_ratio=8,
        eligible=lambda *_: True,
    )

    by_param = {id(p): g for g in split for p in g["params"]}
    assert by_param[id(lora_b_weight)]["lr"] == 8e-3
    assert by_param[id(lora_b_bias)]["lr"] == 8e-3
    assert by_param[id(lora_a_vector)]["lr"] == 1e-3


def test_loraplus_keeps_embedding_and_saved_head_overrides():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Module()
            self.layer.lora_A = torch.nn.ModuleDict(
                {"default": torch.nn.Linear(4, 2, bias=False)}
            )
            self.layer.lora_B = torch.nn.ModuleDict(
                {"default": torch.nn.Linear(2, 4, bias=False)}
            )
            self.embed_tokens = torch.nn.Module()
            self.embed_tokens.token_adapter = torch.nn.Module()
            self.embed_tokens.token_adapter.register_parameter(
                "trainable_tokens_delta", torch.nn.Parameter(torch.randn(151, 4))
            )
            self.score_head = torch.nn.Module()
            self.score_head.modules_to_save = torch.nn.ModuleDict(
                {"default": torch.nn.Linear(4, 151, bias=False)}
            )

    model = Model()
    named = dict(model.named_parameters())
    groups = [
        {
            "params": [
                named["layer.lora_A.default.weight"],
                named["layer.lora_B.default.weight"],
            ],
            "weight_decay": 0.1,
            "lr": 2.5e-5,
        },
        {
            "params": [
                named["embed_tokens.token_adapter.trainable_tokens_delta"],
                named["score_head.modules_to_save.default.weight"],
            ],
            "weight_decay": 0.0,
            "lr": 1e-4,
        },
    ]
    result = apply_loraplus_lr_groups(
        groups,
        model.named_parameters(),
        default_lr=2.5e-5,
        loraplus_lr_ratio=8,
        eligible=lambda *_: True,
    )
    parameter_groups = _params_to_groups(result)

    assert parameter_groups[id(named["layer.lora_A.default.weight"])]["lr"] == 2.5e-5
    assert parameter_groups[id(named["layer.lora_B.default.weight"])]["lr"] == 2e-4
    assert (
        parameter_groups[
            id(named["embed_tokens.token_adapter.trainable_tokens_delta"])
        ]["lr"]
        == 1e-4
    )
    assert (
        parameter_groups[id(named["score_head.modules_to_save.default.weight"])]["lr"]
        == 1e-4
    )

    optimizer = torch.optim.SGD(result, lr=2.5e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 0.5)
    optimizer.step()
    scheduler.step()
    scheduled_groups = _params_to_groups(optimizer.param_groups)
    assert scheduled_groups[id(named["layer.lora_A.default.weight"])]["lr"] == 1.25e-5
    assert scheduled_groups[id(named["layer.lora_B.default.weight"])]["lr"] == 1e-4
    assert (
        scheduled_groups[
            id(named["embed_tokens.token_adapter.trainable_tokens_delta"])
        ]["lr"]
        == 5e-5
    )
    assert (
        scheduled_groups[id(named["score_head.modules_to_save.default.weight"])]["lr"]
        == 5e-5
    )


def _optimizer_mixin_module():
    path = Path(__file__).parents[2] / "src/axolotl/core/trainers/mixins/optimizer.py"
    spec = importlib.util.spec_from_file_location("optimizer_mixin_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _optimizer_model():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Module()
            self.layer.lora_A = torch.nn.ModuleDict(
                {"default": torch.nn.Linear(4, 2, bias=False)}
            )
            self.layer.lora_B = torch.nn.ModuleDict(
                {"default": torch.nn.Linear(2, 4, bias=False)}
            )
            self.embed_tokens = torch.nn.Module()
            self.embed_tokens.token_adapter = torch.nn.Module()
            self.embed_tokens.token_adapter.register_parameter(
                "trainable_tokens_delta", torch.nn.Parameter(torch.randn(151, 4))
            )
            self.score_head = torch.nn.Module()
            self.score_head.modules_to_save = torch.nn.ModuleDict(
                {"default": torch.nn.Linear(4, 151, bias=False)}
            )

    return Model()


def _optimizer_stub(module, model, *, embedding_lr_scale):
    class Stub:
        create_optimizer = module.OptimizerMixin.create_optimizer
        create_optimizer_grouped_parameters = (
            module.OptimizerMixin.create_optimizer_grouped_parameters
        )

        def __init__(self):
            self.args = SimpleNamespace(
                loraplus_lr_ratio=8,
                loraplus_lr_embedding=1e-6,
                embedding_lr_scale=embedding_lr_scale,
                embedding_lr=None,
                lr_groups=None,
                weight_decay=0.1,
            )
            self.model = model
            self.optimizer = None
            self.optimizer_cls_and_kwargs = (torch.optim.SGD, {"lr": 2.5e-5})

        def get_decay_parameter_names(self, input_model):
            return {name for name, _ in input_model.named_parameters()}

    return Stub()


def test_optimizer_mixin_composes_loraplus_with_embedding_lr_scale():
    module = _optimizer_mixin_module()
    model = _optimizer_model()
    optimizer = _optimizer_stub(
        module, model, embedding_lr_scale=4.0
    ).create_optimizer()
    named = dict(model.named_parameters())
    groups = _params_to_groups(optimizer.param_groups)

    assert groups[id(named["layer.lora_A.default.weight"])]["lr"] == 2.5e-5
    assert groups[id(named["layer.lora_B.default.weight"])]["lr"] == 2e-4
    assert (
        groups[id(named["embed_tokens.token_adapter.trainable_tokens_delta"])]["lr"]
        == 1e-4
    )
    assert groups[id(named["score_head.modules_to_save.default.weight"])]["lr"] == 1e-4

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 0.5)
    optimizer.step()
    scheduler.step()
    groups = _params_to_groups(optimizer.param_groups)
    assert groups[id(named["layer.lora_B.default.weight"])]["lr"] == 1e-4
    assert (
        groups[id(named["embed_tokens.token_adapter.trainable_tokens_delta"])]["lr"]
        == 5e-5
    )


def test_optimizer_mixin_uses_peft_factory_without_embedding_override(monkeypatch):
    module = _optimizer_mixin_module()
    model = _optimizer_model()
    expected = object()
    calls = []

    def create_loraplus(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(module, "create_loraplus_optimizer", create_loraplus)
    optimizer = _optimizer_stub(
        module, model, embedding_lr_scale=None
    ).create_optimizer()

    assert optimizer is expected
    assert calls[0][1]["loraplus_lr_ratio"] == 8


def _embedding_lora_model():
    model = _optimizer_model()
    model.embed = torch.nn.Module()
    model.embed.lora_embedding_A = torch.nn.ParameterDict(
        {"default": torch.nn.Parameter(torch.randn(2, 151))}
    )
    model.embed.lora_embedding_B = torch.nn.ParameterDict(
        {"default": torch.nn.Parameter(torch.randn(4, 2))}
    )
    return model


def test_optimizer_mixin_warns_loraplus_lr_embedding_unused(monkeypatch):
    module = _optimizer_mixin_module()
    warnings = []
    monkeypatch.setattr(
        module.LOG, "warning", lambda msg, *a, **k: warnings.append(msg)
    )

    _optimizer_stub(
        module, _embedding_lora_model(), embedding_lr_scale=4.0
    ).create_optimizer()

    assert len(warnings) == 1
    assert "loraplus_lr_embedding" in warnings[0]


def test_optimizer_mixin_no_embedding_warning_without_embedding_lora(monkeypatch):
    module = _optimizer_mixin_module()
    warnings = []
    monkeypatch.setattr(
        module.LOG, "warning", lambda msg, *a, **k: warnings.append(msg)
    )

    _optimizer_stub(
        module, _optimizer_model(), embedding_lr_scale=4.0
    ).create_optimizer()

    assert warnings == []


def test_optimizer_mixin_routes_lora_embedding_factors_to_embedding_lr():
    module = _optimizer_mixin_module()
    model = _embedding_lora_model()
    optimizer = _optimizer_stub(
        module, model, embedding_lr_scale=4.0
    ).create_optimizer()
    named = dict(model.named_parameters())
    groups = _params_to_groups(optimizer.param_groups)

    assert groups[id(named["embed.lora_embedding_A.default"])]["lr"] == 1e-4
    assert groups[id(named["embed.lora_embedding_B.default"])]["lr"] == 1e-4
    assert groups[id(named["layer.lora_B.default.weight"])]["lr"] == 2e-4
