"""Tests for reusable LoRA+ optimizer parameter groups."""

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
