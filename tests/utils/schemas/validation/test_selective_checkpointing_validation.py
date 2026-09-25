"""Test for config validation for selective activation checkpointing."""

import pytest

from axolotl.utils.config import validate_config
from axolotl.utils.dict import DictDefault


class TestSelectiveCheckpointing:
    """
    Test cases for selective_checkpointing schema validation
    """

    def test_bool_shorthand_normalizes_to_attention(self, min_base_cfg):
        cfg = (
            DictDefault(
                gradient_checkpointing=True,
                selective_checkpointing=True,
            )
            | min_base_cfg
        )

        cfg = validate_config(cfg)
        assert cfg.selective_checkpointing["save"] == ["attention"]
        assert cfg.selective_checkpointing["save_sliding_window"] is False

    def test_false_normalizes_to_none(self, min_base_cfg):
        cfg = (
            DictDefault(
                gradient_checkpointing=True,
                selective_checkpointing=False,
            )
            | min_base_cfg
        )

        cfg = validate_config(cfg)
        assert cfg.selective_checkpointing is None

    def test_custom_save_list_preserved(self, min_base_cfg):
        cfg = (
            DictDefault(
                gradient_checkpointing=True,
                selective_checkpointing={"save": ["attention", "aten::mm"]},
            )
            | min_base_cfg
        )

        cfg = validate_config(cfg)
        assert cfg.selective_checkpointing["save"] == ["attention", "aten::mm"]

    def test_requires_gradient_checkpointing(self, min_base_cfg):
        cfg = (
            DictDefault(
                selective_checkpointing=True,
            )
            | min_base_cfg
        )

        with pytest.raises(ValueError, match="requires gradient_checkpointing"):
            validate_config(cfg)

    def test_rejects_reentrant(self, min_base_cfg):
        cfg = (
            DictDefault(
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": True},
                selective_checkpointing=True,
            )
            | min_base_cfg
        )

        with pytest.raises(ValueError, match="non-reentrant"):
            validate_config(cfg)

    def test_rejects_trl_activation_offloading(self, min_base_cfg):
        cfg = (
            DictDefault(
                gradient_checkpointing=True,
                activation_offloading=True,
                selective_checkpointing=True,
            )
            | min_base_cfg
        )

        with pytest.raises(ValueError, match="hidden_states"):
            validate_config(cfg)

    def test_composes_with_hidden_states_offload(self, min_base_cfg):
        cfg = (
            DictDefault(
                gradient_checkpointing=True,
                activation_offloading="hidden_states",
                selective_checkpointing=True,
            )
            | min_base_cfg
        )

        cfg = validate_config(cfg)
        assert cfg.selective_checkpointing["save"] == ["attention"]
        assert cfg.activation_offloading == "hidden_states"

    def test_rejects_matmul_save_with_adapter(self, min_base_cfg):
        cfg = (
            DictDefault(
                gradient_checkpointing=True,
                adapter="lora",
                lora_r=16,
                lora_alpha=32,
                lora_target_linear=True,
                selective_checkpointing={"save": ["attention", "aten::mm"]},
            )
            | min_base_cfg
        )

        with pytest.raises(ValueError, match="in-place"):
            validate_config(cfg)

    def test_allows_matmul_save_without_adapter(self, min_base_cfg):
        cfg = (
            DictDefault(
                gradient_checkpointing=True,
                selective_checkpointing={"save": ["attention", "aten::mm"]},
            )
            | min_base_cfg
        )

        cfg = validate_config(cfg)
        assert cfg.selective_checkpointing["save"] == ["attention", "aten::mm"]


_KERNELS_OFF = {
    "lora_mlp_kernel": False,
    "lora_qkv_kernel": False,
    "lora_o_kernel": False,
}


def _sac_cfg(min_base_cfg, sac, **overrides):
    return (
        DictDefault(
            gradient_checkpointing=True, selective_checkpointing=sac, **overrides
        )
        | min_base_cfg
    )


def _lora_cfg(min_base_cfg, sac, **overrides):
    return _sac_cfg(
        min_base_cfg,
        sac,
        adapter="lora",
        lora_r=16,
        lora_alpha=32,
        lora_target_linear=True,
        **overrides,
    )


class TestSelectiveCheckpointingRules:
    """save_modules / save_matmul_min_k and their fused-kernel conflicts."""

    def test_save_modules_preserved(self, min_base_cfg):
        cfg = validate_config(
            _sac_cfg(min_base_cfg, {"save_modules": [" down_proj ", "*.o_proj"]})
        )
        assert cfg.selective_checkpointing["save_modules"] == ["down_proj", "*.o_proj"]
        assert cfg.selective_checkpointing["save_matmul_min_k"] is None
        assert cfg.selective_checkpointing["save"] == ["attention"]

    def test_empty_save_modules_normalizes_to_none(self, min_base_cfg):
        cfg = validate_config(_sac_cfg(min_base_cfg, {"save_modules": []}))
        assert cfg.selective_checkpointing["save_modules"] is None

    def test_rejects_empty_save_modules_entry(self, min_base_cfg):
        with pytest.raises(ValueError, match="non-empty"):
            validate_config(
                _sac_cfg(min_base_cfg, {"save_modules": ["down_proj", " "]})
            )

    def test_save_matmul_min_k_preserved(self, min_base_cfg):
        cfg = validate_config(_sac_cfg(min_base_cfg, {"save_matmul_min_k": 8192}))
        assert cfg.selective_checkpointing["save_matmul_min_k"] == 8192

    def test_rejects_min_k_below_one(self, min_base_cfg):
        with pytest.raises(ValueError, match="save_matmul_min_k"):
            validate_config(_sac_cfg(min_base_cfg, {"save_matmul_min_k": 0}))

    def test_lora_default_rejects_save_modules_covered_by_mlp_kernel(
        self, min_base_cfg
    ):
        with pytest.raises(ValueError, match="'down_proj'.*lora_mlp_kernel: true"):
            validate_config(_lora_cfg(min_base_cfg, {"save_modules": ["down_proj"]}))

    def test_lora_default_rejection_with_capabilities(self, min_base_cfg):
        cfg = _lora_cfg(min_base_cfg, {"save_modules": ["down_proj"]})
        with pytest.raises(ValueError, match="lora_mlp_kernel"):
            validate_config(
                cfg,
                capabilities={"n_gpu": 1, "bf16": True, "compute_capability": None},
                env_capabilities={"torch_version": "2.13.0"},
            )

    @pytest.mark.parametrize(
        "entry", ["down_proj", "mlp.down_proj", "*down_proj*", "*.mlp.*_proj", "mlp"]
    )
    def test_lora_mlp_kernel_rejects_mlp_entries(self, min_base_cfg, entry):
        cfg = _lora_cfg(
            min_base_cfg,
            {"save_modules": [entry]},
            **{**_KERNELS_OFF, "lora_mlp_kernel": True},
        )
        with pytest.raises(ValueError, match="lora_mlp_kernel"):
            validate_config(cfg)

    def test_lora_allows_save_modules_when_mlp_kernel_off(self, min_base_cfg):
        cfg = validate_config(
            _lora_cfg(
                min_base_cfg, {"save_modules": ["down_proj"]}, lora_mlp_kernel=False
            )
        )
        assert cfg.selective_checkpointing["save_modules"] == ["down_proj"]

    def test_explicit_kernel_flag_stops_auto_enable(self, min_base_cfg):
        cfg = validate_config(
            _lora_cfg(
                min_base_cfg,
                {"save_modules": ["q_proj", "o_proj"]},
                lora_mlp_kernel=False,
            )
        )
        assert cfg.selective_checkpointing["save_modules"] == ["q_proj", "o_proj"]

    @pytest.mark.parametrize(
        "flag,entry",
        [
            ("lora_qkv_kernel", "q_proj"),
            ("lora_qkv_kernel", "self_attn.v_proj"),
            ("lora_qkv_kernel", "in_proj_qkv"),
            ("lora_o_kernel", "o_proj"),
            ("lora_o_kernel", "out_proj"),
            ("lora_o_kernel", "self_attn"),
        ],
    )
    def test_attention_kernels_reject_their_entries(self, min_base_cfg, flag, entry):
        cfg = _lora_cfg(
            min_base_cfg, {"save_modules": [entry]}, **{**_KERNELS_OFF, flag: True}
        )
        with pytest.raises(ValueError, match=flag):
            validate_config(cfg)

    def test_kernel_does_not_reject_unrelated_entries(self, min_base_cfg):
        cfg = validate_config(
            _lora_cfg(
                min_base_cfg,
                {"save_modules": ["down_proj", "gate_proj"]},
                lora_mlp_kernel=False,
                lora_qkv_kernel=True,
                lora_o_kernel=True,
            )
        )
        assert cfg.selective_checkpointing["save_modules"] == ["down_proj", "gate_proj"]

    def test_lora_rejects_shape_rule_with_fused_kernels(self, min_base_cfg):
        with pytest.raises(ValueError, match="save_matmul_min_k"):
            validate_config(_lora_cfg(min_base_cfg, {"save_matmul_min_k": 8192}))

    def test_lora_allows_shape_rule_with_kernels_off(self, min_base_cfg):
        cfg = validate_config(
            _lora_cfg(min_base_cfg, {"save_matmul_min_k": 8192}, **_KERNELS_OFF)
        )
        assert cfg.selective_checkpointing["save_matmul_min_k"] == 8192

    def test_lora_allows_raw_matmul_save_with_kernels_off(self, min_base_cfg):
        cfg = validate_config(
            _lora_cfg(min_base_cfg, {"save": ["attention", "aten::mm"]}, **_KERNELS_OFF)
        )
        assert cfg.selective_checkpointing["save"] == ["attention", "aten::mm"]

    def test_rl_lora_does_not_auto_enable_kernels(self, min_base_cfg):
        cfg = validate_config(
            _lora_cfg(
                min_base_cfg,
                {"save_modules": ["down_proj"], "save_matmul_min_k": 8192},
                rl="dpo",
            )
        )
        assert cfg.selective_checkpointing["save_modules"] == ["down_proj"]

    def test_kernel_flags_without_adapter_are_ignored(self, min_base_cfg):
        cfg = validate_config(
            _sac_cfg(
                min_base_cfg,
                {"save_modules": ["down_proj"], "save_matmul_min_k": 8192},
                lora_mlp_kernel=True,
            )
        )
        assert cfg.selective_checkpointing["save_modules"] == ["down_proj"]

    def test_flash_attn_fuse_mlp_rejects_mlp_entries(self, min_base_cfg):
        cfg = _sac_cfg(
            min_base_cfg, {"save_modules": ["down_proj"]}, flash_attn_fuse_mlp=True
        )
        with pytest.raises(ValueError, match="flash_attn_fuse_mlp"):
            validate_config(cfg)

    @pytest.mark.parametrize(
        "entry",
        [
            "layers.0",
            "model.layers.3",
            "layers",
            "model",
            "language_model",
            "*layers.1*",
            "*",
            "mixer",
            "shared_expert",
            "feedforward",
        ],
    )
    def test_lora_default_rejects_layer_and_kernel_containers(
        self, min_base_cfg, entry
    ):
        with pytest.raises(ValueError, match="lora_mlp_kernel"):
            validate_config(_lora_cfg(min_base_cfg, {"save_modules": [entry]}))

    def test_layer_container_rejected_by_attention_kernel_alone(self, min_base_cfg):
        cfg = _lora_cfg(
            min_base_cfg,
            {"save_modules": ["layers.0"]},
            **{**_KERNELS_OFF, "lora_o_kernel": True},
        )
        with pytest.raises(ValueError, match="'layers.0' scopes a decoder layer"):
            validate_config(cfg)

    @pytest.mark.parametrize("entry", ["layers.0", "*layers.1*", "mixer"])
    def test_layer_containers_allowed_with_kernels_off(self, min_base_cfg, entry):
        cfg = validate_config(
            _lora_cfg(min_base_cfg, {"save_modules": [entry]}, **_KERNELS_OFF)
        )
        assert cfg.selective_checkpointing["save_modules"] == [entry]
