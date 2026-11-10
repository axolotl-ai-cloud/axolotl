"""Per-step publishing of a Qwen3.6-style mixed NVFP4 + FP8 modelopt checkpoint, on CPU."""

import pytest
import torch

from axolotl.core.publishing import pin_base
from axolotl.core.publishing.adapters import ADAPTER_WEIGHTS, validate_adapter
from axolotl.core.publishing.pinned_base import safetensors_layout
from axolotl.core.publishing.rank_builder import MergeContext

from tests.core.publishing.nvfp4_moe import assert_bytes_equal, load_checkpoint
from tests.core.publishing.qwen3_5_moe_mixed import (
    ALPHA,
    LAYERS,
    TEXT,
    E,
    I,
    R,
    build_base,
    lora_modules,
    save_adapter,
)
from tests.core.publishing.test_publishing_step import (
    offline_chain,
    reconstruct,
    run_ranks,
)

pytest.importorskip("torchao")

SCALING = ALPHA / R


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("qwen3_5_moe")
    base = build_base(root / "base")
    adapters = [save_adapter(root / f"adapter-{step}", step) for step in range(3)]
    return root, base, adapters


def test_mixed_checkpoint_inventory(tmp_path, setup):
    _, base_dir, adapters = setup
    base = pin_base(base_dir, tmp_path / "work")
    assert base.format == "modelopt" and base.num_experts == E
    targets = validate_adapter(adapters[0], base.layout, num_experts=E)
    assert targets.linear == len(lora_modules())
    assert targets.expert_groups == 2 * len(LAYERS)

    context = MergeContext(
        base,
        {"r": R, "lora_alpha": ALPHA, "nvfp4_merge_aware": True},
        safetensors_layout(adapters[0] / ADAPTER_WEIGHTS),
    )
    inventory = context.inventory
    formats = {
        name.removesuffix(".weight").split(".", 4)[-1]: fmt
        for name, (fmt, _) in inventory.quantized.items()
        if ".layers.1." in name
    }
    assert formats == {
        "self_attn.q_proj": "modelopt_fp8",
        "self_attn.k_proj": "modelopt_fp8",
        "self_attn.v_proj": "modelopt_fp8",
        "self_attn.o_proj": "modelopt_fp8",
        "mlp.shared_expert.gate_proj": "nvfp4",
        "mlp.shared_expert.up_proj": "nvfp4",
        "mlp.shared_expert.down_proj": "nvfp4",
    }
    stem = f"{TEXT}.layers.1"
    # FP8 keeps its checkpoint scale; NVFP4 rewrites its block scale
    assert f"{stem}.self_attn.q_proj.weight_scale" not in inventory.targeted
    assert f"{stem}.mlp.shared_expert.up_proj.weight_scale" in inventory.targeted
    untargeted = {name for names in inventory.files.values() for name in names}
    for name in (
        "lm_head.weight",
        "lm_head.weight_scale",
        f"{stem}.self_attn.q_proj.weight_scale",
        f"{stem}.self_attn.q_proj.input_scale",
        f"{stem}.mlp.shared_expert.up_proj.input_scale",
        "mtp.layers.0.mlp.experts.gate_up_proj",
    ):
        assert name in untargeted, name
    assert any(name.startswith("model.visual.") for name in untargeted)


@pytest.mark.parametrize(("world", "ep"), [(2, None), (4, 2)])
def test_rank_builds_match_the_offline_merge(tmp_path, setup, world, ep):
    _, base, adapters = setup
    results = run_ranks(
        tmp_path,
        world,
        base=base,
        adapters=adapters,
        steps=list(range(len(adapters))),
        work=tmp_path / "work",
        published=tmp_path / "published",
        ep=ep,
        expert_chunk=1,
    )
    assert [m["full"] for m in results[0]] == [True, False, False]
    published = [tmp_path / "published" / m["identity"] for m in results[0]]
    offline = offline_chain(tmp_path, base, adapters)
    ours, theirs = reconstruct(published), reconstruct(offline)
    for mine, reference in zip(ours, theirs, strict=True):
        assert_bytes_equal(mine, reference)

    base_weights = load_checkpoint(base)
    last = ours[-1]
    stem = f"{TEXT}.layers.1"
    for name in (
        f"{stem}.self_attn.q_proj.weight",
        f"{TEXT}.layers.0.linear_attn.in_proj_qkv.weight",
        f"{stem}.mlp.shared_expert.gate_proj.weight",
        f"{stem}.mlp.experts.3.down_proj.weight",
    ):
        assert not torch.equal(
            last[name].view(torch.uint8), base_weights[name].view(torch.uint8)
        ), name
    for name in (
        f"{stem}.self_attn.q_proj.weight_scale",
        f"{stem}.self_attn.q_proj.input_scale",
        "lm_head.weight",
        "model.visual.merger.linear_fc1.weight",
    ):
        assert torch.equal(last[name], base_weights[name]), name


def _merge_aware_snap(weight, lora_a, lora_b, pts):
    """The merged weight's NVFP4 bytes as the merge-aware training forward snaps it."""
    from axolotl.integrations.kernels.libs.sonicmoe.nvfp4_quant import (
        quantize_nvfp4_merge,
    )

    if weight.ndim == 3:
        e, rows, columns = weight.shape
        a = lora_a.reshape(e, R, columns).float()
        b = lora_b.reshape(rows, R, e).permute(2, 0, 1).float()
        delta = torch.bmm(b, a)
    else:
        delta = lora_b.float() @ lora_a.float()
    w_eff = (weight.float() + delta * SCALING).to(weight.dtype)
    return quantize_nvfp4_merge(w_eff.float(), pts, scale_mode="fresh")


def test_published_nvfp4_matches_the_merge_aware_forward(tmp_path, setup):
    """Shared expert and routed experts: published bytes are what training quantized."""
    pytest.importorskip("triton")  # the expert loader package imports it
    from safetensors.torch import load_file
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    from axolotl.integrations.kernels.libs.scattermoe_lora.nvfp4_moe_loading import (
        fuse_nvfp4_experts,
    )

    _, base, adapters = setup
    results = run_ranks(
        tmp_path,
        2,
        base=base,
        adapters=adapters[:1],
        steps=[0],
        work=tmp_path / "work",
        published=tmp_path / "published",
    )
    published = load_checkpoint(tmp_path / "published" / results[0][0]["identity"])
    base_weights = load_checkpoint(base)
    lora = load_file(adapters[0] / ADAPTER_WEIGHTS)
    prefix = "base_model.model."

    for layer in LAYERS:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            stem = f"{TEXT}.layers.{layer}.mlp.shared_expert.{proj}"
            pts = base_weights[f"{stem}.weight_scale_2"]
            # the loader's dequantized base weight, as the training forward sees it
            weight = NVFP4Tensor(
                base_weights[f"{stem}.weight"],
                base_weights[f"{stem}.weight_scale"],
                16,
                torch.bfloat16,
                per_tensor_scale=pts.view(1, 1, 1),
            ).dequantize(torch.bfloat16)
            packed, scale = _merge_aware_snap(
                weight,
                lora[f"{prefix}{stem}.lora_A.weight"],
                lora[f"{prefix}{stem}.lora_B.weight"],
                pts,
            )
            assert torch.equal(published[f"{stem}.weight"], packed), stem
            assert torch.equal(
                published[f"{stem}.weight_scale"].view(torch.uint8),
                scale.view(torch.uint8),
            ), stem
            assert torch.equal(published[f"{stem}.weight_scale_2"], pts), stem

        experts = f"{TEXT}.layers.{layer}.mlp.experts"
        for projs, lora_stem in (
            (("gate_proj", "up_proj"), f"{experts}.base_layer"),
            (("down_proj",), experts),
        ):
            parts = [
                {
                    key: [
                        base_weights[f"{experts}.{e}.{proj}.{leaf}"] for e in range(E)
                    ]
                    for key, leaf in (
                        ("qd", "weight"),
                        ("sc", "weight_scale"),
                        ("pts", "weight_scale_2"),
                    )
                }
                for proj in projs
            ]
            nvfp4 = fuse_nvfp4_experts(parts)
            pts = nvfp4.per_tensor_scale.reshape(E, 1, 1)
            packed, scale = _merge_aware_snap(
                nvfp4.dequantize(torch.bfloat16),
                lora[f"{prefix}{lora_stem}.lora_A.weight"],
                lora[f"{prefix}{lora_stem}.lora_B.weight"],
                pts,
            )
            rows = 0
            for proj in projs:
                count = I if proj != "down_proj" else packed.shape[1]
                for e in range(E):
                    stem = f"{experts}.{e}.{proj}"
                    assert torch.equal(
                        published[f"{stem}.weight"], packed[e, rows : rows + count]
                    ), stem
                    assert torch.equal(
                        published[f"{stem}.weight_scale"].view(torch.uint8),
                        scale[e, rows : rows + count].view(torch.uint8),
                    ), stem
                    assert torch.equal(
                        published[f"{stem}.weight_scale_2"], pts[e].reshape(())
                    ), stem
                rows += count
