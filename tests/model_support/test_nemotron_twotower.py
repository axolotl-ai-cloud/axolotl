"""CPU tests for the in-tree Nemotron-Labs-TwoTower block-diffusion model.

The parity oracle below is an independent, token-by-token transcription of the
math in NVIDIA's reference ``modeling_nemotron_twotower.py`` /
``modeling_nemotron_h.py`` (sequential Mamba-2 recurrence, explicit attention,
per-expert MoE loop with the fp64 sigmoid router). The reference file itself is
never imported: it is remote code and needs CUDA-only mamba kernels.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from packaging.version import Version
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file

from axolotl.model_support.nemotron_twotower import (
    CONFIG_CLASS_PATH,
    NemotronTwoTowerSupport,
    _check_context_frozen,
    _matches_cfg,
    _model_class,
    _validate,
    scope_target_modules,
    scope_target_parameters,
    scope_trainable_token_indices,
)
from axolotl.model_support.nemotron_twotower.configuration import (
    NemotronHTwoTowerConfig,
)
from axolotl.model_support.nemotron_twotower.modeling import (
    NemotronHTwoTowerForDiffusion,
)
from axolotl.model_support.nemotron_twotower.ssm import causal_conv1d_with_state
from axolotl.model_support.profile import ModelHookContext
from axolotl.utils.dict import DictDefault

MASK = 3
PATTERN = (
    "linear_attention",
    "moe",
    "full_attention",
    "moe",
    "linear_attention",
    "moe",
)


@pytest.fixture(autouse=True)
def _few_threads():
    # Tiny matmuls: oversubscribed intra-op threads cost ~50x on a busy host.
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 4))
    yield
    torch.set_num_threads(previous)


def tiny_config(**overrides) -> NemotronHTwoTowerConfig:
    values = dict(
        vocab_size=257,
        hidden_size=64,
        num_hidden_layers=len(PATTERN),
        layers_block_type=list(PATTERN),
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        mamba_num_heads=8,
        mamba_head_dim=8,
        n_groups=2,
        ssm_state_size=16,
        conv_kernel=4,
        chunk_size=16,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        moe_shared_expert_intermediate_size=48,
        intermediate_size=32,
        routed_scaling_factor=2.5,
        norm_topk_prob=True,
        mask_token_id=MASK,
        time_step_limit=(0.0, float("inf")),
    )
    values.update(overrides)
    return NemotronHTwoTowerConfig(**values)


def tiny_model(seed: int = 0, **overrides) -> NemotronHTwoTowerForDiffusion:
    torch.manual_seed(seed)
    model = NemotronHTwoTowerForDiffusion(tiny_config(**overrides)).float()
    # Non-trivial router bias, A/D/dt and norms so the oracle exercises them.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith(("norm.weight", "norm_f.weight")):
                parameter.uniform_(0.5, 1.5)
            elif name.endswith(("mixer.D", "mixer.dt_bias")):
                parameter.uniform_(-1.0, 1.0)
            elif name.endswith("mixer.A_log"):
                parameter.uniform_(-1.0, 1.5)
            elif name.endswith("conv1d.bias"):
                parameter.uniform_(-0.2, 0.2)
        for buffer_name, buffer in model.named_buffers():
            if buffer_name.endswith("e_score_correction_bias"):
                buffer.uniform_(-0.05, 0.05)
    return model


# ----------------------------------------------------------------------- oracle
def _rmsnorm(x, weight, eps):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (weight.float() * xf).to(x.dtype)


def _gated_rmsnorm(y, z, weight, eps, group_size):
    # mamba_ssm rmsnorm_fn(..., z, norm_before_gate=False): gate first, groupwise norm
    h = y.float() * F.silu(z.float())
    shape = h.shape
    h = h.view(*shape[:-1], -1, group_size)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return (h.view(shape) * weight.float()).to(y.dtype)


def _oracle_mamba(mixer, h, conv_init=None, ssm_init=None):
    """Sequential Mamba-2: explicit conv window and per-token SSM update."""
    batch, length, _ = h.shape
    d_inner, conv_dim, heads = mixer.intermediate_size, mixer.conv_dim, mixer.num_heads
    groups, d_state, head_dim = mixer.n_groups, mixer.ssm_state_size, mixer.head_dim
    kernel = mixer.conv_kernel_size
    proj = h @ mixer.in_proj.weight.T
    z, xbc, dt = torch.split(proj, [d_inner, conv_dim, heads], dim=-1)
    history = (
        conv_init
        if conv_init is not None
        else xbc.new_zeros(batch, conv_dim, kernel - 1)
    )
    sequence = torch.cat([history, xbc.transpose(1, 2)], dim=-1)
    weight = mixer.conv1d.weight[:, 0, :]
    conv = torch.stack(
        [
            F.silu((sequence[..., t : t + kernel] * weight).sum(-1) + mixer.conv1d.bias)
            for t in range(length)
        ],
        dim=1,
    )
    x, b, c = torch.split(conv, [d_inner, groups * d_state, groups * d_state], dim=-1)
    x = x.reshape(batch, length, heads, head_dim).double()
    b = (
        b.reshape(batch, length, groups, d_state)
        .repeat_interleave(heads // groups, 2)
        .double()
    )
    c = (
        c.reshape(batch, length, groups, d_state)
        .repeat_interleave(heads // groups, 2)
        .double()
    )
    a = -torch.exp(mixer.A_log.double())
    dt = F.softplus(dt.double() + mixer.dt_bias.double())
    state = (
        ssm_init.double()
        if ssm_init is not None
        else torch.zeros(batch, heads, head_dim, d_state, dtype=torch.float64)
    )
    ys = []
    for t in range(length):
        decay = torch.exp(dt[:, t] * a)[..., None, None]
        state = (
            state * decay
            + dt[:, t, :, None, None] * x[:, t, :, :, None] * b[:, t, :, None, :]
        )
        ys.append(
            (state * c[:, t, :, None, :]).sum(-1) + mixer.D.double()[:, None] * x[:, t]
        )
    y = torch.stack(ys, dim=1).reshape(batch, length, d_inner).to(h.dtype)
    y = _gated_rmsnorm(
        y, z, mixer.norm.weight, mixer.norm.variance_epsilon, d_inner // groups
    )
    return y @ mixer.out_proj.weight.T, sequence[..., -(kernel - 1) :], state


def _oracle_attention(mixer, h, ctx_k=None, ctx_v=None, causal=False):
    batch, length, _ = h.shape
    head_dim = mixer.head_dim
    q = (h @ mixer.q_proj.weight.T).view(batch, length, -1, head_dim).transpose(1, 2)
    k = (h @ mixer.k_proj.weight.T).view(batch, length, -1, head_dim).transpose(1, 2)
    v = (h @ mixer.v_proj.weight.T).view(batch, length, -1, head_dim).transpose(1, 2)
    block_k, block_v = k, v
    if ctx_k is not None:
        k = torch.cat([ctx_k, k], dim=2)
        v = torch.cat([ctx_v, v], dim=2)
    groups = q.shape[1] // k.shape[1]
    k_rep = k.repeat_interleave(groups, 1)
    v_rep = v.repeat_interleave(groups, 1)
    scores = (q @ k_rep.transpose(-1, -2)) / math.sqrt(head_dim)
    if causal:
        mask = torch.ones(length, k.shape[2], dtype=torch.bool).tril(
            k.shape[2] - length
        )
        scores = scores.masked_fill(~mask, float("-inf"))
    out = scores.softmax(-1) @ v_rep
    out = out.transpose(1, 2).reshape(batch, length, -1) @ mixer.o_proj.weight.T
    return out, block_k, block_v


def _oracle_moe(moe, h, config):
    shape = h.shape
    x = h.reshape(-1, shape[-1])
    scores = torch.sigmoid(x.double() @ moe.gate.weight.double().T)
    choice = scores + moe.gate.e_score_correction_bias.double()
    index = choice.topk(config.num_experts_per_tok, dim=-1).indices
    weights = scores.gather(1, index)
    if config.norm_topk_prob:
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
    weights = (weights * config.routed_scaling_factor).float()
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for token in range(x.shape[0]):
        for slot in range(index.shape[1]):
            expert = int(index[token, slot])
            hidden = F.relu(x[token] @ moe.experts.up_proj[expert].T) ** 2
            out[token] += weights[token, slot] * (
                hidden @ moe.experts.down_proj[expert].T
            )
    shared = moe.shared_experts
    shared_out = (F.relu(x @ shared.up_proj.weight.T) ** 2) @ shared.down_proj.weight.T
    return (out.to(h.dtype) + shared_out).view(shape)


def _oracle_time(model, t):
    half = model.config.time_embed_frequency_size // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
    args = (t * model.config.time_embed_max_period)[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    mlp = model.t_embedder.mlp
    emb = F.silu(emb @ mlp[0].weight.T + mlp[0].bias) @ mlp[2].weight.T + mlp[2].bias
    return F.silu(emb) @ model.t_block[1].weight.T + model.t_block[1].bias


def oracle_canvas_logits(model, prompt, canvas):
    """Reference math for one document: clean ``prompt`` [S], noisy ``canvas`` [W]."""
    config = model.config
    types = config.layers_block_type
    with torch.no_grad():
        # context tower (generate_mask_diffusion -> _build_context_cache)
        tower = model.context_tower
        hidden = tower.embeddings(prompt[None])
        states = {}
        for index, block in enumerate(tower.layers):
            residual = hidden
            h = _rmsnorm(hidden, block.norm.weight, block.norm.variance_epsilon)
            if types[index] == "linear_attention":
                h, conv, ssm = _oracle_mamba(block.mixer, h)
                states[index] = (conv, ssm)
            elif types[index] == "full_attention":
                h, k, v = _oracle_attention(block.mixer, h, causal=True)
                states[index] = (k, v)
            else:
                h = _oracle_moe(block.mixer, h, config)
            hidden = residual + h
        # denoiser step (_run_denoiser_step_diffusion)
        t = (canvas == MASK).float().mean()[None]
        t_emb = _oracle_time(model, t)
        tower = model.denoiser_tower
        hidden = tower.embeddings(canvas[None])
        for index, block in enumerate(tower.layers):
            residual = hidden
            table = model.scale_shift_tables[index][None] + t_emb.reshape(1, 3, -1)
            shift, scale, gate = table[:, 0], table[:, 1], table[:, 2]
            eps = block.norm.variance_epsilon
            if types[index] in ("linear_attention", "full_attention"):
                h = hidden * (1 + scale[:, None]) + shift[:, None]
                h = _rmsnorm(h, block.norm.weight, eps)
            else:
                h = _rmsnorm(hidden, block.norm.weight, eps)
                h = h * (1 + scale[:, None]) + shift[:, None]
            if types[index] == "linear_attention":
                conv, ssm = states[index]
                h, _, _ = _oracle_mamba(block.mixer, h, conv, ssm)
            elif types[index] == "full_attention":
                k, v = states[index]
                h, _, _ = _oracle_attention(block.mixer, h, k, v)
            else:
                h = _oracle_moe(block.mixer, h, config)
            hidden = residual + gate[:, None] * h
        hidden = _rmsnorm(hidden, tower.norm_f.weight, tower.norm_f.variance_epsilon)
        return (hidden @ model.lm_head.weight.T)[0]


def packed_batch():
    """Two documents in one row: prompts of 22 and 30 tokens (multi-chunk), 8-wide canvases."""
    generator = torch.Generator().manual_seed(5)
    lengths = ((22, 8), (30, 8))
    ids, positions, canvas = [], [], []
    for prompt_len, width in lengths:
        tokens = torch.randint(10, 257, (prompt_len + width,), generator=generator)
        tokens[prompt_len + 1 : prompt_len + 4] = MASK
        ids.append(tokens)
        positions.append(torch.arange(prompt_len + width))
        canvas.append(torch.cat([torch.zeros(prompt_len), torch.ones(width)]).bool())
    return (
        torch.cat(ids)[None],
        torch.cat(positions)[None],
        torch.cat(canvas)[None],
        lengths,
    )


def _doc_slices(lengths):
    offset = 0
    for prompt_len, width in lengths:
        yield offset, offset + prompt_len, offset + prompt_len + width
        offset += prompt_len + width


# ------------------------------------------------------------------------ tests
def test_conv_with_state_matches_full_sequence_conv():
    torch.manual_seed(1)
    weight, bias = torch.randn(6, 1, 4), torch.randn(6)
    x = torch.randn(2, 11, 6)
    full, _ = causal_conv1d_with_state(x, weight, bias, "silu")
    head, state = causal_conv1d_with_state(x[:, :7], weight, bias, "silu")
    tail, _ = causal_conv1d_with_state(x[:, 7:], weight, bias, "silu", state)
    torch.testing.assert_close(torch.cat([head, tail], 1), full)
    reference = F.silu(F.conv1d(x.transpose(1, 2), weight, bias, padding=3, groups=6))
    torch.testing.assert_close(full, reference[..., :11].transpose(1, 2))


def test_denoiser_logits_match_reference_math():
    model = tiny_model().eval()
    ids, positions, canvas, lengths = packed_batch()
    logits = model(ids, position_ids=positions, axolotl_canvas_mask=canvas).logits[0]
    for start, canvas_start, end in _doc_slices(lengths):
        expected = oracle_canvas_logits(
            model, ids[0, start:canvas_start], ids[0, canvas_start:end]
        )
        torch.testing.assert_close(
            logits[canvas_start:end], expected, rtol=2e-4, atol=2e-4
        )
        assert torch.count_nonzero(logits[start:canvas_start]) == 0


def test_packing_is_document_local():
    model = tiny_model().eval()
    ids, positions, canvas, lengths = packed_batch()
    packed = model(ids, position_ids=positions, axolotl_canvas_mask=canvas).logits[0]
    for start, _canvas_start, end in _doc_slices(lengths):
        alone = model(
            ids[:, start:end],
            position_ids=positions[:, start:end],
            axolotl_canvas_mask=canvas[:, start:end],
        ).logits[0]
        torch.testing.assert_close(packed[start:end], alone, rtol=1e-5, atol=1e-5)


def test_trailing_padding_segments_are_ignored():
    model = tiny_model().eval()
    ids, positions, canvas, _ = packed_batch()
    padded_ids = F.pad(ids, (0, 5))
    padded_positions = F.pad(positions, (0, 5))
    padded_canvas = F.pad(canvas, (0, 5))
    base = model(ids, position_ids=positions, axolotl_canvas_mask=canvas).logits
    padded = model(
        padded_ids, position_ids=padded_positions, axolotl_canvas_mask=padded_canvas
    ).logits
    torch.testing.assert_close(padded[:, : ids.shape[1]], base)


def test_selected_logits_match_full_logits():
    model = tiny_model().eval()
    ids, positions, canvas, _ = packed_batch()
    full = model(ids, position_ids=positions, axolotl_canvas_mask=canvas).logits
    rows = torch.tensor([[0, 0], [0, -1]])
    cols = torch.tensor([[23, 25], [61, -1]])
    selected = model(
        ids,
        position_ids=positions,
        axolotl_canvas_mask=canvas,
        axolotl_selected_logits=(rows, cols),
    )
    assert selected.axolotl_selected_logits
    torch.testing.assert_close(selected.logits[0], full[0, [23, 25]])
    torch.testing.assert_close(selected.logits[1, 0], full[0, 61])


def test_canvas_must_be_document_suffix():
    model = tiny_model().eval()
    ids, positions, canvas, _ = packed_batch()
    broken = canvas.clone()
    broken[0, 5] = True
    with pytest.raises(ValueError, match="contiguous suffix"):
        model(ids, position_ids=positions, axolotl_canvas_mask=broken)
    with pytest.raises(ValueError, match="axolotl_canvas_mask"):
        model(ids, position_ids=positions)


def test_time_is_masked_fraction_and_can_be_overridden():
    model = tiny_model().eval()
    ids, positions, canvas, _ = packed_batch()
    automatic = model(ids, position_ids=positions, axolotl_canvas_mask=canvas).logits
    explicit = model(
        ids,
        position_ids=positions,
        axolotl_canvas_mask=canvas,
        timesteps=torch.tensor([3 / 8, 3 / 8]),
    ).logits
    other = model(
        ids,
        position_ids=positions,
        axolotl_canvas_mask=canvas,
        timesteps=torch.tensor([0.9, 0.9]),
    ).logits
    torch.testing.assert_close(automatic, explicit)
    assert not torch.allclose(automatic, other)
    per_token = torch.full(ids.shape, 0.9)
    from_map = model(
        ids,
        position_ids=positions,
        axolotl_canvas_mask=canvas,
        axolotl_canvas_times=per_token,
    ).logits
    torch.testing.assert_close(from_map, other)
    with pytest.raises(ValueError, match="position_ids"):
        model(ids, axolotl_canvas_mask=canvas)


def test_context_tower_gets_no_gradient_and_checkpointing_matches():
    ids, positions, canvas, _ = packed_batch()
    grads = []
    for checkpointing in (False, True):
        model = tiny_model().train()
        if checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        out = model(ids, position_ids=positions, axolotl_canvas_mask=canvas)
        out.logits[canvas].float().logsumexp(-1).sum().backward()
        grads.append(
            {
                name: p.grad.clone()
                for name, p in model.named_parameters()
                if p.grad is not None
            }
        )
        assert not any(name.startswith("context") for name in grads[-1])
        assert any(
            name.startswith("denoiser_tower.layers.1.mixer.experts")
            for name in grads[-1]
        )
        assert "t_block.1.weight" in grads[-1]
    assert grads[0].keys() == grads[1].keys()
    for name in grads[0]:
        torch.testing.assert_close(grads[0][name], grads[1][name], rtol=1e-5, atol=1e-6)


# ------------------------------------------------------------- checkpoint layout
def _checkpoint_state_dict(model):
    """Our state dict in the released checkpoint's per-expert key layout."""
    state = {}
    for name, tensor in model.state_dict().items():
        match = re.fullmatch(r"(.*\.mixer\.experts)\.(up_proj|down_proj)", name)
        if match:
            for expert in range(tensor.shape[0]):
                state[f"{match[1]}.{expert}.{match[2]}.weight"] = tensor[expert].clone()
        else:
            state[name] = tensor.clone()
    return state


def _pattern(key: str) -> str:
    return re.sub(r"\.\d+(?=\.|$)", ".N", key)


# Key patterns of nvidia/Nemotron-Labs-TwoTower-30B-A3B-Base-BF16 @ 35b6498 (index json),
# minus ``context_lm_head.weight`` (AR-mode head; ignored on load).
RELEASED_KEY_PATTERNS = {
    f"{tower}.{suffix}"
    for tower in ("context_tower", "denoiser_tower")
    for suffix in (
        "embeddings.weight",
        "norm_f.weight",
        "layers.N.norm.weight",
        "layers.N.mixer.A_log",
        "layers.N.mixer.D",
        "layers.N.mixer.conv1d.bias",
        "layers.N.mixer.conv1d.weight",
        "layers.N.mixer.dt_bias",
        "layers.N.mixer.in_proj.weight",
        "layers.N.mixer.norm.weight",
        "layers.N.mixer.out_proj.weight",
        "layers.N.mixer.q_proj.weight",
        "layers.N.mixer.k_proj.weight",
        "layers.N.mixer.v_proj.weight",
        "layers.N.mixer.o_proj.weight",
        "layers.N.mixer.gate.weight",
        "layers.N.mixer.gate.e_score_correction_bias",
        "layers.N.mixer.experts.N.up_proj.weight",
        "layers.N.mixer.experts.N.down_proj.weight",
        "layers.N.mixer.shared_experts.up_proj.weight",
        "layers.N.mixer.shared_experts.down_proj.weight",
    )
} | {
    "lm_head.weight",
    "scale_shift_tables.N",
    "t_block.N.weight",
    "t_block.N.bias",
    "t_embedder.mlp.N.weight",
    "t_embedder.mlp.N.bias",
}


def test_state_dict_matches_released_checkpoint_layout():
    model = tiny_model()
    patterns = {_pattern(key) for key in _checkpoint_state_dict(model)}
    assert patterns == RELEASED_KEY_PATTERNS


RELEASED_META = Path("/mnt/nasdata/s1-taxonomy/twotower_meta")


@pytest.mark.skipif(
    not (RELEASED_META / "model.safetensors.index.json").is_file(),
    reason="released checkpoint metadata not downloaded",
)
def test_full_size_keys_match_released_index():
    config = NemotronHTwoTowerConfig.from_pretrained(RELEASED_META)
    with torch.device("meta"):
        model = NemotronHTwoTowerForDiffusion(config)
    ours = set(_checkpoint_state_dict(model))
    released = set(
        json.loads((RELEASED_META / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
    )
    assert ours == released - {"context_lm_head.weight"}


def _save_released_format(model, directory: Path):
    state = _checkpoint_state_dict(model)
    state["context_lm_head.weight"] = torch.randn(
        model.config.vocab_size, model.config.hidden_size
    )
    save_file(
        {key: value.contiguous() for key, value in state.items()},
        directory / "model.safetensors",
    )
    config = model.config.to_dict()
    config.update(
        model_type="nemotron_h",
        architectures=["NemotronHTwoTowerForCausalLM"],
        auto_map={
            "AutoConfig": "configuration_nemotron_h.NemotronHConfig",
            "AutoModelForCausalLM": "modeling_nemotron_twotower.NemotronHTwoTowerForCausalLM",
        },
    )
    config.pop("mask_token_id")
    (directory / "config.json").write_text(json.dumps(config))


def test_loads_released_per_expert_checkpoint(tmp_path):
    source = tiny_model(seed=3)
    _save_released_format(source, tmp_path)
    config = NemotronHTwoTowerConfig.from_pretrained(tmp_path)
    assert config.model_type == "nemotron_h_twotower"
    assert config.mask_token_id == MASK
    loaded = _model_class().from_pretrained(
        str(tmp_path), config=config, dtype=torch.float32, trust_remote_code=True
    )
    assert isinstance(loaded, NemotronHTwoTowerForDiffusion)
    expected = source.state_dict()
    actual = loaded.state_dict()
    assert expected.keys() == actual.keys()
    for name, tensor in expected.items():
        torch.testing.assert_close(actual[name], tensor, msg=name)


# --------------------------------------------------------------- run config hooks
def test_cfg_matcher_and_config_class_hook(tmp_path):
    assert _matches_cfg(
        DictDefault(base_model="nvidia/Nemotron-Labs-TwoTower-30B-A3B-Base-BF16")
    )
    assert not _matches_cfg(
        DictDefault(base_model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["NemotronHTwoTowerForCausalLM"]})
    )
    assert _matches_cfg(DictDefault(base_model=str(tmp_path)))
    cfg = DictDefault(base_model=str(tmp_path))
    NemotronTwoTowerSupport().pre_config_load(cfg)
    assert cfg.cls_model_config == CONFIG_CLASS_PATH


def test_lora_targets_are_scoped_to_the_denoiser():
    config = tiny_config()
    regex = scope_target_modules(["q_proj", "in_proj", "up_proj"])
    assert re.fullmatch(regex, "denoiser_tower.layers.2.mixer.q_proj")
    assert re.fullmatch(regex, "denoiser_tower.layers.1.mixer.shared_experts.up_proj")
    assert not re.fullmatch(regex, "context_tower.layers.2.mixer.q_proj")
    assert scope_target_parameters(["mixer.experts.up_proj"], config) == [
        "denoiser_tower.layers.1.mixer.experts.up_proj",
        "denoiser_tower.layers.3.mixer.experts.up_proj",
        "denoiser_tower.layers.5.mixer.experts.up_proj",
    ]
    with pytest.raises(ValueError, match="routed experts"):
        scope_target_parameters(["mixer.in_proj.weight"], config)
    assert scope_trainable_token_indices(
        {"embed_tokens": [5, 6], "diffusion_head": [5, 7]}
    ) == {"denoiser_tower.embeddings": [5, 6], "lm_head": [5, 7]}
    with pytest.raises(ValueError, match="context tower"):
        scope_trainable_token_indices({"context_tower.embeddings": [1]})


def test_validate_hook_rewrites_run_config():
    cfg = DictDefault(
        adapter="lora",
        lora_target_modules=["q_proj", "o_proj"],
        lora_target_parameters=["mixer.experts.down_proj"],
        peft_trainable_token_indices={"embed_tokens": [5]},
        diffusion_lm={"canvas_width": 8},
        diffusion_decision={"max_questions_per_canvas": 2},
    )
    _validate(ModelHookContext(cfg=cfg, model_config=tiny_config()))
    assert cfg.attn_implementation == "varlen"
    assert cfg.diffusion_lm["mask_token_id"] == MASK
    assert cfg.lora_target_modules.startswith(r"denoiser_tower\.")
    assert all(
        name.startswith("denoiser_tower.") for name in cfg.lora_target_parameters
    )
    assert cfg.peft_trainable_token_indices == {"denoiser_tower.embeddings": [5]}
    with pytest.raises(ValueError, match="mask_token_id"):
        _validate(
            ModelHookContext(
                cfg=DictDefault(
                    diffusion_lm={"mask_token_id": 100}, diffusion_decision={"a": 1}
                ),
                model_config=tiny_config(),
            )
        )
    with pytest.raises(ValueError, match="diffusion_decision plugin"):
        _validate(
            ModelHookContext(
                cfg=DictDefault(diffusion_lm={"canvas_width": 8}),
                model_config=tiny_config(),
            )
        )
    with pytest.raises(ValueError, match="trainable_token_indices"):
        scope_target_modules(["q_proj", "lm_head"])
    with pytest.raises(ValueError, match="lora_target_linear"):
        _validate(
            ModelHookContext(
                cfg=DictDefault(adapter="lora", lora_target_linear=True),
                model_config=tiny_config(),
            )
        )


def _peft_model(target_modules, target_parameters, token_indices):
    return get_peft_model(
        tiny_model(),
        LoraConfig(
            r=4,
            lora_alpha=4,
            use_rslora=True,
            lora_dropout=0.0,
            target_modules=target_modules,
            target_parameters=target_parameters,
            trainable_token_indices=token_indices,
        ),
    )


def test_lora_and_trainable_tokens_attach_to_the_denoiser_only():
    config = tiny_config()
    ids, positions, canvas, _ = packed_batch()
    # a token present (unmasked) in a canvas, so its embedding row gets a gradient
    seen = int(ids[0, 22])
    model = _peft_model(
        scope_target_modules(
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "in_proj",
                "out_proj",
                "up_proj",
                "down_proj",
            ]
        ),
        scope_target_parameters(
            ["mixer.experts.up_proj", "mixer.experts.down_proj"], config
        ),
        scope_trainable_token_indices(
            {"embed_tokens": [seen, 6], "diffusion_head": [seen, 6]}
        ),
    )
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable
    assert not any(
        ".context_tower." in f".{name}" or "context_lm_head" in name
        for name in trainable
    )
    assert any("mixer.experts" in name and "lora_A" in name for name in trainable)
    assert any("shared_experts.up_proj.lora_A" in name for name in trainable)
    assert any("mixer.in_proj.lora_A" in name for name in trainable)
    assert any("denoiser_tower.embeddings.token_adapter" in name for name in trainable)
    assert any("lm_head.token_adapter" in name for name in trainable)
    _check_context_frozen(ModelHookContext(cfg=DictDefault(), model=model))

    base = tiny_model().eval()
    model.eval()
    torch.testing.assert_close(
        model(ids, position_ids=positions, axolotl_canvas_mask=canvas).logits,
        base(ids, position_ids=positions, axolotl_canvas_mask=canvas).logits,
    )
    model.train()
    out = model(ids, position_ids=positions, axolotl_canvas_mask=canvas)
    out.logits[canvas].logsumexp(-1).sum().backward()
    missing = [
        name
        for name, p in model.named_parameters()
        if p.requires_grad
        and (p.grad is None or not torch.any(p.grad != 0))
        and "lora_A" not in name  # B starts at zero, so A's first gradient is zero
    ]
    assert not missing, missing


def test_unscoped_targets_would_reach_the_context_tower():
    model = _peft_model(["q_proj"], ["mixer.experts.up_proj"], None)
    with pytest.raises(ValueError, match="context tower must stay frozen"):
        _check_context_frozen(ModelHookContext(cfg=DictDefault(), model=model))


# ------------------------------------------------------------- decision training
def test_decision_loss_runs_through_the_trainer():
    from axolotl.integrations.diffusion_decision.training_collator import (
        DecisionTrainingCollator,
    )
    from axolotl.model_support import DiffusionLayout

    from tests.integrations.diffusion_decision.helpers import (
        TrainerHarness,
        make_canvas,
        trainer_spec,
    )

    spec = trainer_spec(DiffusionLayout.FULL_SEQUENCE)
    canvases = [
        make_canvas(
            (11, 12, 13, 14, 15), (20, 21, 22, 23), (1,), "a", allowed=(30, 31, 32)
        ),
        make_canvas((16, 17, 18), (24, 25, 26, 27), (2,), "b", allowed=(30, 31)),
    ]
    inputs = DecisionTrainingCollator(spec)(
        [
            {"canvas": canvas, "source": name}
            for canvas, name in zip(canvases, "ab", strict=True)
        ]
    )
    assert inputs["decision_canvas_mask"].sum() == 8
    trainer = TrainerHarness(
        spec,
        {"labels": {"label_softmax": "both", "brier_weight": 0.1}},
        mask_token_id=MASK,
    )
    model = tiny_model().train()
    seen = []
    original = model.forward

    def recording(*args, **kwargs):
        seen.append(kwargs.get("axolotl_canvas_times"))
        return original(*args, **kwargs)

    model.forward = recording
    for selected in (True, False):
        model.supports_selected_logits = selected
        model.zero_grad(set_to_none=True)
        loss = trainer.compute_loss(model, inputs)
        assert torch.isfinite(loss)
        loss.backward()
        assert model.denoiser_tower.layers[0].mixer.in_proj.weight.grad.abs().sum() > 0
        assert model.context_tower.layers[0].mixer.in_proj.weight.grad is None
    canvas_times = seen[-1][inputs["decision_canvas_mask"]]
    assert canvas_times.numel() == 8 and bool((canvas_times > 0).all())


# ------------------------------------------------------------ axolotl loader e2e
def _tiny_tokenizer(directory: Path, vocab_size: int):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "[INST]": 3}
    vocab.update({f"t{index}": index for index in range(4, vocab_size)})
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", bos_token="<s>", eos_token="</s>"
    ).save_pretrained(directory)


@pytest.mark.skipif(
    Version(torch.__version__.split("+", maxsplit=1)[0]) < Version("2.14.0"),
    reason="TwoTower trains with attn_implementation: varlen, which needs torch >= 2.14",
)
def test_axolotl_loads_released_layout_with_scoped_lora(tmp_path):
    from axolotl.cli.config import prepare_plugins
    from axolotl.loaders import ModelLoader, load_tokenizer
    from axolotl.utils.config import normalize_config, validate_config

    checkpoint = tmp_path / "Nemotron-Labs-TwoTower-tiny"
    checkpoint.mkdir()
    source = tiny_model(seed=3)
    _save_released_format(source, checkpoint)
    _tiny_tokenizer(checkpoint, source.config.vocab_size)
    cfg = DictDefault(
        {
            "base_model": str(checkpoint),
            "plugins": [
                "axolotl.integrations.diffusion_decision.DiffusionDecisionPlugin"
            ],
            "adapter": "lora",
            "lora_r": 4,
            "lora_alpha": 4,
            "peft_use_rslora": True,
            "lora_dropout": 0.0,
            "lora_target_modules": [
                "q_proj",
                "o_proj",
                "in_proj",
                "out_proj",
                "up_proj",
                "down_proj",
            ],
            "lora_target_parameters": [
                "mixer.experts.up_proj",
                "mixer.experts.down_proj",
            ],
            "peft_trainable_token_indices": {
                "embed_tokens": [40, 41],
                "diffusion_head": [40, 41],
            },
            "bf16": False,
            "fp16": False,
            "sequence_len": 256,
            "batch_flattening": True,
            "attn_implementation": "varlen",
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-5,
            "datasets": [
                {
                    "path": "json",
                    "data_files": "unused.jsonl",
                    "type": "diffusion_decision.jsonl",
                }
            ],
            "output_dir": str(tmp_path / "out"),
            "diffusion_lm": {
                "from_causal_lm": False,
                "generate_samples": False,
                "canvas_width": 8,
            },
            "diffusion_decision": {"max_questions_per_canvas": 2},
            "remove_unused_columns": False,
        }
    )
    prepare_plugins(cfg)
    cfg = validate_config(cfg)
    normalize_config(cfg)
    assert cfg.model_config_type == "nemotron_h_twotower"
    assert cfg.diffusion_lm.mask_token_id == MASK
    model, _ = ModelLoader(cfg, load_tokenizer(cfg)).load()
    base = model.get_base_model()
    assert isinstance(base, NemotronHTwoTowerForDiffusion)
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable and not any("context" in name for name in trainable)
    assert (
        sum(".experts." in name for name in trainable) == 12
    )  # 3 MoE layers x up/down x A/B
    assert {
        name.split(".token_adapter")[0] for name in trainable if "token_adapter" in name
    } == {
        "base_model.model.denoiser_tower.embeddings",
        "base_model.model.lm_head",
    }
    state = base.state_dict()
    unwrapped = [name for name in source.state_dict() if name in state]
    assert any(name.startswith("context_tower.layers.1.mixer") for name in unwrapped)
    for name in unwrapped:
        torch.testing.assert_close(
            state[name].cpu(), source.state_dict()[name].cpu(), msg=name
        )
    # zero-initialised LoRA B and token deltas: the adapted model reproduces the source
    ids, positions, canvas, _ = packed_batch()
    model.eval()
    device = next(base.parameters()).device
    adapted = model(
        input_ids=ids.to(device),
        position_ids=positions.to(device),
        axolotl_canvas_mask=canvas.to(device),
    ).logits
    torch.testing.assert_close(
        adapted.cpu(),
        source.eval()(ids, position_ids=positions, axolotl_canvas_mask=canvas).logits,
    )


def test_one_experts_implementation_call_reaches_both_towers(monkeypatch):
    """ScatterMoE plugs in through transformers' ExpertsInterface under the name
    ``scattermoe``; both towers share one config, so a single
    ``set_experts_implementation`` must route all 2 x 3 MoE layers through it.
    The CUDA kernel is replaced by a recording eager call (GPU parity lives in
    tests/e2e/kernels/test_scattermoe_twotower.py)."""
    pytest.importorskip("triton")
    import inspect

    from transformers.integrations import use_experts_implementation
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
    from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHExperts

    from axolotl.integrations.kernels.libs.scattermoe_lora.experts import (
        register_scattermoe_experts,
    )

    register_scattermoe_experts()
    eager = NemotronHExperts.forward.__wrapped__
    calls = []

    def recording(module, *args, **kwargs):
        calls.append(module)
        return eager(module, *args, **kwargs)

    registries = {
        id(registry): registry
        for registry in (
            ALL_EXPERTS_FUNCTIONS,
            inspect.signature(use_experts_implementation)
            .parameters["experts_interface"]
            .default,
        )
    }
    for registry in registries.values():
        monkeypatch.setitem(type(registry)._global_mapping, "scattermoe", recording)

    model = tiny_model().eval()
    ids, positions, canvas, _ = packed_batch()
    expected = model(ids, position_ids=positions, axolotl_canvas_mask=canvas).logits
    model.set_experts_implementation("scattermoe")
    got = model(ids, position_ids=positions, axolotl_canvas_mask=canvas).logits
    torch.testing.assert_close(got, expected)
    experts = {
        id(layer.mixer.experts)
        for tower in (model.context_tower, model.denoiser_tower)
        for layer in tower.layers
        if hasattr(layer.mixer, "experts")
    }
    # context: 2 documents x 3 MoE layers (the last layer is skipped: nothing reads it)
    assert {id(module) for module in calls} == experts - {
        id(model.context_tower.layers[5].mixer.experts)
    }
