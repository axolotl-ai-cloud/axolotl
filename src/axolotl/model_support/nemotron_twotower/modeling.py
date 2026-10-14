"""In-tree Nemotron-Labs-TwoTower block-diffusion model for canvas training.

Re-implements the denoising step of NVIDIA's ``modeling_nemotron_twotower.py``
(``NemotronHTwoTowerForCausalLM._run_denoiser_step_diffusion``) on top of
transformers' native Nemotron-H modules, so axolotl never runs the checkpoint's
remote code and the MoE layers dispatch through transformers' ``ExpertsInterface``
(``experts_implementation: scattermoe`` and PEFT ``target_parameters`` LoRA).

Architecture (both towers are the 52-layer hybrid Mamba-2 / attention / MoE
Nemotron-3-Nano-30B-A3B backbone, with separate weights):

* the context tower runs causally over the clean prefix (the prompt) and yields,
  per layer, the attention K/V and the Mamba-2 conv/SSM state at the prefix end;
  it is frozen and evaluated without autograd;
* the denoiser tower runs once over the noisy block (the canvas): attention
  layers attend bidirectionally over ``[context K/V | block K/V]``, Mamba-2 layers
  chunk-scan the block causally from the context state, and every layer is
  time-conditioned with adaLN-single (``t`` = masked fraction of the block).

Packed batches carry several documents per row (``position_ids`` restart at 0).
Each document is ``prompt + canvas`` and ``axolotl_canvas_mask`` marks the canvas,
which must be a contiguous suffix of its document. Prefix-only segments (padding)
are skipped. Logits exist only on canvas positions.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers.generation import GenerationMixin
from transformers.models.nemotron_h.modeling_nemotron_h import (
    NemotronHModel,
    NemotronHPreTrainedModel,
)
from transformers.utils import ModelOutput

from .configuration import NemotronHTwoTowerConfig
from .ssm import mamba2_mixer_with_state

MAMBA = "linear_attention"
ATTENTION = "full_attention"


class TimestepEmbedder(nn.Module):
    """Sinusoidal + MLP embedding of ``t`` in [0, 1] (reference ``TimestepEmbedder``)."""

    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        max_period: int = 1000,
    ):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding.to(t.dtype)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(
            t * self.max_period, self.frequency_embedding_size
        )
        return self.mlp(t_freq)


@dataclass
class TwoTowerDiffusionOutput(ModelOutput):
    logits: torch.Tensor | None = None
    axolotl_selected_logits: bool | None = None


@dataclass(frozen=True)
class _Document:
    row: int
    start: int
    canvas_start: int
    end: int


@dataclass
class _ContextStates:
    """Per-layer context-tower states, batched over documents."""

    keys: dict[int, torch.Tensor]  # [docs, kv_heads, max_ctx, head_dim]
    values: dict[int, torch.Tensor]
    key_valid: torch.Tensor  # [docs, max_ctx] bool
    conv: dict[int, torch.Tensor]  # [docs, conv_dim, kernel - 1]
    ssm: dict[int, torch.Tensor]  # [docs, heads, head_dim, state] fp32


def _current_device(device: torch.device):
    """Make ``device`` the current CUDA device while a tower runs on it.

    Triton kernels (scattermoe experts) launch on the current device, not on their
    tensors' device; with the towers on two GPUs the second tower would otherwise
    read the other GPU's addresses. Backward needs nothing: autograd runs each
    device's backward on a thread that already has that device current.
    """
    if device.type != "cuda":
        return contextlib.nullcontext()
    return torch.cuda.device(device)


def _modulate(x, shift, scale):
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _attention(q, k, v, mask, *, causal=False):
    """SDPA with GQA expansion done explicitly (portable across torch builds)."""
    groups = q.shape[1] // k.shape[1]
    if groups > 1:
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal)


class NemotronHTwoTowerForDiffusion(NemotronHPreTrainedModel, GenerationMixin):
    """Two-tower Nemotron-H denoiser evaluated over packed ``prompt + canvas`` rows."""

    config: NemotronHTwoTowerConfig
    config_class = NemotronHTwoTowerConfig
    # Checkpoint keys are tower-qualified (``context_tower.*``); no prefix exists to
    # strip or add, and "model" matches none of them.
    base_model_prefix = "model"
    _tied_weights_keys: list[str] = []
    # The context tower's LM head only serves the reference's AR / mock-AR modes.
    # It is not built: denoising never reads it, and PEFT's suffix matching would
    # otherwise give ``context_lm_head`` the trainable-token rows meant for ``lm_head``.
    _keys_to_ignore_on_load_unexpected = [r"mtp.*", r"^context_lm_head\.weight$"]

    supports_diffusion_varlen = True
    supports_selected_logits = True
    requires_canvas_mask = True
    # The MoE layers see per-document prompts and [docs, width] canvases, not the
    # batch's [rows, tokens]: a padding mask built from input_ids does not line up.
    moe_tokens_match_input_ids = False

    def __init__(self, config: NemotronHTwoTowerConfig):
        super().__init__(config)
        hidden = config.hidden_size
        # One config object for both towers: experts_implementation, attention
        # settings and dtype changes reach every layer through it.
        self.context_tower = NemotronHModel(config)
        self.denoiser_tower = NemotronHModel(config)
        self.lm_head = nn.Linear(hidden, config.vocab_size, bias=False)
        self.t_embedder = TimestepEmbedder(
            hidden, config.time_embed_frequency_size, config.time_embed_max_period
        )
        self.t_block = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden, 3 * hidden, bias=True)
        )
        self.scale_shift_tables = nn.ParameterList(
            [
                nn.Parameter(torch.randn(3, hidden) / hidden**0.5)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.gradient_checkpointing = False
        self.post_init()
        # The context tower is the frozen AR backbone in the released recipe; it is
        # evaluated under no_grad, so trainable context weights would never move.
        self.context_tower.requires_grad_(False)

    # ------------------------------------------------------------------ HF API
    @classmethod
    def _can_set_experts_implementation(cls) -> bool:
        # transformers decides this by grepping the class's module source for
        # ``@use_experts_implementation``; ours builds the decorated NemotronHExperts
        # without spelling it, and the towers share this config, so without the
        # override ``set_experts_implementation("scattermoe")`` is a silent no-op.
        return True

    def get_input_embeddings(self):
        return self.denoiser_tower.embeddings

    def set_input_embeddings(self, value):
        self.denoiser_tower.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, *args, **kwargs):
        # PEFT's causal-LM wrapper binds this at construction; generation itself
        # (block-wise confidence unmasking) is not implemented in axolotl.
        raise NotImplementedError(
            "Nemotron-Labs-TwoTower generation is not implemented in axolotl"
        )

    def place_towers_on_devices(self, context_device, denoiser_device):
        """Put the frozen context tower and the denoiser (+ head, time MLP) on two devices.

        Both bf16 towers are ~63 GB each for the 30B-A3B release, so one 80-96 GB GPU
        cannot hold the pair; the reference splits them the same way.
        """
        self.context_tower.to(context_device)
        self.denoiser_tower.to(denoiser_device)
        self.lm_head.to(denoiser_device)
        self.t_embedder.to(denoiser_device)
        self.t_block.to(denoiser_device)
        self.scale_shift_tables.to(denoiser_device)
        return self

    @property
    def _layer_types(self) -> list[str]:
        return list(self.config.layers_block_type)

    @property
    def _dt_limit(self) -> tuple[float, float]:
        limit = tuple(self.config.time_step_limit)
        return (float(limit[0]), float(limit[1]))

    # ---------------------------------------------------------- segmentation
    @staticmethod
    def _documents(
        position_ids: torch.Tensor, canvas_mask: torch.Tensor
    ) -> list[_Document]:
        if position_ids.shape != canvas_mask.shape or position_ids.ndim != 2:
            raise ValueError(
                "position_ids and axolotl_canvas_mask must both be [rows, tokens]"
            )
        rows, length = position_ids.shape
        positions = position_ids.detach().to("cpu", torch.long)
        canvas = canvas_mask.detach().to("cpu", torch.bool)
        new_segment = torch.ones_like(positions, dtype=torch.bool)
        new_segment[:, 1:] = positions[:, 1:] != positions[:, :-1] + 1
        documents: list[_Document] = []
        for row in range(rows):
            starts = new_segment[row].nonzero().flatten().tolist() + [length]
            row_canvas = canvas[row]
            for start, end in zip(starts[:-1], starts[1:], strict=True):
                count = int(row_canvas[start:end].sum())
                if count == 0:
                    continue
                canvas_start = end - count
                if not bool(row_canvas[canvas_start:end].all()):
                    raise ValueError(
                        "TwoTower canvases must be one contiguous suffix of their document "
                        f"(row {row}, tokens {start}:{end})"
                    )
                documents.append(_Document(row, start, canvas_start, end))
        if not documents:
            raise ValueError("TwoTower forward found no canvas positions")
        return documents

    # ---------------------------------------------------------- context tower
    def _context_layer_limit(self) -> int:
        types = self._layer_types
        needed = [i for i, kind in enumerate(types) if kind in (MAMBA, ATTENTION)]
        return needed[-1] + 1 if needed else 0

    @torch.no_grad()
    def _context_states_one(self, prompt_ids: torch.Tensor):
        """Run the frozen context tower over one clean prefix ``[1, S]``."""
        tower = self.context_tower
        types = self._layer_types
        keys, values, conv, ssm = {}, {}, {}, {}
        hidden = tower.embeddings(prompt_ids)
        for index in range(self._context_layer_limit()):
            block = tower.layers[index]
            residual = hidden
            h = block.norm(hidden.to(dtype=block.norm.weight.dtype))
            if types[index] == MAMBA:
                h, conv[index], ssm[index] = mamba2_mixer_with_state(
                    block.mixer, h, dt_limit=self._dt_limit
                )
            elif types[index] == ATTENTION:
                mixer = block.mixer
                batch, length, _ = h.shape
                head_dim = mixer.head_dim
                q = mixer.q_proj(h).view(batch, length, -1, head_dim).transpose(1, 2)
                k = mixer.k_proj(h).view(batch, length, -1, head_dim).transpose(1, 2)
                v = mixer.v_proj(h).view(batch, length, -1, head_dim).transpose(1, 2)
                keys[index], values[index] = k, v
                out = _attention(q, k, v, None, causal=True)
                h = mixer.o_proj(out.transpose(1, 2).reshape(batch, length, -1))
            else:
                h = block.mixer(h)
            hidden = residual + h
        return keys, values, conv, ssm

    def _empty_context(self, device, dtype):
        """States of an empty prefix: zero conv/SSM state, no keys."""
        config = self.config
        keys, values, conv, ssm = {}, {}, {}, {}
        for index, kind in enumerate(self._layer_types[: self._context_layer_limit()]):
            mixer = self.context_tower.layers[index].mixer
            if kind == MAMBA:
                conv[index] = torch.zeros(
                    1,
                    mixer.conv_dim,
                    mixer.conv_kernel_size - 1,
                    device=device,
                    dtype=dtype,
                )
                ssm[index] = torch.zeros(
                    1,
                    mixer.num_heads,
                    mixer.head_dim,
                    mixer.ssm_state_size,
                    device=device,
                    dtype=torch.float32,
                )
            elif kind == ATTENTION:
                shape = (1, config.num_key_value_heads, 0, mixer.head_dim)
                keys[index] = torch.zeros(shape, device=device, dtype=dtype)
                values[index] = torch.zeros(shape, device=device, dtype=dtype)
        return keys, values, conv, ssm

    def _context_experts_parallel(self) -> bool:
        """True when the context tower's routed experts are expert-parallel sharded."""
        return any(
            getattr(module, "num_experts_global", 0) > getattr(module, "num_experts", 0)
            for module in self.context_tower.modules()
        )

    def _ep_context_padding_calls(self, calls: int, device) -> int:
        """Extra context-tower passes this rank runs so every rank runs the same number.

        Under expert parallelism every MoE forward is a collective (token all-to-all)
        across ranks. The context tower runs once per document with a prompt, and ranks
        hold different packed batches, so the counts differ and the rank with fewer
        documents would leave the others blocked in their next all-to-all. The max is
        taken over the whole world, a superset of every EP group.
        """
        if not (
            dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
            and self._context_experts_parallel()
        ):
            return 0
        count = torch.tensor([calls], device=device, dtype=torch.long)
        dist.all_reduce(count, op=dist.ReduceOp.MAX)
        return int(count.item()) - calls

    def _context_states(self, input_ids, documents, device) -> _ContextStates:
        """Context states of every document, batched on ``device`` (the denoiser's)."""
        context_device = self.context_tower.embeddings.weight.device
        dtype = self.context_tower.embeddings.weight.dtype
        per_doc = []
        with _current_device(context_device):
            # agreed before any context pass: each pass is a collective under EP
            calls = sum(doc.canvas_start > doc.start for doc in documents)
            padding = self._ep_context_padding_calls(calls, context_device)
            for doc in documents:
                if doc.canvas_start > doc.start:
                    prompt = input_ids[
                        doc.row : doc.row + 1, doc.start : doc.canvas_start
                    ]
                    per_doc.append(self._context_states_one(prompt.to(context_device)))
                else:
                    per_doc.append(self._empty_context(context_device, dtype))
            for _ in range(padding):
                # discarded one-token pass: only its MoE collectives matter
                self._context_states_one(
                    torch.zeros(1, 1, dtype=torch.long, device=context_device)
                )
        lengths = [doc.canvas_start - doc.start for doc in documents]
        max_ctx = max(lengths)
        key_valid = torch.zeros(
            len(documents), max_ctx, dtype=torch.bool, device=device
        )
        for row, length in enumerate(lengths):
            key_valid[row, :length] = True
        keys, values, conv, ssm = {}, {}, {}, {}
        first = per_doc[0]
        for index in first[0]:
            keys[index] = torch.cat(
                [
                    F.pad(
                        states[0][index], (0, 0, 0, max_ctx - states[0][index].shape[2])
                    )
                    for states in per_doc
                ]
            )
            values[index] = torch.cat(
                [
                    F.pad(
                        states[1][index], (0, 0, 0, max_ctx - states[1][index].shape[2])
                    )
                    for states in per_doc
                ]
            )
        for index in first[2]:
            conv[index] = torch.cat([states[2][index] for states in per_doc])
            ssm[index] = torch.cat([states[3][index] for states in per_doc])
        if device != context_device:
            # towers on separate GPUs (place_towers_on_devices / a device_map)
            keys = {index: value.to(device) for index, value in keys.items()}
            values = {index: value.to(device) for index, value in values.items()}
            conv = {index: value.to(device) for index, value in conv.items()}
            ssm = {index: value.to(device) for index, value in ssm.items()}
        return _ContextStates(keys, values, key_valid, conv, ssm)

    # --------------------------------------------------------- denoiser tower
    def _time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        # The reference casts t to the model dtype before embedding it.
        t = t.to(dtype=self.t_block[1].weight.dtype)
        return self.t_block(self.t_embedder(t))

    def _denoiser_layer(
        self,
        index: int,
        hidden: torch.Tensor,
        t_emb: torch.Tensor,
        attention_mask: torch.Tensor | None,
        context: _ContextStates,
    ) -> torch.Tensor:
        block = self.denoiser_tower.layers[index]
        kind = self._layer_types[index]
        docs, width, size = hidden.shape
        combined = self.scale_shift_tables[index][None] + t_emb.reshape(docs, 3, size)
        shift, scale, gate = combined.unbind(1)
        residual = hidden
        # mcore fuses the mamba/attention norm into the input projection (applied
        # after modulation); MoE keeps a separate pre-norm (before modulation).
        if kind in (MAMBA, ATTENTION):
            h = block.norm(
                _modulate(hidden, shift, scale).to(dtype=block.norm.weight.dtype)
            )
        else:
            h = _modulate(
                block.norm(hidden.to(dtype=block.norm.weight.dtype)), shift, scale
            )
        if kind == MAMBA:
            h, _, _ = mamba2_mixer_with_state(
                block.mixer,
                h,
                conv_state=context.conv[index],
                ssm_state=context.ssm[index],
                dt_limit=self._dt_limit,
            )
        elif kind == ATTENTION:
            mixer = block.mixer
            head_dim = mixer.head_dim
            q = mixer.q_proj(h).view(docs, width, -1, head_dim).transpose(1, 2)
            k = mixer.k_proj(h).view(docs, width, -1, head_dim).transpose(1, 2)
            v = mixer.v_proj(h).view(docs, width, -1, head_dim).transpose(1, 2)
            k = torch.cat([context.keys[index].to(k.dtype), k], dim=2)
            v = torch.cat([context.values[index].to(v.dtype), v], dim=2)
            out = _attention(q, k, v, attention_mask)
            h = mixer.o_proj(out.transpose(1, 2).reshape(docs, width, -1))
        else:
            h = block.mixer(h)
        return residual + gate.unsqueeze(1) * h

    def denoise(
        self,
        canvas_ids: torch.Tensor,
        canvas_valid: torch.Tensor,
        context: _ContextStates,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Denoiser hidden states (after ``norm_f``) for ``canvas_ids`` [docs, width]."""
        if t is None:
            masked = (canvas_ids == self.config.mask_token_id) & canvas_valid
            t = masked.sum(1).float() / canvas_valid.sum(1).clamp_min(1).float()
        t_emb = self._time_embedding(t)
        # Keys are [context | canvas]; padding exists only when documents differ in
        # prompt or canvas length. One host sync here instead of one per layer.
        key_valid = torch.cat([context.key_valid, canvas_valid], dim=1)
        attention_mask = None if bool(key_valid.all()) else key_valid[:, None, None, :]
        hidden = self.denoiser_tower.embeddings(canvas_ids)
        for index in range(self.config.num_hidden_layers):
            layer = partial(self._denoiser_layer, index)
            if self.gradient_checkpointing and self.training:
                hidden = self._gradient_checkpointing_func(
                    layer, hidden, t_emb, attention_mask, context
                )
            else:
                hidden = layer(hidden, t_emb, attention_mask, context)
        return self.denoiser_tower.norm_f(hidden)

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask=None,
        axolotl_canvas_mask: torch.Tensor | None = None,
        axolotl_canvas_times: torch.Tensor | None = None,
        axolotl_selected_logits: tuple[torch.Tensor, torch.Tensor] | None = None,
        timesteps: torch.Tensor | None = None,
        labels=None,
        cce_return_hidden_states: bool = False,
        **_ignored,
    ) -> TwoTowerDiffusionOutput:
        """Denoise every canvas in a packed batch.

        Document boundaries come from ``position_ids`` (restarting at 0), so dense,
        flex and varlen packing metadata (``attention_mask`` / ``diffusion_varlen``)
        are accepted and not needed. The denoiser's ``t`` is, in priority order:
        ``timesteps`` (one value per canvas, in document order),
        ``axolotl_canvas_times`` (``[rows, tokens]``, read at each canvas start; the
        decision trainer passes its sampled corruption time), else the masked
        fraction of the canvas (the reference's inference rule).
        """
        del attention_mask
        if labels is not None:
            raise ValueError("TwoTower diffusion forward does not take labels")
        if cce_return_hidden_states:
            raise NotImplementedError("TwoTower does not support cut_cross_entropy")
        if axolotl_canvas_mask is None:
            raise ValueError(
                "TwoTower forward needs axolotl_canvas_mask marking each document's canvas"
            )
        if input_ids.ndim != 2:
            raise ValueError("TwoTower input_ids must be [rows, tokens]")
        if position_ids is None:
            raise ValueError(
                "TwoTower forward needs document-local position_ids (restarting at 0 "
                "per packed document)"
            )
        documents = self._documents(position_ids, axolotl_canvas_mask)
        # Logits go back to the caller's device: with the towers on two GPUs the
        # trainer's label/supervision tensors stay where the batch was placed.
        output_device = input_ids.device
        device = self.denoiser_tower.embeddings.weight.device
        context = self._context_states(input_ids, documents, device)

        width = max(doc.end - doc.canvas_start for doc in documents)
        rows: list[int] = []
        cols: list[int] = []
        valid: list[bool] = []
        for doc in documents:
            count = doc.end - doc.canvas_start
            rows.extend([doc.row] * width)
            cols.extend(range(doc.canvas_start, doc.canvas_start + count))
            cols.extend([doc.canvas_start] * (width - count))
            valid.extend([True] * count + [False] * (width - count))
        input_ids = input_ids.to(device)
        rows_t = torch.tensor(rows, device=device)
        cols_t = torch.tensor(cols, device=device)
        valid_t = torch.tensor(valid, device=device)
        canvas_ids = torch.where(
            valid_t, input_ids[rows_t, cols_t], self.config.mask_token_id
        ).view(len(documents), width)
        canvas_valid = valid_t.view(len(documents), width)
        if timesteps is None and axolotl_canvas_times is not None:
            starts = torch.tensor(
                [[doc.row, doc.canvas_start] for doc in documents],
                device=axolotl_canvas_times.device,
            )
            timesteps = axolotl_canvas_times[starts[:, 0], starts[:, 1]]
        if timesteps is not None:
            timesteps = timesteps.to(device)
        with _current_device(device):
            hidden = self.denoise(canvas_ids, canvas_valid, context, timesteps)
        flat_hidden = hidden.reshape(-1, hidden.shape[-1])

        if axolotl_selected_logits is not None:
            if (
                not isinstance(axolotl_selected_logits, tuple)
                or len(axolotl_selected_logits) != 2
            ):
                raise TypeError("axolotl_selected_logits must be (rows, positions)")
            sel_rows, sel_positions = axolotl_selected_logits
            if sel_rows.shape != sel_positions.shape or sel_rows.ndim != 2:
                raise ValueError(
                    "selected logits coordinates must be matching [examples, questions]"
                )
            sel_rows, sel_positions = sel_rows.to(device), sel_positions.to(device)
            lookup = torch.full(input_ids.shape, -1, dtype=torch.long, device=device)
            flat_index = torch.arange(len(rows), device=device)
            lookup[rows_t[valid_t], cols_t[valid_t]] = flat_index[valid_t]
            active = (sel_rows >= 0) & (sel_positions >= 0)
            index = lookup[sel_rows.clamp_min(0), sel_positions.clamp_min(0)]
            torch._assert_async(
                ~(active & (index < 0)).any(),
                "selected logits must point at canvas positions",
            )
            logits = self.lm_head(flat_hidden[index.clamp_min(0)])
            return TwoTowerDiffusionOutput(
                logits=logits.to(output_device), axolotl_selected_logits=True
            )

        canvas_logits = self.lm_head(flat_hidden[valid_t])
        logits = canvas_logits.new_zeros(*input_ids.shape, canvas_logits.shape[-1])
        logits[rows_t[valid_t], cols_t[valid_t]] = canvas_logits
        return TwoTowerDiffusionOutput(logits=logits.to(output_device))
