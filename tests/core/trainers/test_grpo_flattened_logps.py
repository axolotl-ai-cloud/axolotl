"""Before/after regression tests for the padding-free GRPO logprob paths.

``_Reference`` carries the previous per-sequence implementations verbatim (with the
CUDA cache flush stubbed). The rewritten methods must match them, values and
gradients, on random mask layouts.
"""

from types import MethodType, SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from trl.trainer.utils import entropy_from_logits, selective_log_softmax

from axolotl.core.trainers.grpo.async_trainer import AsyncGRPOTrainer
from axolotl.core.trainers.grpo.flatten import (
    boundary_index,
    chunk_lengths,
    completion_gather_index,
)

VOCAB = 13


class _Reference:
    """The previous implementations, kept as the oracle."""

    is_fsdp_enabled: bool
    accelerator: Any
    temperature: float

    def _get_per_token_logps_flattened(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        prompt_mask=None,
    ) -> torch.Tensor:
        """Compute per-token log-probs using batch flattening (padding-free).

        Instead of processing padded batches where attention wastes compute on
        padding tokens, this method:
        1. Chunks the batch into sub-batches of ``batch_size`` sequences
        2. For each chunk, flattens non-padding tokens into [1, chunk_tokens]
        3. Uses FlashAttentionKwargs (cu_seq_lens) for varlen attention
        4. Computes selective_log_softmax on the flat logits
        5. Gathers completion logprobs back to (B, logits_to_keep) padded format

        Args:
            prompt_mask: (B, L) mask where 1 = prompt token, 0 = completion/padding.
                Used to determine the exact prompt length per sequence for correct
                logprob gathering. If None, inferred as seq_len - logits_to_keep.

        Chunking prevents OOM when the total flattened sequence is too long
        (e.g., 32 sequences × 2048 tokens = 65K tokens → 20GB logits tensor).

        Requires flash_attention_2 attention implementation.
        """
        if not self.is_fsdp_enabled:
            model = self.accelerator.unwrap_model(model, keep_fp32_wrapper=False)

        device = input_ids.device
        B, L = input_ids.shape
        if batch_size is None:
            batch_size = max(1, B)

        autocast_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        all_logps = torch.zeros(B, logits_to_keep, device=device)

        for chunk_start in range(0, B, batch_size):
            chunk_end = min(chunk_start + batch_size, B)
            chunk_ids = input_ids[chunk_start:chunk_end]
            chunk_mask = attention_mask[chunk_start:chunk_end]
            n = chunk_end - chunk_start

            seq_lens = chunk_mask.sum(dim=1).to(torch.int32)
            total_tokens = seq_lens.sum().item()
            cu_seqlens = torch.zeros(n + 1, dtype=torch.int32, device=device)
            cu_seqlens[1:] = seq_lens.cumsum(0)

            valid = chunk_mask.bool()
            flat_ids = chunk_ids[valid].unsqueeze(0)
            positions = torch.arange(L, device=device).unsqueeze(0).expand(n, L)
            flat_pos = positions[valid].unsqueeze(0)

            with autocast_ctx:
                logits = model(
                    input_ids=flat_ids,
                    position_ids=flat_pos,
                    use_cache=False,
                    cu_seq_lens_q=cu_seqlens,
                    cu_seq_lens_k=cu_seqlens,
                    max_length_q=seq_lens.max().item(),
                    max_length_k=seq_lens.max().item(),
                ).logits
                logits = torch.nan_to_num(logits, nan=0.0)

                # Compute logprobs on the flat shifted tensor
                flat_logits = logits[0, :-1, :] / self.temperature
                flat_targets = flat_ids[0, 1:]
                flat_logps = selective_log_softmax(
                    flat_logits.unsqueeze(0), flat_targets.unsqueeze(0)
                )[0]

                # Mask out cross-sequence boundary positions. In the shifted
                # tensor, position cu_seqlens[i]-1 (for i>0) is where sequence
                # i-1's last token "predicts" sequence i's first token — garbage.
                for boundary in cu_seqlens[1:-1]:
                    idx = boundary.item() - 1
                    if 0 <= idx < flat_logps.size(0):
                        flat_logps[idx] = 0.0

            # Gather completion logprobs per sequence.
            # Use prompt_mask to determine exact prompt length (not logits_to_keep,
            # which is the padded completion dimension and may exceed the actual
            # completion length for shorter sequences).
            for i in range(n):
                slen = seq_lens[i].item()
                abs_i = chunk_start + i  # absolute index in the full batch
                if prompt_mask is not None:
                    plen = int(prompt_mask[abs_i].sum().item())
                else:
                    plen = max(1, slen - logits_to_keep)
                n_compl = slen - plen
                start = cu_seqlens[i].item() + plen - 1
                start = max(0, start)
                actual = min(n_compl, total_tokens - 1 - start)
                if actual > 0:
                    all_logps[chunk_start + i, :actual] = flat_logps[
                        start : start + actual
                    ]

            del logits, flat_logits, flat_logps, flat_ids
            pass  # (old code flushed the CUDA cache here)

        return all_logps

    def _get_per_token_logps_and_entropies_flattened(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        prompt_mask=None,
        compute_entropy=True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Flattened forward pass for training (with gradients).

        Same padding removal as the scoring path, but:
        - Gradients flow through for backward pass
        - Computes entropy alongside logprobs
        - Per-sequence logprob/entropy extraction preserves grad graph
        """
        device = input_ids.device
        B, L = input_ids.shape
        if batch_size is None:
            batch_size = max(1, B)

        autocast_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16)

        # Pre-allocate output containers (will be filled with grad-carrying slices)
        all_logps_list: list[torch.Tensor] = []
        all_entropy_list: list[torch.Tensor] = []

        for chunk_start in range(0, B, batch_size):
            chunk_end = min(chunk_start + batch_size, B)
            chunk_ids = input_ids[chunk_start:chunk_end]
            chunk_mask = attention_mask[chunk_start:chunk_end]
            n = chunk_end - chunk_start

            seq_lens = chunk_mask.sum(dim=1).to(torch.int32)
            cu_seqlens = torch.zeros(n + 1, dtype=torch.int32, device=device)
            cu_seqlens[1:] = seq_lens.cumsum(0)

            valid = chunk_mask.bool()
            flat_ids = chunk_ids[valid].unsqueeze(0)
            positions = torch.arange(L, device=device).unsqueeze(0).expand(n, L)
            flat_pos = positions[valid].unsqueeze(0)

            with autocast_ctx:
                logits = model(
                    input_ids=flat_ids,
                    position_ids=flat_pos,
                    use_cache=False,
                    cu_seq_lens_q=cu_seqlens,
                    cu_seq_lens_k=cu_seqlens,
                    max_length_q=seq_lens.max().item(),
                    max_length_k=seq_lens.max().item(),
                ).logits
                logits = torch.nan_to_num(logits, nan=0.0)

            # Extract logprobs and entropy per-sequence (avoids cross-sequence targets,
            # preserves gradient graph through selective_log_softmax → logits → model)
            for i in range(n):
                slen = seq_lens[i].item()
                abs_i = chunk_start + i
                if prompt_mask is not None:
                    plen = int(prompt_mask[abs_i].sum().item())
                else:
                    plen = max(1, slen - logits_to_keep)
                n_compl = slen - plen
                s = cu_seqlens[i].item()

                if n_compl <= 0:
                    # No completion tokens — append zeros
                    all_logps_list.append(torch.zeros(logits_to_keep, device=device))
                    if compute_entropy:
                        all_entropy_list.append(
                            torch.zeros(logits_to_keep, device=device)
                        )
                    continue

                with autocast_ctx:
                    # Shifted logits and targets for this sequence only
                    seq_logits = logits[0, s + plen - 1 : s + slen - 1, :]
                    seq_logits = seq_logits / self.temperature
                    seq_targets = flat_ids[0, s + plen : s + slen]

                    # Log probs (differentiable)
                    lps = selective_log_softmax(
                        seq_logits.unsqueeze(0), seq_targets.unsqueeze(0)
                    )[0]  # (n_compl,)

                    # Pad to logits_to_keep
                    if n_compl < logits_to_keep:
                        lps = F.pad(lps, (0, logits_to_keep - n_compl))
                    all_logps_list.append(lps[:logits_to_keep])

                    if compute_entropy:
                        ent = entropy_from_logits(seq_logits)  # (n_compl,)
                        if n_compl < logits_to_keep:
                            ent = F.pad(ent, (0, logits_to_keep - n_compl))
                        all_entropy_list.append(ent[:logits_to_keep])

        # Stack per-sequence results into (B, logits_to_keep) tensors
        all_logps = torch.stack(all_logps_list, dim=0)
        all_entropies = (
            torch.stack(all_entropy_list, dim=0) if compute_entropy else None
        )
        return all_logps, all_entropies


class _Model(nn.Module):
    """Logits depend on both token and position so index mistakes change the output."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.tok = nn.Parameter(torch.randn(VOCAB, VOCAB))
        self.pos = nn.Parameter(torch.randn(64, VOCAB))

    def forward(self, input_ids, position_ids, **kwargs):
        assert input_ids.dim() == 2 and input_ids.size(0) == 1
        assert kwargs["cu_seq_lens_q"].dtype == torch.int32
        assert isinstance(kwargs["max_length_q"], int)
        return SimpleNamespace(logits=self.tok[input_ids] + self.pos[position_ids])


def _batch(seed, batch, length, zero_completion_row=False):
    """Left-padded prompts followed by right-padded completions, like GRPO batches."""
    gen = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, VOCAB, (batch, length), generator=gen)
    attention_mask = torch.zeros(batch, length, dtype=torch.long)
    prompt_mask = torch.zeros(batch, length, dtype=torch.long)
    max_completion = 0
    for row in range(batch):
        plen = int(torch.randint(1, length // 2, (1,), generator=gen))
        clen = int(torch.randint(1, length - plen, (1,), generator=gen))
        if zero_completion_row and row == 0:
            clen = 0
        left_pad = int(torch.randint(0, length - plen - clen + 1, (1,), generator=gen))
        attention_mask[row, left_pad : left_pad + plen + clen] = 1
        prompt_mask[row, left_pad : left_pad + plen] = 1
        max_completion = max(max_completion, clen)
    return input_ids, attention_mask, prompt_mask, max_completion


def _trainer(model):
    trainer = object.__new__(AsyncGRPOTrainer)
    trainer.is_fsdp_enabled = True  # skip accelerator.unwrap_model
    trainer.temperature = 0.7
    trainer.model = model
    return trainer


def _reference(trainer):
    ref = SimpleNamespace(is_fsdp_enabled=True, temperature=trainer.temperature)
    ref._get_per_token_logps_flattened = MethodType(
        _Reference._get_per_token_logps_flattened, ref
    )
    ref._get_per_token_logps_and_entropies_flattened = MethodType(
        _Reference._get_per_token_logps_and_entropies_flattened, ref
    )
    return ref


CASES = [
    # seed, batch, length, chunk batch_size, extra logits_to_keep, zero-completion row
    (1, 1, 12, None, 0, False),
    (2, 5, 16, None, 0, False),
    (3, 5, 16, 2, 0, False),
    (4, 6, 20, 4, 3, False),
    (5, 4, 16, 3, 0, True),
    (6, 7, 24, 1, 0, False),
]


@pytest.mark.parametrize(("seed", "batch", "length", "bs", "extra", "zero_row"), CASES)
@pytest.mark.parametrize("with_prompt_mask", [True, False])
def test_scoring_path_matches_reference(
    seed, batch, length, bs, extra, zero_row, with_prompt_mask
):
    model = _Model()
    trainer = _trainer(model)
    input_ids, attention_mask, prompt_mask, max_completion = _batch(
        seed, batch, length, zero_row
    )
    logits_to_keep = max_completion + extra
    pm = prompt_mask if with_prompt_mask else None

    with torch.no_grad():
        got = trainer._get_per_token_logps_flattened(
            model, input_ids, attention_mask, logits_to_keep, bs, pm
        )
        want = _reference(trainer)._get_per_token_logps_flattened(
            model, input_ids, attention_mask, logits_to_keep, bs, pm
        )

    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


# The training path also truncates completions longer than logits_to_keep.
TRAIN_CASES = CASES + [(7, 5, 24, 2, -2, False), (8, 3, 16, None, -1, False)]


@pytest.mark.parametrize(
    ("seed", "batch", "length", "bs", "extra", "zero_row"), TRAIN_CASES
)
@pytest.mark.parametrize("compute_entropy", [True, False])
def test_training_path_matches_reference_with_gradients(
    seed, batch, length, bs, extra, zero_row, compute_entropy
):
    input_ids, attention_mask, prompt_mask, max_completion = _batch(
        seed, batch, length, zero_row
    )
    logits_to_keep = max_completion + extra

    def run(impl_is_new):
        model = _Model()
        trainer = _trainer(model)
        fn = (
            trainer._get_per_token_logps_and_entropies_flattened
            if impl_is_new
            else _reference(trainer)._get_per_token_logps_and_entropies_flattened
        )
        logps, entropies = fn(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            bs,
            prompt_mask,
            compute_entropy=compute_entropy,
        )
        loss = (logps * 1.3).sum()
        if compute_entropy:
            loss = loss + (entropies * 0.7).sum()
        loss.backward()
        return logps, entropies, model.tok.grad.clone(), model.pos.grad.clone()

    got = run(True)
    want = run(False)

    torch.testing.assert_close(got[0], want[0], rtol=1e-4, atol=1e-5)
    if compute_entropy:
        torch.testing.assert_close(got[1], want[1], rtol=1e-4, atol=1e-5)
    else:
        assert got[1] is None and want[1] is None
    torch.testing.assert_close(got[2], want[2], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(got[3], want[3], rtol=1e-4, atol=1e-5)


def test_chunk_lengths_and_gather_index_shapes():
    mask = torch.tensor([[0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0]])
    prompt = torch.tensor([[0, 1, 0, 0, 0], [1, 1, 1, 0, 0], [0, 0, 1, 0, 0]])

    lens, plens, cu = chunk_lengths(mask, prompt, logits_to_keep=4)

    assert (lens, plens, cu) == ([3, 5, 1], [1, 3, 1], [0, 3, 8, 9])
    rows, cols, src = completion_gather_index(lens, plens, cu, 4, row_offset=10)
    # seq 0: 2 completion tokens from shifted positions 0,1; seq 1: 2 from 5,6; seq 2: none
    assert rows.tolist() == [10, 10, 11, 11]
    assert cols.tolist() == [0, 1, 0, 1]
    assert src.tolist() == [0, 1, 5, 6]
    assert boundary_index(cu).tolist() == [2, 7]


def test_chunk_lengths_without_prompt_mask_infers_prompt():
    mask = torch.ones(2, 6, dtype=torch.long)

    lens, plens, cu = chunk_lengths(mask, None, logits_to_keep=4)

    assert lens == [6, 6] and plens == [2, 2] and cu == [0, 6, 12]
