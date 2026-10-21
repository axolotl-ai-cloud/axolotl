"""Multiscale residual vector quantizer with EMA codebooks.

Ported from HazyResearch/ncm (MIT), keeping only the options the released
checkpoints use. Parameter and buffer names match the reference so released
checkpoints load without key remapping.
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


class Phi(nn.Conv1d):
    """Per-level residual refinement: (1 - r) * h + r * conv(h)."""

    def __init__(self, dim: int, ratio: float):
        super().__init__(dim, dim, kernel_size=3, padding=1)
        self.ratio = ratio

    def forward(self, h_BCL: torch.Tensor) -> torch.Tensor:
        return h_BCL * (1 - self.ratio) + super().forward(h_BCL) * self.ratio


def _phi_index(level: int, num_levels: int) -> int:
    # Reference picks the Phi whose tick is nearest to level / (K - 1); not always the identity.
    K = num_levels
    lo = 1 / 3 / K if K == 4 else 1 / 2 / K
    ticks = [lo + (1 - 2 * lo) * j / (K - 1) for j in range(K)] if K > 1 else [lo]
    at = level / max(K - 1, 1)
    return min(range(K), key=lambda j: abs(ticks[j] - at))


def _plan_cascade_strides(total: int, base: int) -> list[int]:
    strides, S = [], int(total)
    while S > 1:
        f = min(base, S)
        while f > 1 and S % f != 0:
            f -= 1
        if f <= 1:
            strides.append(S)
            S = 1
        else:
            strides.append(f)
            S //= f
    return strides or [1]


def _identity_down(dim: int, stride: int) -> nn.Conv1d:
    conv = nn.Conv1d(dim, dim, kernel_size=stride, stride=stride, bias=True)
    with torch.no_grad():
        conv.weight.zero_()
        idx = torch.arange(dim)
        conv.weight[idx, idx, :] = 1.0 / stride
        conv.bias.zero_()
    return conv


def _identity_up(dim: int, stride: int) -> nn.ConvTranspose1d:
    conv = nn.ConvTranspose1d(dim, dim, kernel_size=stride, stride=stride, bias=True)
    with torch.no_grad():
        conv.weight.zero_()
        idx = torch.arange(dim)
        conv.weight[idx, idx, :] = 1.0
        conv.bias.zero_()
    return conv


def _l0_sampler(
    dim: int,
    stride: int,
    cascade_base: int | None,
    inter_norm: bool,
    final_norm: bool,
    up: bool,
) -> nn.Module:
    if cascade_base is None:
        return _identity_up(dim, stride) if up else _identity_down(dim, stride)
    strides = _plan_cascade_strides(stride, cascade_base)
    if up:
        strides = strides[::-1]
    mods: list[nn.Module] = []
    for i, s in enumerate(strides):
        if inter_norm and i > 0:
            mods.append(nn.GroupNorm(8, dim, affine=False))
        mods.append(_identity_up(dim, s) if up else _identity_down(dim, s))
    if final_norm and not up:
        mods.append(nn.GroupNorm(8, dim, affine=False))
    return nn.Sequential(*mods)


class MultiscaleResidualQuantizer(nn.Module):
    """Quantizes a length-L latent into K levels of codes with lengths ``levels``.

    Level k downsamples the current residual to ``levels[k]`` vectors, snaps each to
    its nearest codebook entry, and adds the upsampled (and Phi-refined) result to
    the cumulative reconstruction. Codebooks are updated by EMA, not gradients.
    """

    def __init__(
        self,
        levels: list[int],
        codebook_sizes: list[int],
        dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        quant_resi_ratio: float = 0.5,
        l0_cascade_base: int | None = 4,
        l0_cascade_norm: bool = False,
        l0_cascade_norm_final: bool = False,
        fine_drop_prob: float = 0.0,
        fine_drop_min_keep: int = 1,
        fine_drop_dual: bool = False,
        dead_code_threshold: float = 0.0,
    ):
        super().__init__()
        assert len(levels) == len(codebook_sizes)
        assert list(levels) == sorted(levels), "levels must be increasing"
        self.levels = list(levels)
        self.codebook_sizes = list(codebook_sizes)
        self.dim = dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.fine_drop_prob = fine_drop_prob
        self.fine_drop_min_keep = fine_drop_min_keep
        self.fine_drop_dual = fine_drop_dual
        self.dead_code_threshold = dead_code_threshold

        self.embeddings = nn.ModuleList(nn.Embedding(v, dim) for v in codebook_sizes)
        for i, emb in enumerate(self.embeddings):
            emb.weight.data.normal_()
            emb.weight.requires_grad_(False)
            # Ones, not zeros: zero counts blow rarely-hit codes out of range on the first update.
            self.register_buffer(f"ema_cluster_size_{i}", torch.ones(codebook_sizes[i]))
            self.register_buffer(f"ema_w_{i}", emb.weight.data.clone())
            self.register_buffer(
                f"codebook_hit_{i}", torch.zeros(codebook_sizes[i], dtype=torch.long)
            )

        K = len(self.levels)
        self.quant_resi = nn.ModuleList(
            Phi(dim, quant_resi_ratio) if quant_resi_ratio else nn.Identity()
            for _ in range(K)
        )
        self._phi_idx = [_phi_index(i, K) for i in range(K)]

        L, l0 = self.levels[-1], self.levels[0]
        assert L % l0 == 0, f"final length {L} must be divisible by levels[0]={l0}"
        args = (dim, L // l0, l0_cascade_base, l0_cascade_norm, l0_cascade_norm_final)
        self.learned_down_l1 = _l0_sampler(*args, up=False)
        self.learned_up_l1 = _l0_sampler(*args, up=True)

    @property
    def num_levels(self) -> int:
        return len(self.levels)

    def _down(self, z_BLC: torch.Tensor, k: int) -> torch.Tensor:
        if k == self.num_levels - 1:
            return z_BLC
        z_BCL = z_BLC.transpose(1, 2)
        if k == 0:
            out = self.learned_down_l1(z_BCL)
        else:
            out = F.interpolate(z_BCL, size=(self.levels[k],), mode="area")
        return out.transpose(1, 2).contiguous()

    def _up(self, q_BLC: torch.Tensor, k: int) -> torch.Tensor:
        if k != self.num_levels - 1:
            q_BCL = q_BLC.transpose(1, 2)
            if k == 0:
                q_BCL = self.learned_up_l1(q_BCL)
            else:
                q_BCL = F.interpolate(q_BCL, size=(self.levels[-1],), mode="linear")
            q_BLC = q_BCL.transpose(1, 2)
        phi = self.quant_resi[self._phi_idx[k]]
        return phi(q_BLC.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()

    def resize(self, z_hat: torch.Tensor, k: int) -> torch.Tensor:
        """Area-resize a full-length reconstruction to the ``levels[k]`` positions."""
        if self.levels[k] == z_hat.shape[1]:
            return z_hat
        return F.interpolate(
            z_hat.transpose(1, 2), size=(self.levels[k],), mode="area"
        ).transpose(1, 2)

    def split_codes(self, codes: torch.Tensor) -> list[torch.Tensor]:
        return list(codes.split(self.levels, dim=1))

    def add_level(
        self, z_hat: torch.Tensor, codes_k: torch.Tensor, k: int
    ) -> torch.Tensor:
        return z_hat + self._up(self.embeddings[k](codes_k).to(z_hat.dtype), k)

    def _distances(self, x: torch.Tensor, emb_w: torch.Tensor) -> torch.Tensor:
        return x.pow(2).sum(1, keepdim=True) + emb_w.pow(2).sum(1) - 2 * x @ emb_w.t()

    @torch.no_grad()
    def _ema_update(self, k: int, flat: torch.Tensor, indices: torch.Tensor):
        vs = self.codebook_sizes[k]
        onehot = F.one_hot(indices, vs).to(flat.dtype)
        counts = onehot.sum(0)
        dw = onehot.t() @ flat
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts)
            dist.all_reduce(dw)
        ema_cs = getattr(self, f"ema_cluster_size_{k}")
        ema_w = getattr(self, f"ema_w_{k}")
        ema_cs.mul_(self.decay).add_(counts, alpha=1 - self.decay)
        n = ema_cs.sum()
        smoothed = (ema_cs + self.epsilon) / (n + vs * self.epsilon) * n
        ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)
        self.embeddings[k].weight.data.copy_(ema_w / smoothed.unsqueeze(1))
        if self.dead_code_threshold > 0:
            self._restart_dead_codes(k, flat, counts, n)

    @torch.no_grad()
    def _restart_dead_codes(
        self, k: int, flat: torch.Tensor, counts: torch.Tensor, n: torch.Tensor
    ):
        """Re-seed codes unused this update whose usage fell below
        ``dead_code_threshold`` x the uniform share with random current inputs."""
        ema_cs = getattr(self, f"ema_cluster_size_{k}")
        share = n / self.codebook_sizes[k]
        dead = ((ema_cs < self.dead_code_threshold * share) & (counts == 0)).nonzero()
        dead = dead.flatten()
        if dead.numel() == 0:
            return
        new = flat[torch.randint(0, flat.shape[0], (dead.numel(),), device=flat.device)]
        if dist.is_available() and dist.is_initialized():
            # The dead set is identical on every rank (EMA stats are all-reduced); the draws are not.
            dist.broadcast(new, src=0)
        self.embeddings[k].weight.data[dead] = new
        # Restart at the uniform share; restarting at the floor re-kills the code next update.
        getattr(self, f"ema_w_{k}")[dead] = new * share
        ema_cs[dead] = share

    def forward(self, z: torch.Tensor):
        """Training/eval quantization of encoder latents.

        Returns ``(z_hat, z_hat_trunc, vq_loss)``. Both reconstructions carry a
        straight-through gradient to ``z``; ``z_hat_trunc`` is the nested-dropout
        reconstruction (``None`` unless fine-level dropout fired).
        """
        z = z.float()
        B, L, C = z.shape
        z_ng = z.detach()
        z_rest = z_ng.clone()
        z_hat = torch.zeros_like(z_rest)

        drop = (
            self.training
            and self.fine_drop_prob > 0
            and float(torch.rand(())) < self.fine_drop_prob
        )
        partials = []
        vq_loss = z.new_zeros(())
        for k, ell in enumerate(self.levels):
            flat_BLC = self._down(z_rest, k)
            flat = flat_BLC.reshape(-1, C)
            emb = self.embeddings[k]
            indices = self._distances(flat, emb.weight).argmin(dim=1)
            if self.training:
                self._ema_update(k, flat.detach(), indices)
            q = emb(indices.view(B, ell))
            if k == 0:
                # Straight-through only at level 0 trains the learned sampler; the reference reports divergence at higher levels.
                q = flat_BLC + (q - flat_BLC).detach()
            quantized = self._up(q, k)
            z_hat = z_hat + quantized
            z_rest = z_rest - quantized
            if drop:
                partials.append(z_hat)
            vq_loss = vq_loss + self.commitment_cost * F.mse_loss(z_hat.detach(), z)
            vq_loss = vq_loss + F.mse_loss(z_hat, z_ng)
            if self.training:
                getattr(self, f"codebook_hit_{k}")[indices] = 1
        vq_loss = vq_loss / self.num_levels

        z_hat_trunc = None
        if drop and self.num_levels >= 2:
            kmin = max(self.fine_drop_min_keep - 1, 0)
            kmax = self.num_levels - 2
            if kmax >= kmin:
                cut = int(torch.randint(kmin, kmax + 1, ()))
                z_hat_trunc = z + (partials[cut] - z).detach()
        z_hat = z + (z_hat - z).detach()
        if z_hat_trunc is not None and not self.fine_drop_dual:
            z_hat, z_hat_trunc = z_hat_trunc, None
        return z_hat, z_hat_trunc, vq_loss

    @torch.no_grad()
    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """Deterministic codes, concatenated across levels: ``(B, sum(levels))``."""
        z = z.float()
        B, _, C = z.shape
        z_rest = z.clone()
        out = []
        for k, ell in enumerate(self.levels):
            flat = self._down(z_rest, k).reshape(-1, C)
            idx = (
                self._distances(flat, self.embeddings[k].weight)
                .argmin(dim=1)
                .view(B, ell)
            )
            out.append(idx)
            z_rest = z_rest - self._up(self.embeddings[k](idx), k)
        return torch.cat(out, dim=1)

    def cumulative(
        self, codes: torch.Tensor, upto: int | None = None
    ) -> list[torch.Tensor]:
        """Cumulative reconstructions after each of the first ``upto`` levels."""
        upto = self.num_levels if upto is None else upto
        B = codes.shape[0]
        z_hat = torch.zeros(
            B,
            self.levels[-1],
            self.dim,
            device=codes.device,
            dtype=self.embeddings[0].weight.dtype,
        )
        out = []
        for k, codes_k in enumerate(self.split_codes(codes)[:upto]):
            z_hat = self.add_level(z_hat, codes_k, k)
            out.append(z_hat)
        return out

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return self.cumulative(codes)[-1]

    def next_level_inputs(self, codes: torch.Tensor) -> torch.Tensor:
        """Generator inputs for levels 1..K-1: each is the cumulative reconstruction
        of the earlier levels, area-resized to that level's length."""
        partials = self.cumulative(codes, upto=self.num_levels - 1)
        return torch.cat(
            [self.resize(z_hat, k + 1) for k, z_hat in enumerate(partials)], dim=1
        )

    def codebook_utilization(self) -> list[float]:
        return [
            getattr(self, f"codebook_hit_{k}").sum().item() / v
            for k, v in enumerate(self.codebook_sizes)
        ]

    def reset_utilization(self):
        for k in range(self.num_levels):
            getattr(self, f"codebook_hit_{k}").zero_()
