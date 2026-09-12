"""8-bit Muon: blockwise int8 momentum (https://arxiv.org/abs/2509.23106)."""

import torch
import torch.nn.functional as F

from axolotl.contribs.mit.dion.opt_utils import (
    create_param_batches,
    dtensor_from_local,
    to_local,
)
from axolotl.contribs.mit.muon.dist_muon import DistMuon, DistMuonOptimizerFactory

BLOCK_SIZE = 2048


def _blocks(x: torch.Tensor) -> torch.Tensor:
    rows = x.reshape(x.shape[0], -1)
    return F.pad(rows, (0, -rows.shape[1] % BLOCK_SIZE)).view(
        rows.shape[0], -1, BLOCK_SIZE
    )


def quantize(m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Linear absmax int8 per block; scale keeps the leading dim so it shards like the param."""
    blocks = _blocks(m)
    scale = (blocks.abs().amax(-1, keepdim=True) / 127).clamp_min(
        torch.finfo(m.dtype).tiny
    )
    q = (blocks / scale).round_().to(torch.int8).view(m.shape[0], -1)[:, : m[0].numel()]
    return q.reshape(m.shape), scale.squeeze(-1)


def dequantize(
    q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    blocks = _blocks(q.to(dtype)) * scale.unsqueeze(-1)
    return blocks.view(q.shape[0], -1)[:, : q[0].numel()].reshape(q.shape)


class Muon8bit(DistMuon):
    """DistMuon whose Muon momentum lives as int8 between steps; AdamW states stay full precision."""

    def _get_or_initialize_state(self, param, algo):
        state = super()._get_or_initialize_state(param, algo)
        if "momentum_q8" in state:
            state["momentum"] = dequantize(
                to_local(state.pop("momentum_q8")),
                to_local(state.pop("momentum_scale")),
                param.dtype,
            )
        return state

    def _create_muon_tasks(self, param_groups, algo_name="muon"):
        # Feed the parent one homogeneous batch at a time: once its tasks are all created the
        # momentum update has run (it precedes the first yield), so only that batch is ever fp.
        for group in param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            for batch in create_param_batches(params, self._world_size):
                yield from super()._create_muon_tasks(
                    [{**group, "params": batch}], algo_name
                )
                for p in batch:
                    state = self.state[p]
                    q, scale = quantize(to_local(state.pop("momentum")))
                    state["momentum_q8"], state["momentum_scale"] = dtensor_from_local(
                        [q, scale], p
                    )


class Muon8bitOptimizerFactory(DistMuonOptimizerFactory):
    optim_cls = Muon8bit
