# Copyright 2025 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Core MixLoRA model components: Router, Expert, and MoE FFN block.

Architecture (from https://arxiv.org/abs/2404.15159):
- Each FFN layer is replaced with a MixLoRA MoE block
- The MoE block contains the original frozen FFN + N LoRA experts + a router
- Each expert is a set of LoRA pairs (A/B) for gate_proj, up_proj, down_proj
- The router is a trainable linear layer: hidden_dim -> num_experts
- Optimization: shared W1/W3 computation, only LoRA deltas are routed
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from axolotl.integrations.mixlora.constants import MIXLORA_WEIGHTS_NAME
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class MixLoraRouter(nn.Module):
    """Top-k router for MixLoRA that assigns tokens to LoRA experts.

    Implements softmax-based routing with optional jitter noise and
    computes an auxiliary load-balance loss to encourage even expert usage.

    Args:
        hidden_dim: Dimension of the hidden states (model width).
        num_experts: Number of LoRA experts to route between.
        top_k: Number of experts each token is routed to.
        init_range: Standard deviation for weight initialization.
        jitter_noise: Multiplicative jitter noise for exploration during training.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int,
        init_range: float = 0.02,
        jitter_noise: float = 0.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.jitter_noise = jitter_noise

        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        nn.init.normal_(self.gate.weight, std=init_range)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route tokens to experts.

        Args:
            hidden_states: [batch*seq, hidden_dim] flattened token representations.

        Returns:
            router_weights: [batch*seq, top_k] softmax weights for selected experts.
            expert_indices: [batch*seq, top_k] indices of selected experts.
            aux_loss: scalar auxiliary load-balance loss.
        """
        # Optional jitter noise during training
        if self.training and self.jitter_noise > 0:
            noise = torch.empty_like(hidden_states).uniform_(
                1.0 - self.jitter_noise, 1.0 + self.jitter_noise
            )
            hidden_states = hidden_states * noise

        # Compute router logits and softmax probabilities
        router_logits = self.gate(hidden_states)  # [T, E]
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)  # [T, E]

        # Select top-k experts
        router_weights, expert_indices = torch.topk(
            router_probs, self.top_k, dim=-1
        )  # [T, K] each

        # Renormalize weights to sum to 1
        router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True)

        # Compute auxiliary load-balance loss (Switch Transformer formula)
        aux_loss = self._compute_load_balance_loss(router_probs, expert_indices)

        return router_weights, expert_indices, aux_loss

    def _compute_load_balance_loss(
        self, router_probs: torch.Tensor, expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """Compute auxiliary load-balance loss to prevent router collapse.

        Loss = N * sum_i(f_i * P_i) where:
        - f_i = fraction of tokens dispatched to expert i
        - P_i = mean router probability for expert i
        - N = num_experts

        Args:
            router_probs: [T, E] softmax probabilities from the router.
            expert_indices: [T, K] selected expert indices.

        Returns:
            Scalar loss value.
        """
        num_tokens = router_probs.shape[0]

        # f_i: fraction of tokens dispatched to each expert
        # Create a one-hot mask for selected experts and average over tokens
        expert_mask = F.one_hot(expert_indices, self.num_experts).float()  # [T, K, E]
        expert_mask = expert_mask.sum(dim=1)  # [T, E] — may have >1 if top_k > 1
        tokens_per_expert = expert_mask.sum(dim=0)  # [E]
        f_i = tokens_per_expert / num_tokens  # [E]

        # P_i: mean router probability for each expert
        p_i = router_probs.mean(dim=0)  # [E]

        # Auxiliary loss = N * dot(f_i, P_i)
        aux_loss = self.num_experts * torch.dot(f_i, p_i)

        return aux_loss


class MixLoraExpert(nn.Module):
    """A single LoRA expert for a SwiGLU-style FFN (gate_proj, up_proj, down_proj).

    Each expert only stores the LoRA A/B matrices — the base FFN weights
    are shared and computed once in MixLoraFFN.

    Args:
        hidden_dim: Model hidden dimension.
        intermediate_dim: FFN intermediate dimension.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha scaling factor.
        lora_dropout: LoRA dropout probability.
        activation_fn: Activation function to use (e.g. F.silu). Supplied by
            MixLoraFFN from the base FFN so expert and base paths stay identical.
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float = 0.0,
        activation_fn: callable = F.silu,
    ):
        super().__init__()
        self.activation_fn = activation_fn
        self.scaling = lora_alpha / lora_r

        # LoRA for gate_proj: hidden_dim -> intermediate_dim
        self.gate_lora_a = nn.Linear(hidden_dim, lora_r, bias=False)
        self.gate_lora_b = nn.Linear(lora_r, intermediate_dim, bias=False)

        # LoRA for up_proj: hidden_dim -> intermediate_dim
        self.up_lora_a = nn.Linear(hidden_dim, lora_r, bias=False)
        self.up_lora_b = nn.Linear(lora_r, intermediate_dim, bias=False)

        # LoRA for down_proj: intermediate_dim -> hidden_dim
        self.down_lora_a = nn.Linear(intermediate_dim, lora_r, bias=False)
        self.down_lora_b = nn.Linear(lora_r, hidden_dim, bias=False)

        # Dropout
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else None

        # Initialize: A with Kaiming uniform, B with zeros (standard LoRA init)
        for name, param in self.named_parameters():
            if "lora_a" in name:
                nn.init.kaiming_uniform_(param, a=5**0.5)
            elif "lora_b" in name:
                nn.init.zeros_(param)

    def forward(
        self, x: torch.Tensor, gate_out: torch.Tensor, up_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute expert output: base FFN output + LoRA deltas.

        The base FFN computation (gate_out, up_out) is shared across all experts
        and computed once in MixLoraFFN. This method computes only the LoRA
        deltas and adds them.

        Args:
            x: [T_expert, hidden_dim] input tokens assigned to this expert.
            gate_out: [T_expert, intermediate_dim] shared gate_proj output.
            up_out: [T_expert, intermediate_dim] shared up_proj output.

        Returns:
            Tuple of:
                - intermediate: [T_expert, intermediate_dim] SwiGLU-activated intermediate.
                - down_delta: [T_expert, hidden_dim] LoRA delta for down_proj.
        """
        if self.lora_dropout is not None:
            x_drop = self.lora_dropout(x)
        else:
            x_drop = x

        # gate_proj LoRA delta
        gate_delta = self.gate_lora_b(self.gate_lora_a(x_drop)) * self.scaling
        gate_total = gate_out + gate_delta

        # up_proj LoRA delta
        up_delta = self.up_lora_b(self.up_lora_a(x_drop)) * self.scaling
        up_total = up_out + up_delta

        # SwiGLU activation: act(gate) * up
        intermediate = self.activation_fn(gate_total) * up_total

        # down_proj: base down + LoRA delta
        # Note: we need to apply the original down_proj to intermediate,
        # but that's handled in MixLoraFFN. Here we compute the
        # down LoRA delta on the intermediate result.
        down_delta = self.down_lora_b(self.down_lora_a(intermediate)) * self.scaling

        return intermediate, down_delta


class MixLoraFFN(nn.Module):
    """MixLoRA MoE Feed-Forward Network block.

    Replaces the original FFN in each transformer layer. Contains:
    - The original frozen FFN (gate_proj, up_proj, down_proj)
    - A MixLoRA router
    - N MixLoRA experts (LoRA adapters)

    Implements the optimized forward from the paper: compute shared
    gate_proj(x) and up_proj(x) once, then route per-token LoRA deltas
    to selected experts and aggregate.

    Args:
        original_ffn: The original FFN module to wrap.
        num_experts: Number of LoRA experts.
        top_k: Number of experts per token.
        lora_r: LoRA rank for experts.
        lora_alpha: LoRA alpha for experts.
        lora_dropout: LoRA dropout for experts.
        router_init_range: Initialization range for router weights.
        jitter_noise: Router jitter noise during training.
    """

    def __init__(
        self,
        original_ffn: nn.Module,
        num_experts: int,
        top_k: int,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float = 0.0,
        router_init_range: float = 0.02,
        jitter_noise: float = 0.0,
    ):
        super().__init__()

        # Keep original FFN frozen
        self.base_ffn = original_ffn
        for param in self.base_ffn.parameters():
            param.requires_grad = False

        # Detect FFN dimensions
        hidden_dim = self.base_ffn.gate_proj.in_features
        intermediate_dim = self.base_ffn.gate_proj.out_features

        # Resolve activation function from the base FFN (default to F.silu)
        self.activation_fn = getattr(self.base_ffn, "act_fn", F.silu)
        if not callable(self.activation_fn):
            self.activation_fn = F.silu

        # Router
        self.router = MixLoraRouter(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            init_range=router_init_range,
            jitter_noise=jitter_noise,
        )

        # Experts
        self.experts = nn.ModuleList([
            MixLoraExpert(
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                activation_fn=self.activation_fn,
            )
            for _ in range(num_experts)
        ])

        # Accumulated auxiliary loss (collected and reset each training step)
        self._aux_loss: torch.Tensor | None = None

    @property
    def aux_loss(self) -> torch.Tensor | None:
        return self._aux_loss

    def reset_aux_loss(self):
        self._aux_loss = None

    def mixlora_state_dict(self) -> dict[str, torch.Tensor]:
        """Return state dict containing only the trainable MixLoRA parameters (router + experts).

        Use this for saving MixLoRA-specific weights separately from the PEFT checkpoint.
        """
        state = {}
        for name, param in self.router.named_parameters():
            state[f"router.{name}"] = param.data
        for name, param in self.experts.named_parameters():
            state[f"experts.{name}"] = param.data
        return state

    def load_mixlora_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True):
        """Load MixLoRA-specific weights (router + experts) from a state dict."""
        router_state = {k.removeprefix("router."): v for k, v in state_dict.items() if k.startswith("router.")}
        expert_state = {k.removeprefix("experts."): v for k, v in state_dict.items() if k.startswith("experts.")}
        self.router.load_state_dict(router_state, strict=strict)
        self.experts.load_state_dict(expert_state, strict=strict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """MixLoRA MoE forward pass.

        Args:
            x: [batch, seq, hidden_dim] or [batch*seq, hidden_dim]

        Returns:
            Output tensor with same shape as input.
        """
        input_shape = x.shape
        if x.dim() == 3:
            x_flat = x.reshape(-1, x.shape[-1])
        else:
            x_flat = x

        # Clear stale aux loss at the start of each forward pass so that
        # activation-checkpoint replays don't leave behind incorrect values.
        self._aux_loss = None

        # Step 1: Compute shared base FFN outputs (optimization from paper)
        gate_out = self.base_ffn.gate_proj(x_flat)  # [T, I]
        up_out = self.base_ffn.up_proj(x_flat)       # [T, I]

        # Step 2: Route tokens to experts
        router_weights, expert_indices, aux_loss = self.router(x_flat)
        self._aux_loss = aux_loss

        # Step 3: Compute base FFN output (shared SwiGLU + down_proj)
        base_intermediate = self.activation_fn(gate_out) * up_out
        base_output = self.base_ffn.down_proj(base_intermediate)  # [T, H]

        # Step 4: Compute expert LoRA deltas and aggregate
        final_output = base_output.clone()

        # Process each expert
        for expert_idx, expert in enumerate(self.experts):
            # Find which tokens are routed to this expert and their weights
            # expert_indices: [T, K], router_weights: [T, K]
            mask = (expert_indices == expert_idx)  # [T, K]
            if not mask.any():
                continue

            # Get token indices and corresponding weights
            token_mask = mask.any(dim=1)  # [T]
            token_indices = token_mask.nonzero(as_tuple=True)[0]  # [T_expert]

            if token_indices.numel() == 0:
                continue

            # Gather per-token weights for this expert (sum across K slots)
            expert_weights = (router_weights * mask.float()).sum(dim=1)  # [T]
            expert_weights = expert_weights[token_indices]  # [T_expert]

            # Get inputs for this expert's tokens
            x_expert = x_flat[token_indices]  # [T_expert, H]
            gate_expert = gate_out[token_indices]  # [T_expert, I]
            up_expert = up_out[token_indices]  # [T_expert, I]

            # Compute expert output (intermediate + down_delta)
            intermediate, down_delta = expert(x_expert, gate_expert, up_expert)

            # Use linearity: down_proj(a) - down_proj(b) = down_proj(a - b)
            intermediate_delta = intermediate - base_intermediate[token_indices]
            expert_delta = self.base_ffn.down_proj(intermediate_delta) + down_delta

            # Weighted accumulation
            final_output[token_indices] += expert_weights.unsqueeze(-1) * expert_delta

        return final_output.reshape(input_shape)


def mixlora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Collect MixLoRA-specific state (router + expert weights) from all patched FFN modules.

    This is the public API for exporting MixLoRA weights separately from
    the base-model or PEFT checkpoint.  It is used by ``MixLoraTrainer._save``
    to persist a sidecar ``mixlora_weights.safetensors`` file alongside the
    standard PEFT adapter checkpoint.

    Args:
        model: The full model (may be wrapped in PeftModel, DataParallel, etc.).

    Returns:
        A flat dict mapping fully-qualified parameter names to tensors,
        containing only the trainable router and expert parameters.
    """
    state: dict[str, torch.Tensor] = {}
    for module_name, module in model.named_modules():
        if isinstance(module, MixLoraFFN):
            module_state = module.mixlora_state_dict()
            for key, value in module_state.items():
                state[f"{module_name}.{key}"] = value
    return state


def load_mixlora_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    strict: bool = True,
) -> None:
    """Load MixLoRA-specific state (router + expert weights) into all patched FFN modules.

    This is the counterpart to :func:`mixlora_state_dict`.  It is used by the
    adapter loader to restore previously-saved MixLoRA sidecar weights.

    When ``strict=True`` the function verifies that:
    1. Every ``MixLoraFFN`` module in the model has corresponding keys in the
       checkpoint (no missing modules).
    2. There are no extra prefixes in the checkpoint that do not correspond to
       any ``MixLoraFFN`` module in the model (no unexpected/renamed blocks).

    Args:
        model: The full model (may be wrapped in PeftModel, DataParallel, etc.).
        state_dict: Flat dict as produced by :func:`mixlora_state_dict`.
        strict: If True, raise ``KeyError`` on missing or unexpected modules.
    """
    model_prefixes: set[str] = set()
    for module_name, module in model.named_modules():
        if isinstance(module, MixLoraFFN):
            model_prefixes.add(module_name)

    # Detect unexpected prefixes in the checkpoint
    if strict:
        checkpoint_prefixes: set[str] = set()
        for key in state_dict:
            # Keys look like "model.layers.0.mlp.router.gate.weight"
            # We need to find the prefix that matches a model module name.
            for mp in model_prefixes:
                if key.startswith(f"{mp}."):
                    checkpoint_prefixes.add(mp)
                    break
            else:
                # Key doesn't match any known MixLoRA module prefix
                # Extract the likely module prefix (everything before router./experts.)
                for marker in ("router.", "experts."):
                    if marker in key:
                        likely_prefix = key[:key.index(marker)].rstrip(".")
                        checkpoint_prefixes.add(likely_prefix)
                        break

        extra_prefixes = checkpoint_prefixes - model_prefixes
        if extra_prefixes:
            raise KeyError(
                f"Unexpected MixLoRA module prefixes in checkpoint that do not "
                f"correspond to any MixLoraFFN in the model: {sorted(extra_prefixes)}"
            )

    for module_name in model_prefixes:
        module = model.get_submodule(module_name)
        prefix = f"{module_name}."
        module_state = {
            key[len(prefix):]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }

        if not module_state and strict:
            raise KeyError(
                f"Missing MixLoRA weights for module '{module_name}' in checkpoint"
            )

        if module_state:
            module.load_mixlora_state_dict(module_state, strict=strict)
