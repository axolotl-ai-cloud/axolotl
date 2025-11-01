"""
E2E comparison test for Ring 2.0 grouped MoE kernels.
"""

from __future__ import annotations

import copy
import json
import os
import random
import shutil
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import tempfile
import unittest
import importlib
from contextlib import nullcontext
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config, prepare_plugins, validate_config
from axolotl.utils.dict import DictDefault


def with_temp_dir(test_func):
    @wraps(test_func)
    def wrapper(*args, **kwargs):
        temp_dir = tempfile.mkdtemp()
        try:
            test_func(*args, temp_dir=temp_dir, **kwargs)
        finally:
            shutil.rmtree(temp_dir)

    return wrapper


class GatingStatsCollector:
    """Monkeypatch grouped experts to record per-expert token counts each forward."""

    def __init__(self) -> None:
        self._orig_forward = None
        self._per_step: list[torch.Tensor] = []

    def __enter__(self) -> "GatingStatsCollector":
        from axolotl.monkeypatch.models.bailing_moe_v2.modeling import BailingMoeV2GroupedExperts

        orig_forward = BailingMoeV2GroupedExperts.forward

        @wraps(orig_forward)  # type: ignore[arg-type]
        def wrapped(
            module: Any,
            hidden_states: torch.Tensor,
            topk_idx: torch.Tensor,
            topk_weight: torch.Tensor,
        ):
            outputs = orig_forward(module, hidden_states, topk_idx, topk_weight)
            with torch.no_grad():
                counts = torch.bincount(
                    topk_idx.reshape(-1).detach().to("cpu"),
                    minlength=module.num_experts,
                )
                self._per_step.append(counts)
            return outputs

        self._orig_forward = orig_forward
        BailingMoeV2GroupedExperts.forward = wrapped  # type: ignore[assignment]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        from axolotl.monkeypatch.models.bailing_moe_v2.modeling import BailingMoeV2GroupedExperts

        if self._orig_forward is not None:
            BailingMoeV2GroupedExperts.forward = self._orig_forward  # type: ignore[assignment]

    def summary(self) -> Optional[dict[str, Any]]:
        if not self._per_step:
            return None

        stacked = torch.stack(self._per_step, dim=0).to(torch.float32)
        total = stacked.sum(dim=0)
        return {
            "steps": stacked.size(0),
            "per_step": [counts.tolist() for counts in self._per_step],
            "total_per_expert": total.tolist(),
            "mean_per_expert": stacked.mean(dim=0).tolist(),
            "max_tokens": float(total.max().item()),
            "min_tokens": float(total.min().item()),
        }


def _ensure_vanilla_dtype_patch() -> None:
    """Patch HF Bailing MoE MLP to preserve bf16 outputs for vanilla parity runs."""
    import sys
    from pathlib import Path

    module = None
    for name, loaded in sys.modules.items():
        if name.endswith("modeling_bailing_moe_v2"):
            module = loaded
            break

    if module is None:
        cache_root = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules"
        if cache_root.exists():
            for path in cache_root.rglob("modeling_bailing_moe_v2.py"):
                rel = path.relative_to(cache_root).with_suffix("")
                module_name = ".".join(("transformers_modules",) + rel.parts)
                try:
                    module = importlib.import_module(module_name)
                    break
                except ImportError:
                    continue

    if module is None:
        return

    mlp_cls = getattr(module, "BailingMoeV2MLP", None)
    if mlp_cls is not None and not getattr(mlp_cls, "_axolotl_dtype_patch", False):
        orig_forward = mlp_cls.forward

        def forward(self, hidden_states, *args, **kwargs):  # type: ignore[override]
            outputs = orig_forward(self, hidden_states, *args, **kwargs)
            if isinstance(outputs, torch.Tensor) and outputs.dtype != hidden_states.dtype:
                outputs = outputs.to(hidden_states.dtype)
            return outputs

        mlp_cls.forward = forward  # type: ignore[assignment]
        mlp_cls._axolotl_dtype_patch = True  # type: ignore[attr-defined]

    attn_cls = getattr(module, "BailingMoeV2Attention", None)
    if attn_cls is not None and not getattr(attn_cls, "_axolotl_dtype_patch", False):
        orig_attn_forward = attn_cls.forward

        def attn_forward(self, hidden_states, *args, **kwargs):  # type: ignore[override]
            qkv_weight = getattr(self, "query_key_value", None)
            if isinstance(qkv_weight, torch.nn.Module):
                weight = getattr(qkv_weight, "weight", None)
                if isinstance(weight, torch.Tensor) and hidden_states.dtype != weight.dtype:
                    hidden_states = hidden_states.to(weight.dtype)
            outputs = orig_attn_forward(self, hidden_states, *args, **kwargs)
            return outputs

        attn_cls.forward = attn_forward  # type: ignore[assignment]
        attn_cls._axolotl_dtype_patch = True  # type: ignore[attr-defined]


class TestRingMoeGrouped(unittest.TestCase):
    """Ensure grouped torch._grouped_mm backend trains identically to vanilla loops."""

    _BASE_CFG = {
        "base_model": "yujiepan/ring-tiny-random",
        "tokenizer_config": "yujiepan/ring-tiny-random",
        "trust_remote_code": True,
        "flash_attention": False,
        "sequence_len": 512,
        "bf16": True,
        "fp16": False,
        "val_set_size": 0.0,
        "special_tokens": {},
        "datasets": [
            {
                "path": "mhenrichsen/alpaca_2k_test",
                "type": "alpaca",
            },
        ],
        "num_epochs": 1,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-4,
        "optimizer": "adamw_torch",
        "lr_scheduler": "cosine",
        "max_steps": 3,
        "save_steps": 0,
        "eval_steps": 0,
        "save_first_step": False,
        "logging_steps": 1,
        "report_to": [],
        "gradient_checkpointing": False,
        "train_on_inputs": False,
        "seed": 1234,
    }

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _run_training(
        self,
        temp_dir: str,
        mlp_impl: str,
        *,
        max_steps: int | None = None,
        seed: int | None = None,
        record_gating: bool = False,
    ) -> tuple[int, list[tuple[int, float]], Optional[dict[str, Any]]]:
        if mlp_impl == "vanilla":
            _ensure_vanilla_dtype_patch()

        cfg_dict = copy.deepcopy(self._BASE_CFG)
        cfg_dict["output_dir"] = os.path.join(temp_dir, mlp_impl)

        cfg_dict["bf16"] = False
        cfg_dict["fp16"] = False

        if mlp_impl == "grouped":
            cfg_dict["mlp_impl"] = "grouped"
            cfg_dict["use_grouped_moe_kernels"] = True
        elif mlp_impl == "megablocks":
            cfg_dict["mlp_impl"] = "megablocks"
            cfg_dict["use_grouped_moe_kernels"] = True
            cfg_dict["bf16"] = True
        else:
            cfg_dict["use_grouped_moe_kernels"] = False
            cfg_dict.pop("mlp_impl", None)

        if max_steps is not None:
            cfg_dict["max_steps"] = max_steps
        if seed is not None:
            cfg_dict["seed"] = seed

        cfg = DictDefault(cfg_dict)
        cfg = validate_config(cfg)
        normalize_config(cfg)
        prepare_plugins(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        self._set_seed(int(cfg.seed or 0))

        capture_gating = record_gating or os.environ.get("MEGABLOCKS_CAPTURE_GATING") == "1"
        gating_collector: Optional[GatingStatsCollector] = GatingStatsCollector() if capture_gating else None

        context_manager = gating_collector if gating_collector is not None else nullcontext()
        with context_manager:
            model, _, trainer = train(cfg=cfg, dataset_meta=dataset_meta)

        patched_count = sum(
            1 for module in model.modules() if getattr(module, "_axolotl_grouped_moe", False)
        )
        loss_entries: list[tuple[int, float]] = []
        for entry in trainer.state.log_history:
            if not isinstance(entry, dict) or "loss" not in entry:
                continue
            step = int(entry.get("step", len(loss_entries) + 1))
            loss_entries.append((step, float(entry["loss"])))

        self.assertGreater(len(loss_entries), 0, "Expected training loss history to be populated.")

        if hasattr(trainer, "accelerator") and hasattr(trainer.accelerator, "free_memory"):
            trainer.accelerator.free_memory()

        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gating_summary = gating_collector.summary() if gating_collector is not None else None
        return patched_count, loss_entries, gating_summary

    @with_temp_dir
    def test_grouped_matches_vanilla_training(self, temp_dir: str):
        vanilla_patched, vanilla_trace, _ = self._run_training(temp_dir, mlp_impl="vanilla")
        grouped_patched, grouped_trace, _ = self._run_training(temp_dir, mlp_impl="grouped")

        self.assertEqual(vanilla_patched, 0, "Vanilla run should not apply grouped kernels.")
        self.assertGreater(grouped_patched, 0, "Grouped run did not patch any MoE blocks.")
        self.assertEqual(
            len(vanilla_trace),
            len(grouped_trace),
            "Loss histories should have identical lengths.",
        )

        for (step_v, vanilla), (step_g, grouped) in zip(vanilla_trace, grouped_trace):
            self.assertEqual(step_v, step_g, "Logged training steps should align between runs.")
            self.assertAlmostEqual(
                vanilla,
                grouped,
                delta=1e-4,
                msg=f"Loss diverged at step {step_v}: vanilla={vanilla}, grouped={grouped}",
            )

    @with_temp_dir
    def test_megablocks_matches_vanilla(self, temp_dir: str):
        from axolotl.monkeypatch.models.bailing_moe_v2.modeling import _load_megablocks_backend

        if not torch.cuda.is_available():
            self.skipTest("CUDA/ROCm device required for MegaBlocks parity test.")

        if _load_megablocks_backend() is None:
            self.skipTest("MegaBlocks backend not available; skipping parity test.")

        vanilla_patched, vanilla_trace, _ = self._run_training(temp_dir, mlp_impl="vanilla")
        megablocks_patched, megablocks_trace, _ = self._run_training(temp_dir, mlp_impl="megablocks")

        self.assertEqual(vanilla_patched, 0, "Vanilla run should not apply grouped kernels.")
        self.assertGreater(megablocks_patched, 0, "MegaBlocks run did not patch any MoE blocks.")
        self.assertEqual(
            len(vanilla_trace),
            len(megablocks_trace),
            "Loss histories should have identical lengths.",
        )

        for (step_v, vanilla), (step_m, megablocks) in zip(vanilla_trace, megablocks_trace):
            self.assertEqual(step_v, step_m, "Logged training steps should align between runs.")
            self.assertAlmostEqual(
                vanilla,
                megablocks,
                delta=1e-4,
                msg=f"MegaBlocks loss diverged at step {step_v}: vanilla={vanilla}, megablocks={megablocks}",
            )

    @with_temp_dir
    def test_megablocks_longer_run(self, temp_dir: str):
        if os.environ.get("RUN_SLOW_MEGABLOCKS_PARITY") != "1":
            self.skipTest("Set RUN_SLOW_MEGABLOCKS_PARITY=1 to enable slower parity test.")

        from axolotl.monkeypatch.models.bailing_moe_v2.modeling import _load_megablocks_backend

        if not torch.cuda.is_available():
            self.skipTest("CUDA/ROCm device required for MegaBlocks parity test.")

        backend = _load_megablocks_backend()
        if backend is None:
            self.skipTest("MegaBlocks backend not available; skipping parity test.")

        steps = int(os.environ.get("MEGABLOCKS_PARITY_STEPS", "50"))
        seed = int(os.environ.get("MEGABLOCKS_PARITY_SEED", "1337"))

        capture_gating = os.environ.get("MEGABLOCKS_CAPTURE_GATING") == "1"

        vanilla_patched, vanilla_trace, vanilla_gating = self._run_training(
            temp_dir,
            mlp_impl="vanilla",
            max_steps=steps,
            seed=seed,
            record_gating=capture_gating,
        )
        megablocks_patched, megablocks_trace, megablocks_gating = self._run_training(
            temp_dir,
            mlp_impl="megablocks",
            max_steps=steps,
            seed=seed,
            record_gating=capture_gating,
        )

        self.assertEqual(vanilla_patched, 0, "Vanilla run should not apply grouped kernels.")
        self.assertGreater(megablocks_patched, 0, "MegaBlocks run did not patch any MoE blocks.")

        self.assertEqual(len(vanilla_trace), len(megablocks_trace))

        trace_out = Path(temp_dir) / "megablocks_parity_trace.json"
        trace_data = []
        for (step_v, vanilla), (step_m, megablocks) in zip(vanilla_trace, megablocks_trace):
            self.assertEqual(step_v, step_m, "Logged training steps should align between runs.")
            trace_data.append(
                {
                    "step": step_v,
                    "vanilla_loss": vanilla,
                    "megablocks_loss": megablocks,
                    "delta": abs(vanilla - megablocks),
                }
            )

        payload: dict[str, Any] = {"loss_trace": trace_data}
        if capture_gating:
            payload["gating"] = {
                "vanilla": vanilla_gating,
                "megablocks": megablocks_gating,
            }
        trace_out.write_text(json.dumps(payload, indent=2))

        final_vanilla = vanilla_trace[-1][1]
        final_megablocks = megablocks_trace[-1][1]
        delta = abs(final_vanilla - final_megablocks)
        self.assertLess(
            delta,
            0.05,
            f"MegaBlocks final loss deviates too much (vanilla={final_vanilla:.4f}, megablocks={final_megablocks:.4f})",
        )

    def test_megablocks_gradient_parity(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA/ROCm device required for MegaBlocks gradient parity test.")

        from axolotl.monkeypatch.models.bailing_moe_v2.modeling import (
            BailingMoeV2GroupedExperts,
            _load_megablocks_backend,
        )

        backend = _load_megablocks_backend()
        if backend is None:
            self.skipTest("MegaBlocks backend not available; skipping gradient parity test.")

        device = torch.device("cuda")
        torch.manual_seed(1337)

        hidden_size = 32
        intermediate_size = 48
        num_experts = 16
        top_k = 2
        batch = 2
        seq = 8

        config = SimpleNamespace(
            hidden_size=hidden_size,
            moe_intermediate_size=intermediate_size,
            num_experts=num_experts,
            hidden_act="silu",
        )

        class _TinyExpert(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=True)
                self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=True)
                self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=True)

        base_experts: list[nn.Module] = []
        for _ in range(num_experts):
            expert = _TinyExpert()
            expert.to(device=device, dtype=torch.bfloat16)
            base_experts.append(expert)

        loop_mod = BailingMoeV2GroupedExperts(config, base_experts, backend_impl="grouped").to(device)
        megablocks_mod = BailingMoeV2GroupedExperts(config, base_experts, backend_impl="megablocks").to(device)

        hidden_seed = torch.randn(batch, seq, hidden_size, device=device, dtype=torch.bfloat16)

        token_slots = batch * seq * top_k
        idx_values = torch.arange(token_slots, device=device, dtype=torch.long) % num_experts
        topk_idx = idx_values.view(batch * seq, top_k)
        topk_weight = torch.full((batch * seq, top_k), 1.0 / top_k, device=device, dtype=torch.bfloat16)

        def python_loop_forward(
            module: BailingMoeV2GroupedExperts,
            hidden_states: torch.Tensor,
            expert_idx: torch.Tensor,
            expert_weight: torch.Tensor,
        ) -> torch.Tensor:
            bsz, seq_len, hidden = hidden_states.shape
            hidden_flat = hidden_states.view(-1, hidden)
            num_tokens = hidden_flat.size(0)
            topk = expert_idx.shape[-1]
            dispatch_indices = torch.arange(num_tokens, device=hidden_states.device, dtype=torch.long)
            dispatch_indices = dispatch_indices.repeat_interleave(topk)
            flat_expert = expert_idx.reshape(-1)
            flat_weight = expert_weight.reshape(-1).to(hidden_states.dtype)

            output_flat = hidden_states.new_zeros((num_tokens, hidden))
            for expert_id in range(module.num_experts):
                mask = flat_expert == expert_id
                if not torch.any(mask):
                    continue
                token_idx = dispatch_indices[mask]
                tokens = hidden_flat.index_select(0, token_idx).float()

                gate = torch.nn.functional.linear(
                    tokens,
                    module.gate_weight[expert_id].float(),
                    module.gate_bias[expert_id].float() if module.gate_bias is not None else None,
                )
                up = torch.nn.functional.linear(
                    tokens,
                    module.up_weight[expert_id].float(),
                    module.up_bias[expert_id].float() if module.up_bias is not None else None,
                )
                activated = module.act_fn(gate).to(up.dtype)
                activated = activated * up
                down = torch.nn.functional.linear(
                    activated,
                    module.down_weight[expert_id].float(),
                    module.down_bias[expert_id].float() if module.down_bias is not None else None,
                )
                weighted = down.to(hidden_states.dtype) * flat_weight[mask].unsqueeze(-1)
                output_flat.index_add_(0, token_idx, weighted)

            return output_flat.view(bsz, seq_len, hidden)

        hidden_loop = hidden_seed.clone().detach().to(torch.float32).requires_grad_(True)
        hidden_megablocks = hidden_seed.clone().detach().requires_grad_(True)

        loop_out = python_loop_forward(loop_mod, hidden_loop, topk_idx, topk_weight.to(torch.float32))
        loss_loop = loop_out.float().pow(2).mean()
        loss_loop.backward()

        out_megablocks = megablocks_mod(hidden_megablocks, topk_idx, topk_weight)
        loss_megablocks = out_megablocks.float().pow(2).mean()
        loss_megablocks.backward()

        self.assertIsNotNone(hidden_loop.grad)
        self.assertIsNotNone(hidden_megablocks.grad)

        input_grad_diff = torch.max(
            torch.abs(hidden_loop.grad.float() - hidden_megablocks.grad.float())
        ).item()
        self.assertLess(
            input_grad_diff,
            5e-3,
            f"Input gradients diverged: max_abs_diff={input_grad_diff:.4e}",
        )

        loop_params = dict(loop_mod.named_parameters())
        mega_params = dict(megablocks_mod.named_parameters())
        self.assertEqual(loop_params.keys(), mega_params.keys())

        for name, loop_param in loop_params.items():
            mega_param = mega_params[name]
            self.assertIsNotNone(loop_param.grad)
            self.assertIsNotNone(mega_param.grad)
            grad_loop = loop_param.grad.float()
            grad_mega = mega_param.grad.float()
            max_diff = torch.max(torch.abs(grad_loop - grad_mega)).item()
            self.assertLess(
                max_diff,
                5e-3,
                f"Gradient mismatch for {name}: max_abs_diff={max_diff:.4e}",
            )


if __name__ == "__main__":
    unittest.main()
