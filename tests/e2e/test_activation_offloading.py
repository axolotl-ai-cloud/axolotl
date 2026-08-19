"""
E2E tests for activation offloading
"""

import pytest

from axolotl.common.datasets import load_datasets
from axolotl.core.trainers.mixins.activation_checkpointing import (
    ActivationOffloadingMixin,
)
from axolotl.train import train
from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault

from .utils import check_model_output_exists


@pytest.mark.xfail(reason="flaky", strict=False)
class TestActivationOffloading:
    """
    E2E test cases for activation offloading
    """

    @pytest.mark.parametrize(
        "adapter",
        ["lora", "qlora", None],
    )
    def test_activation_offloading(
        self,
        temp_dir,
        adapter,
    ):
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "sequence_len": 1024,
                "val_set_size": 0.0,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                    "eos_token": "<|im_end|>",
                },
                "datasets": [
                    {
                        "chat_template": "chatml",
                        "path": "mlabonne/FineTome-100k",
                        "type": "chat_template",
                        "split": "train[:10%]",
                        "field_messages": "conversations",
                        "message_field_role": "from",
                        "message_field_content": "value",
                    },
                ],
                "num_epochs": 1,
                "max_steps": 2,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch_8bit",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "sample_packing": True,
                "bf16": "auto",
                "gradient_checkpointing": True,
                "activation_offloading": True,
                "save_first_step": False,
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_target_linear": True,
            }
        )
        if adapter == "lora":
            cfg["adapter"] = "lora"
        if adapter == "qlora":
            cfg["adapter"] = "qlora"
            cfg["load_in_4bit"] = True

        cfg = validate_config(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)

    @pytest.mark.parametrize(
        "offload_mode,expect_streams",
        [(True, True), ("legacy", False), ("disk", True)],
    )
    def test_offload_mode_wiring(
        self, temp_dir, monkeypatch, offload_mode, expect_streams
    ):
        """`activation_offloading` must reach the trainer as a live offload
        context: True => stream-overlapped, 'legacy' => synchronous. Guards the
        regression where string modes fell through to plain gradient
        checkpointing (no offload at all)."""
        from trl.models.activation_offloading import OffloadActivations

        captured = {}
        original_step = ActivationOffloadingMixin.training_step

        def capture_step(self, *args, **kwargs):
            captured["ctx"] = self.activation_offload_context
            return original_step(self, *args, **kwargs)

        monkeypatch.setattr(ActivationOffloadingMixin, "training_step", capture_step)

        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "sequence_len": 1024,
                "val_set_size": 0.0,
                "special_tokens": {"pad_token": "<|endoftext|>"},
                "datasets": [
                    {"path": "mhenrichsen/alpaca_2k_test", "type": "alpaca"},
                ],
                "max_steps": 2,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 1e-5,
                "optimizer": "adamw_torch",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "sample_packing": True,
                "bf16": "auto",
                "gradient_checkpointing": True,
                "activation_offloading": offload_mode,
                "adapter": "lora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_target_linear": True,
                "save_first_step": False,
            }
        )
        cfg = validate_config(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)
        train(cfg=cfg, dataset_meta=dataset_meta)

        ctx = captured.get("ctx")
        assert isinstance(ctx, OffloadActivations), (
            f"activation_offloading={offload_mode!r} did not produce an offload "
            f"context (got {type(ctx).__name__}) — string modes likely fell "
            f"through to plain gradient checkpointing"
        )
        assert ctx.use_streams is expect_streams

    @pytest.mark.parametrize(
        "adapter,expect_recompute_wrap",
        [("lora", False), (None, True)],
    )
    def test_offload_is_adapter_aware(
        self, temp_dir, monkeypatch, adapter, expect_recompute_wrap
    ):
        """activation_offloading is adapter-aware: LoRA offloads *instead of*
        recomputing (no checkpoint wrap — pure offload is leaner/faster), while
        full finetune keeps recompute and offloads the checkpoint boundaries
        (the combo — pure offload is PCIe-bound and OOMs at full-param scale)."""
        captured = {}
        original_step = ActivationOffloadingMixin.training_step

        def capture_step(self, *args, **kwargs):
            captured["wrapped"] = any(
                "_checkpoint_wrapped_module" in n for n, _ in self.model.named_modules()
            )
            return original_step(self, *args, **kwargs)

        monkeypatch.setattr(ActivationOffloadingMixin, "training_step", capture_step)

        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "sequence_len": 1024,
                "val_set_size": 0.0,
                "special_tokens": {"pad_token": "<|endoftext|>"},
                "datasets": [
                    {"path": "mhenrichsen/alpaca_2k_test", "type": "alpaca"},
                ],
                "max_steps": 2,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 1e-5,
                "optimizer": "adamw_torch",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "sample_packing": True,
                "bf16": "auto",
                "gradient_checkpointing": True,
                "activation_offloading": True,
                "save_first_step": False,
            }
        )
        if adapter:
            cfg["adapter"] = adapter
            cfg["lora_r"] = 8
            cfg["lora_alpha"] = 16
            cfg["lora_target_linear"] = True

        cfg = validate_config(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)
        train(cfg=cfg, dataset_meta=dataset_meta)

        assert captured.get("wrapped") is expect_recompute_wrap

    @pytest.mark.parametrize("use_reentrant", [False, True])
    def test_hidden_states_offload_full_param(
        self, temp_dir, monkeypatch, use_reentrant
    ):
        """activation_offloading: hidden_states trains in both checkpoint modes."""
        import torch.utils.checkpoint as ckpt

        from axolotl.monkeypatch.activation_offload_checkpoint import (
            HiddenStatesOffloadCheckpoint,
            unpatch_hidden_states_offload,
        )
        from axolotl.monkeypatch.checkpoint_activation_offload import (
            CheckpointHiddenStatesOffload,
        )

        entered_checkpoint_offload = False
        original_enter = CheckpointHiddenStatesOffload.__enter__

        def wrapped_enter(self):
            nonlocal entered_checkpoint_offload
            entered_checkpoint_offload = True
            return original_enter(self)

        if not use_reentrant:
            monkeypatch.setattr(
                CheckpointHiddenStatesOffload, "__enter__", wrapped_enter
            )

        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "sequence_len": 1024,
                "val_set_size": 0.0,
                "special_tokens": {"pad_token": "<|endoftext|>"},
                "datasets": [{"path": "mhenrichsen/alpaca_2k_test", "type": "alpaca"}],
                "max_steps": 2,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 1e-5,
                "optimizer": "adamw_torch",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "sample_packing": True,
                "bf16": "auto",
                "gradient_checkpointing": True,
                "activation_offloading": "hidden_states",
                "save_first_step": False,
            }
        )
        if use_reentrant:
            cfg["gradient_checkpointing_kwargs"] = {"use_reentrant": True}
        try:
            cfg = validate_config(cfg)
            assert cfg.gradient_checkpointing_kwargs["use_reentrant"] is use_reentrant
            normalize_config(cfg)
            dataset_meta = load_datasets(cfg=cfg)
            train(cfg=cfg, dataset_meta=dataset_meta)
            if use_reentrant:
                assert ckpt.CheckpointFunction is HiddenStatesOffloadCheckpoint
            else:
                assert ckpt.CheckpointFunction is not HiddenStatesOffloadCheckpoint
                assert entered_checkpoint_offload is True
            check_model_output_exists(temp_dir, cfg)
        finally:
            unpatch_hidden_states_offload()

    def test_hidden_states_offload_numerical_parity(self):
        """The offloaded checkpoint only round-trips the layer input through CPU,
        so loss and grads must match plain reentrant checkpointing. Same model
        instance for both runs => identical weights; the only difference is the
        d2h/h2d of the per-layer input."""
        import torch
        import torch.utils.checkpoint as ckpt
        from transformers import AutoModelForCausalLM

        from axolotl.monkeypatch.activation_offload_checkpoint import (
            HiddenStatesOffloadCheckpoint,
            patch_hidden_states_offload,
            unpatch_hidden_states_offload,
        )

        if not torch.cuda.is_available():
            pytest.skip("hidden_states offload requires CUDA")

        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceTB/SmolLM2-135M", dtype=torch.bfloat16
        ).to("cuda")
        model.train()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": True}
        )

        torch.manual_seed(1)
        input_ids = torch.randint(0, 1000, (1, 512), device="cuda")
        batch = {"input_ids": input_ids, "labels": input_ids.clone()}

        def run():
            model.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            grad = model.model.layers[0].mlp.down_proj.weight.grad
            return loss.detach().clone(), grad.detach().clone()

        assert ckpt.CheckpointFunction is not HiddenStatesOffloadCheckpoint
        loss_ref, grad_ref = run()

        patch_hidden_states_offload()
        try:
            assert ckpt.CheckpointFunction is HiddenStatesOffloadCheckpoint
            loss_off, grad_off = run()
        finally:
            unpatch_hidden_states_offload()

        # Forward output is identical; backward recompute introduces only bf16
        # float noise, so grads match within a loose tolerance.
        assert torch.allclose(loss_ref, loss_off, rtol=0, atol=1e-3), (
            f"loss diverged: ref={loss_ref.item()} offload={loss_off.item()}"
        )
        assert torch.allclose(grad_ref, grad_off, rtol=1e-2, atol=1e-2), (
            f"grad diverged: max|d|={(grad_ref - grad_off).abs().max().item()}"
        )

    def test_no_vram_leak_regression(self, temp_dir, monkeypatch):
        """#3638 regression — fail on linear VRAM growth across training steps.

        The bug: ``OffloadActivations.__enter__`` doesn't clear cross-step
        state, so a saved tensor that never unpacks during backward
        (MoE / ``torch.compile``) sits in ``ctx.tracker`` forever — and its
        GPU storage stays alive. Across many steps memory grows linearly.

        Tiny CI models won't exhibit the upstream MoE/compile unpack failure
        on their own, so we *inject* the same leftover: after every step we
        stash a small CUDA tensor into ``ctx.tracker``. The fix clears it on
        the next ``__enter__`` (memory flat); without the fix it accumulates
        (memory grows ~constant bytes/step). The fail mode is the bug's own
        symptom — ``torch.cuda.memory_allocated`` increasing across steps.
        """
        import torch

        if not torch.cuda.is_available():
            pytest.skip("VRAM-leak test requires CUDA")

        mem_per_step: list[int] = []
        seed_id = [10**9]
        seed_bytes = 4 * 1024 * 1024  # 4 MB / step

        original_step = ActivationOffloadingMixin.training_step

        def wrapped_step(self, *args, **kwargs):
            torch.cuda.synchronize()
            mem_per_step.append(torch.cuda.memory_allocated())
            out = original_step(self, *args, **kwargs)

            # Inject the MoE-style leftover: a CUDA tensor stuck in
            # OffloadActivations.tracker. The local `seed` ref dies on
            # return — only ctx.tracker keeps it alive, so the next
            # __enter__'s clear (with the fix) actually releases the GPU
            # memory. Without the fix these accumulate step-over-step.
            ctx = self.activation_offload_context
            seed_id[0] += 1
            seed = torch.empty(seed_bytes // 2, dtype=torch.float16, device="cuda")
            ctx.tracker[seed_id[0]] = (seed, False, None, None, None)
            # Stop the next forward's pack_tensor from raising on its
            # "tracker should have been cleared" guard. With the fix this
            # flag gets reset by __enter__ anyway; on main it would
            # otherwise crash before our VRAM measurement on step 2.
            ctx.is_first_forward_call = False
            return out

        monkeypatch.setattr(ActivationOffloadingMixin, "training_step", wrapped_step)

        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "sequence_len": 1024,
                "val_set_size": 0.0,
                "special_tokens": {"pad_token": "<|endoftext|>"},
                "datasets": [
                    {"path": "mhenrichsen/alpaca_2k_test", "type": "alpaca"},
                ],
                "max_steps": 10,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 1e-5,
                "optimizer": "adamw_torch",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "bf16": "auto",
                "gradient_checkpointing": True,
                "activation_offloading": True,
                "save_first_step": False,
            }
        )
        cfg = validate_config(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)
        train(cfg=cfg, dataset_meta=dataset_meta)

        # Drop warm-up steps; allocator settling distorts early samples.
        warmup = 3
        samples = mem_per_step[warmup:]
        assert len(samples) >= 5, (
            f"need >= 5 post-warmup samples, got {len(samples)} "
            f"(total {len(mem_per_step)})"
        )

        # Injection is 4 MB/step. With the fix __enter__ clears each seed
        # before the next step → growth ≈ 0. Without the fix seeds pile up
        # → growth ≈ 4 MB × (steps-1). 10 MB is well above allocator jitter
        # and well below the leaky-build floor.
        growth_mb = (samples[-1] - samples[0]) / (1024**2)
        tolerance_mb = 10

        per_step_mb = [round(m / 1024**2, 1) for m in mem_per_step]
        assert growth_mb < tolerance_mb, (
            f"VRAM grew {growth_mb:.1f} MB across {len(samples)} post-warmup "
            f"steps — linear-increase signature of the #3638 VRAM leak. "
            f"Per-step memory_allocated (MB): {per_step_mb}"
        )

        check_model_output_exists(temp_dir, cfg)
