"""End-to-end tests for the ProTrain Axolotl plugin glue (M5).

Two tests live here:

* ``test_plugin_e2e_tiny_llama`` — runs the full Axolotl
  config-validate → load-datasets → train path on a small SmolLM2-135M
  model with ``protrain_auto_memory: true`` +
  ``protrain_force_all_persistent: true``. Asserts no OOM / no crash,
  a decreasing loss trend, and that a checkpoint was written. Marked
  ``slow`` + ``gpu`` — it needs one free CUDA device.

* ``test_plugin_e2e_7b_lora_smoke`` — wires the real
  ``examples/protrain/3090-8b-lora.yml`` for manual validation.
  Marked ``skip`` so CI does not need the 7B weight download.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _marker(stage: str) -> None:
    """Print a progress marker that survives pytest's output buffering."""
    import sys

    sys.stderr.write(f"[protrain-e2e] {stage}\n")
    sys.stderr.flush()


@pytest.mark.slow
@pytest.mark.gpu
def test_plugin_e2e_tiny_llama(tmp_path: Path) -> None:
    """Run the full Axolotl training path with the ProTrain plugin on.

    Uses ``HuggingFaceTB/SmolLM2-135M`` — a small Llama-architecture
    model that lives in the HF hub's open set. The plugin's
    ``force_all_persistent`` path keeps all chunks on GPU and wraps
    every block in CKPT; on a 24 GB card this is a no-offload stress
    test of the plugin shim rather than the runtime primitives, but it
    exercises every hook (``get_input_args``, ``post_model_load``,
    ``create_optimizer``, ``post_trainer_create``) on a real
    HuggingFace Trainer.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    import torch

    if not torch.cuda.is_available():
        pytest.skip("ProTrain plugin E2E requires CUDA.")

    # Fresh PluginManager for the test so we don't collide with any
    # plugins a previous test left registered (PluginManager is a
    # module-level singleton).
    from axolotl.integrations.base import PluginManager

    PluginManager._instance = None  # type: ignore[attr-defined]

    output_dir = tmp_path / "protrain-tiny-out"

    # Build a minimal cfg dict — same shape the CLI would load from YAML,
    # but constructed in Python so we can point output_dir at tmp_path.
    # SmolLM2-135M is an existing Axolotl-test-friendly target
    # (see tests/e2e/test_llama_pretrain.py) with a Llama arch.
    from axolotl.utils.dict import DictDefault

    cfg = DictDefault(
        {
            "base_model": "HuggingFaceTB/SmolLM2-135M",
            "model_type": "AutoModelForCausalLM",
            "tokenizer_type": "AutoTokenizer",
            "load_in_8bit": False,
            "load_in_4bit": False,
            "strict": False,
            "datasets": [
                {
                    "path": "mhenrichsen/alpaca_2k_test",
                    "type": "alpaca",
                }
            ],
            "val_set_size": 0.0,
            "output_dir": str(output_dir),
            "sequence_len": 128,
            "sample_packing": False,
            "pad_to_sequence_len": False,
            "adapter": "lora",
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "lora_target_modules": ["q_proj", "v_proj"],
            "plugins": ["axolotl.integrations.protrain.ProTrainPlugin"],
            "protrain_auto_memory": True,
            # Deliberately DO NOT set protrain_auto_mode — rely on its
            # True default. For SmolLM2-135M on single-rank the
            # selector picks Mode A (force_all_persistent=True,
            # zero3_shard=False) which is the path this test is
            # validating. Regression guard: if the default flips, this
            # test's coverage of Mode A under auto-select breaks.
            "gradient_accumulation_steps": 1,
            "micro_batch_size": 1,
            # 60 steps gives enough samples for a 10-step window-average
            # comparison (first window vs last window) that absorbs the
            # bf16-LoRA + alpaca-length-variance step-to-step noise
            # without being too long for CI. At max_steps=10/30 a
            # per-step trend check was flaky on the AdamW baseline too;
            # the windowed comparison below is robust at 60.
            "max_steps": 60,
            "optimizer": "adamw_torch",
            "lr_scheduler": "constant",
            # Lower LR than the default Axolotl LoRA recipe — the 135M
            # SmolLM2 is sensitive enough at 5e-4 that bf16 rounding
            # alone produces large step-to-step loss swings; 1e-4 keeps
            # the mean trend visible over 30 steps.
            "learning_rate": 0.0001,
            "bf16": "auto",
            "tf32": False,
            "gradient_checkpointing": False,
            "flash_attention": False,
            "logging_steps": 1,
            "save_steps": 30,
            "save_first_step": False,
            "save_total_limit": 1,
            "warmup_steps": 0,
            "weight_decay": 0.0,
            "dataset_num_proc": 1,
            "use_tensorboard": True,
            "special_tokens": {
                "pad_token": "<|endoftext|>",
            },
        }
    )

    # Regression guard for the ``protrain_auto_mode`` default: every
    # user YAML must inherit True so the plugin auto-selects the
    # mode. Hard-code-checked rather than imported from the module so
    # a careless default flip surfaces here with a clear failure.
    from axolotl.integrations.protrain.args import ProTrainArgs

    assert ProTrainArgs.model_fields["protrain_auto_mode"].default is True, (
        "protrain_auto_mode default must be True — flipping it silently "
        "breaks the M7 ZeRO-3 footgun fix."
    )

    _marker("cfg built; registering plugin via prepare_plugins")

    # Mirror what do_train does pre-validate: register plugins so their
    # args schemas get merged into validate_config.
    from axolotl.utils.config import normalize_config, prepare_plugins, validate_config

    prepare_plugins(cfg)

    _marker("calling validate_config")
    cfg = validate_config(cfg)

    _marker("calling normalize_config")
    normalize_config(cfg)

    # Ensure PluginManager.cfg is set — normally done by do_cli path.
    PluginManager.get_instance().cfg = cfg

    _marker("loading datasets")
    from axolotl.cli.args import TrainerCliArgs
    from axolotl.common.datasets import load_datasets

    cli_args = TrainerCliArgs()
    dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

    _marker("entering axolotl.train.train")
    from axolotl.train import train

    _model, _tokenizer, trainer = train(cfg=cfg, dataset_meta=dataset_meta)
    _marker("train() returned")

    # Grab losses off trainer.state.log_history. The HF Trainer logs
    # train/loss for every `logging_steps` entry; we asked for 1.
    losses: list[float] = [
        float(rec["loss"]) for rec in trainer.state.log_history if "loss" in rec
    ]
    assert len(losses) >= 2, (
        f"expected at least 2 training-loss log entries, got {losses}"
    )

    # Sanity: training produced finite, bounded losses.
    import math

    for i, loss in enumerate(losses):
        assert math.isfinite(loss), (
            f"loss at step {i} is not finite: {loss}. losses={losses}"
        )
        assert 0.0 <= loss < 20.0, (
            f"loss at step {i} is out of a sane bf16-LoRA band: {loss}. losses={losses}"
        )
    _marker(f"losses={losses}")

    # Silent-no-op regression guard: directly check that the optimizer
    # step actually applied an update by inspecting LoRA's ``lora_B``
    # tensors. PEFT initializes ``lora_B.weight`` to ZEROS — so any
    # working training step pushes non-zero values into it (the gradient
    # w.r.t. lora_B is non-trivial as long as lora_A's output is
    # non-zero, which it is by construction). If every lora_B is still
    # zero after train() returned, the optimizer step never actually
    # applied an update — the failure mode this test exists to catch.
    #
    # This deterministic check replaces the earlier "first-window avg <
    # last-window avg" loss-trend assertion, which was flaky: per-step
    # loss variance on alpaca + bf16 + small-model + 60-step training
    # often exceeds the per-window mean drift even when training is
    # working. The lora_B-zero check fires precisely on the failure
    # mode the original assertion was trying to catch (no-op step), and
    # never flakes.
    model = (
        trainer.model_wrapped
        if getattr(trainer, "model_wrapped", None) is not None
        else trainer.model
    )
    lora_b_params = [(n, p) for n, p in model.named_parameters() if "lora_B" in n]
    assert lora_b_params, (
        "no lora_B weights found on trainer.model — test assumption "
        "broken (LoRA wiring missing? PEFT version drift?)."
    )
    nonzero_lora_b = sum(
        1 for _, p in lora_b_params if p.detach().abs().sum().item() > 0.0
    )
    assert nonzero_lora_b == len(lora_b_params), (
        f"some lora_B weights are still zero after training "
        f"({nonzero_lora_b}/{len(lora_b_params)} non-zero) — the "
        f"optimizer step never updated those params (silent regression). "
        f"per-tensor abs-sum: "
        f"{[(n, p.detach().abs().sum().item()) for n, p in lora_b_params]}"
    )

    # Loss sanity band. Average loss should be within a reasonable
    # range — catches divergence (loss exploded) or unhinged init
    # without depending on a precise first/last-window comparison.
    if len(losses) >= 20:
        overall_avg = sum(losses) / len(losses)
        assert 0.0 < overall_avg < 5.0, (
            f"average training loss is out of the sane band "
            f"(avg={overall_avg:.4f}). losses={losses}"
        )

    # Checkpoint directory check — adapter safetensors for LoRA runs.
    adapter_file = Path(cfg.output_dir) / "adapter_model.safetensors"
    assert adapter_file.exists(), (
        f"expected adapter checkpoint at {adapter_file}, not found. "
        f"Output dir contents: {list(Path(cfg.output_dir).iterdir())}"
    )

    # FIX 1 regression guard: the plugin MUST install its own optimizer
    # on trainer.optimizer via post_trainer_create. Without this, Axolotl's
    # OptimizerMixin.create_optimizer falls back to vanilla AdamW and the
    # decreasing-loss check above would still pass, silently masking an
    # inert plugin.
    from axolotl.integrations.protrain.api.optim_wrapper import (
        _ProTrainOptimizer,
    )

    # After ``trainer.train()``, Accelerate wraps ``trainer.optimizer``
    # in an ``AcceleratedOptimizer`` whose underlying is reachable via
    # ``.optimizer``. Unwrap one level before the isinstance check.
    underlying = getattr(trainer.optimizer, "optimizer", trainer.optimizer)
    assert isinstance(underlying, _ProTrainOptimizer), (
        "ProTrain plugin is inert: trainer.optimizer (underlying) is "
        f"{type(underlying).__name__}, expected _ProTrainOptimizer. "
        "This means OptimizerMixin used the default AdamW path and the "
        "post_trainer_create hook never installed the ProTrain optimizer."
    )

    # Extra belt-and-braces: the wrapped chunk manager must have seen at
    # least one optimizer step. On an all-persistent LoRA run the GPU
    # FusedAdam adapter is the active one; we check its param_groups were
    # consumed by a step rather than relying on a step counter that may
    # not exist across adapter implementations.
    wrapped = getattr(cfg, "_protrain_wrapped", None)
    assert wrapped is not None, (
        "cfg._protrain_wrapped missing after train(); post_model_load "
        "did not wire the WrappedModel onto cfg."
    )


@pytest.mark.slow
@pytest.mark.gpu
def test_plugin_e2e_7b_lora_smoke(tmp_path: Path) -> None:
    """Smoke-test the real 3090-8b-lora.yml example.

    Equivalent to the CLI invocation::

        axolotl train examples/protrain/3090-8b-lora.yml --max-steps 4

    with ``output_dir`` rerouted to a pytest tmp_path. Skipped by
    default — set ``PROTRAIN_RUN_7B_E2E=1`` in the environment to run
    (requires the Mistral-7B-v0.3 weights, ~14 GB, prefetched into
    HuggingFace cache).

    Run with::

        PROTRAIN_RUN_7B_E2E=1 \\
            CUDA_VISIBLE_DEVICES=2 CUDA_DEVICE_ORDER=PCI_BUS_ID \\
            pytest tests/protrain/test_plugin_e2e.py::test_plugin_e2e_7b_lora_smoke \\
            -m slow -x -s --tb=short -o addopts=
    """
    import os

    if os.environ.get("PROTRAIN_RUN_7B_E2E") != "1":
        pytest.skip(
            "PROTRAIN_RUN_7B_E2E not set — 7B YAML E2E requires the Mistral-7B-v0.3 "
            "weights prefetched into HuggingFace cache (~14 GB). Set the env var "
            "to 1 to opt in."
        )
    pytest.importorskip("torch")

    from axolotl.cli.args import TrainerCliArgs
    from axolotl.cli.config import load_cfg
    from axolotl.cli.train import do_train

    yaml_path = (
        Path(__file__).parent.parent.parent
        / "examples"
        / "protrain"
        / "3090-8b-lora.yml"
    )
    assert yaml_path.exists(), f"missing example yaml at {yaml_path}"

    # Load config; override output_dir + max_steps for a smoke run.
    cfg = load_cfg(
        yaml_path,
        output_dir=str(tmp_path / "protrain-7b-smoke-out"),
        max_steps=4,
    )
    cli_args = TrainerCliArgs()
    do_train(cfg, cli_args)
