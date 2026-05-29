"""Multi-GPU e2e test for ``fp32_norms`` under FSDP2.

Two-GPU subprocess run with ``fp32_norms: true`` + ``fsdp_version: 2`` + bf16
training. The test plugin
``tests.e2e.multigpu._fp32_norms_dtype_capture.DtypeCapturePlugin`` dumps
post-step-1 param dtypes as JSON; the outer test asserts norms stayed fp32 and
at least one non-norm param dropped to bf16 (proving the two policies are
genuinely independent, not a globally-cast model).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml
from accelerate.test_utils import execute_subprocess_async
from transformers.testing_utils import get_torch_dist_unique_port

from axolotl.utils.dict import DictDefault

from tests.e2e.utils import require_torch_2_7_0

AXOLOTL_ROOT = Path(__file__).parent.parent.parent.parent


def _base_fp32_norms_config(
    temp_dir: str, *, cpu_ram_efficient_loading: bool = False, **overrides
) -> DictDefault:
    """Base config for fp32_norms + FSDP2 multi-GPU."""
    cfg = {
        "base_model": "axolotl-ai-co/tiny-qwen3-129m",
        "sequence_len": 256,
        "val_set_size": 0.0,
        "datasets": [
            {
                "path": "tatsu-lab/alpaca",
                "type": "alpaca",
                "split": "train[:1%]",
            },
        ],
        # Full FT (no adapter) — fp32_norms is about base-model norm precision,
        # which adapters wouldn't exercise.
        "num_epochs": 1,
        "max_steps": 2,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "output_dir": temp_dir,
        "learning_rate": 1e-4,
        "optimizer": "adamw_torch_fused",
        "lr_scheduler": "cosine",
        "flash_attention": True,
        "bf16": True,
        "fp32_norms": True,
        "fsdp_version": 2,
        "fsdp_config": {
            "offload_params": False,
            "cpu_ram_efficient_loading": cpu_ram_efficient_loading,
            "transformer_layer_cls_to_wrap": "Qwen3DecoderLayer",
            "state_dict_type": "FULL_STATE_DICT",
            "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "reshard_after_forward": True,
        },
        "plugins": [
            "tests.e2e.multigpu._fp32_norms_dtype_capture.DtypeCapturePlugin",
        ],
        "save_safetensors": True,
    }
    cfg.update(overrides)
    return DictDefault(cfg)


def _run_training(temp_dir: str, cfg: DictDefault, dump_path: Path) -> None:
    """Write yaml + spawn 2-process training; plugin path goes via PYTHONPATH."""
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(temp_dir) / "config.yaml", "w", encoding="utf-8") as fout:
        fout.write(yaml.dump(cfg.to_dict(), Dumper=yaml.Dumper))

    env = os.environ | {
        # Make the test-only plugin module importable in the subprocess.
        "PYTHONPATH": (f"{AXOLOTL_ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"),
        "FP32_NORMS_DTYPE_DUMP_PATH": str(dump_path),
    }
    execute_subprocess_async(
        [
            "axolotl",
            "train",
            str(Path(temp_dir) / "config.yaml"),
            "--num-processes",
            "2",
            "--main-process-port",
            f"{get_torch_dist_unique_port()}",
        ],
        env=env,
    )


class TestFSDP2Fp32Norms:
    """Verifies the fp32_norms FSDP2 path end-to-end across 2 GPUs."""

    @require_torch_2_7_0
    @pytest.mark.parametrize(
        "cpu_ram_efficient_loading",
        [False, True],
        ids=["materialized-load", "cpu-ram-efficient-load"],
    )
    def test_norms_stay_fp32_under_fsdp2_bf16(
        self, temp_dir, cpu_ram_efficient_loading
    ):
        """fp32_norms keeps RMSNorm params in fp32 while the rest stays bf16."""
        dump_path = Path(temp_dir) / "dtype_capture.json"
        cfg = _base_fp32_norms_config(
            temp_dir,
            cpu_ram_efficient_loading=cpu_ram_efficient_loading,
        )
        _run_training(temp_dir, cfg, dump_path)

        # Training completed (no FSDP1-style flat-param dtype crash) AND the
        # plugin captured dtypes after step 1.
        assert dump_path.exists(), (
            f"plugin did not dump dtype capture to {dump_path}; "
            "training may have failed before step 1"
        )

        captured = json.loads(dump_path.read_text())
        norms = captured["norms"]
        non_norms = captured["non_norms"]

        assert norms, "no norm params captured — matcher likely failed"
        assert all(d == "float32" for d in norms.values()), (
            "fp32_norms claim violated: at least one norm param is not fp32. "
            f"Captured norm dtypes: {norms}"
        )

        # At least one non-norm param must be bf16. Without this check the
        # test would pass on a globally-fp32 model that didn't shard anything.
        non_norm_dtypes = set(non_norms.values())
        assert "bfloat16" in non_norm_dtypes, (
            "expected at least one non-norm param in bfloat16 (proves the two "
            "policies are independent); got non-norm dtypes: "
            f"{non_norm_dtypes}"
        )
