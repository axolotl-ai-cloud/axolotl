"""DeepSpeed ZeRO-2 + context_parallel_size (Ulysses/ALST) trains the same sequences as a
single process: same per-step loss and gradient norm."""

import os
from pathlib import Path

import pytest
import yaml
from accelerate.test_utils import execute_subprocess_async
from transformers.testing_utils import get_torch_dist_unique_port

from axolotl.utils.dict import DictDefault

pytest.importorskip("deepspeed")


def _scalars(out_dir, tag):
    from tbparse import SummaryReader

    scalars = SummaryReader(str(out_dir), pivot=False).scalars
    values = scalars[scalars.tag == tag].value.values
    assert len(values), f"no {tag} logged"
    return [float(v) for v in values]


class TestDeepSpeedContextParallel:
    def test_two_gpu_cp_matches_single_process(self, temp_dir):
        ds_config = Path(temp_dir) / "zero2.json"
        ds_config.write_text(
            '{"zero_optimization": {"stage": 2}, "bf16": {"enabled": "auto"}, '
            '"gradient_accumulation_steps": "auto", "gradient_clipping": "auto", '
            '"train_batch_size": "auto", "train_micro_batch_size_per_gpu": "auto"}'
        )

        def run(variant, num_processes, context_parallel_size):
            out_dir = Path(temp_dir) / variant
            cfg = DictDefault(
                {
                    "base_model": "axolotl-ai-co/tiny-qwen3-129m",
                    "datasets": [
                        {
                            "path": "tatsu-lab/alpaca",
                            "type": "alpaca",
                            "split": "train[:2]",
                        }
                    ],
                    "dataset_prepared_path": str(out_dir / "prepared"),
                    "val_set_size": 0,
                    "sequence_len": 512,
                    "sample_packing": False,
                    "micro_batch_size": 1,
                    "gradient_accumulation_steps": 1,
                    "max_steps": 2,
                    "learning_rate": 1e-5,
                    "lr_scheduler": "constant",
                    "warmup_steps": 0,
                    "optimizer": "adamw_torch",
                    "bf16": True,
                    "attn_implementation": "sdpa",
                    "seed": 42,
                    "deepspeed": str(ds_config),
                    "context_parallel_size": context_parallel_size,
                    "output_dir": str(out_dir),
                    "save_strategy": "no",
                    "use_tensorboard": True,
                    "logging_steps": 1,
                }
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "config.yaml", "w", encoding="utf-8") as fout:
                fout.write(yaml.dump(cfg.to_dict(), Dumper=yaml.Dumper))
            port = get_torch_dist_unique_port()
            env = dict(os.environ)
            if num_processes == 1:
                # no launcher runs the single-process reference, and DeepSpeed still
                # initializes torch.distributed
                env.update(
                    MASTER_ADDR="127.0.0.1",
                    MASTER_PORT=str(port),
                    RANK="0",
                    LOCAL_RANK="0",
                    WORLD_SIZE="1",
                )
            execute_subprocess_async(
                [
                    "axolotl",
                    "train",
                    str(out_dir / "config.yaml"),
                    "--num-processes",
                    str(num_processes),
                    "--main-process-port",
                    f"{port}",
                ],
                env=env,
            )
            return _scalars(out_dir, "train/loss"), _scalars(out_dir, "train/grad_norm")

        loss_ref, norm_ref = run("single", 1, 1)
        loss_cp, norm_cp = run("cp2", 2, 2)
        assert len(loss_cp) == len(loss_ref) == 2
        for step, (a, b) in enumerate(zip(loss_cp, loss_ref, strict=True)):
            assert abs(a - b) <= 0.02 * max(abs(b), 1.0), (step, loss_cp, loss_ref)
        for step, (a, b) in enumerate(zip(norm_cp, norm_ref, strict=True)):
            assert abs(a - b) <= 0.03 * b, (step, norm_cp, norm_ref)
