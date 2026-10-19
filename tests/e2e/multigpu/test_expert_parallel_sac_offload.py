"""SAC CPU offload over the chunked, async torch expert-parallel dispatch."""

from pathlib import Path

import torch
import yaml
from accelerate.test_utils import execute_subprocess_async
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers.testing_utils import get_torch_dist_unique_port

from axolotl.utils.dict import DictDefault

from tests.e2e.utils import most_recent_subdir, require_torch_2_7_0


def _load_safetensors(directory, glob="model*.safetensors"):
    state_dict = {}
    for file in sorted(Path(directory).glob(glob)):
        state_dict.update(load_file(file))
    return state_dict


def _read_grad_norm(out_dir):
    from tbparse import SummaryReader

    scalars = SummaryReader(str(most_recent_subdir(out_dir / "runs"))).scalars
    values = scalars[scalars.tag == "train/grad_norm"].value.values
    assert len(values), "no grad_norm logged"
    return float(values[-1])


class TestExpertParallelSacOffload:
    @require_torch_2_7_0
    def test_chunked_dispatch_offload_gradient_parity(self, temp_dir):
        """One SGD step with EP2 must apply the same update whether the dispatch runs
        unchunked without checkpointing or chunked under SAC offload with the dispatch
        saved.

        The chunked dispatch returns async collective outputs whose wait runs on the
        compute stream only at first use; an offload copy issued before that wait
        snapshots a buffer NCCL is still writing, and the recompute replays it into
        the combine. SGD makes the saved weight delta the (clipped) gradient, so a torn
        replay shows up in the per-key update norms and in the logged grad_norm.
        """

        def run(variant, chunked_offload):
            out_dir = Path(temp_dir) / variant
            cfg = DictDefault(
                {
                    "base_model": "axolotl-ai-co/tiny-mixtral-30m",
                    "experts_implementation": "grouped_mm",
                    "sequence_len": 512,
                    "val_set_size": 0,
                    "datasets": [
                        {
                            "path": "tatsu-lab/alpaca",
                            "type": "alpaca",
                            "split": "train[:1%]",
                        },
                    ],
                    "max_steps": 1,
                    "warmup_steps": 0,
                    "micro_batch_size": 2,
                    "gradient_accumulation_steps": 1,
                    "output_dir": str(out_dir),
                    "learning_rate": 10.0,
                    "optimizer": "sgd",
                    "lr_scheduler": "constant",
                    "max_grad_norm": 1.0,
                    "weight_decay": 0.0,
                    "fsdp_version": 2,
                    "fsdp_config": {
                        "offload_params": False,
                        "cpu_ram_efficient_loading": True,
                        "transformer_layer_cls_to_wrap": "MixtralDecoderLayer",
                        "state_dict_type": "FULL_STATE_DICT",
                        "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                        "reshard_after_forward": True,
                    },
                    "plugins": [
                        "axolotl.integrations.expert_parallel.ExpertParallelPlugin"
                    ],
                    "expert_parallel_size": 2,
                    "expert_parallel_backend": "torch",
                    "dp_shard_size": 1,
                    "expert_parallel_dispatch_chunks": 1,
                    "seed": 42,
                    "bf16": True,
                    "save_strategy": "no",
                    "save_only_model": True,
                    "use_tensorboard": True,
                    "logging_steps": 1,
                }
            )
            if chunked_offload:
                cfg["expert_parallel_dispatch_chunks"] = 2
                cfg["expert_parallel_save_dispatch"] = True
                cfg["gradient_checkpointing"] = True
                cfg["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
                # every SAC-saved tensor goes through the offload engine, and the
                # empty save list leaves only the plugin's registered saves
                cfg["selective_checkpointing"] = {"save": [], "offload": True}
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "config.yaml", "w", encoding="utf-8") as fout:
                fout.write(yaml.dump(cfg.to_dict(), Dumper=yaml.Dumper))
            execute_subprocess_async(
                [
                    "axolotl",
                    "train",
                    str(out_dir / "config.yaml"),
                    "--num-processes",
                    "2",
                    "--main-process-port",
                    f"{get_torch_dist_unique_port()}",
                ]
            )
            return _load_safetensors(out_dir), _read_grad_norm(out_dir)

        ref, grad_norm_ref = run("unchunked", chunked_offload=False)
        test, grad_norm_test = run("chunked_offload", chunked_offload=True)

        # clipping to max_grad_norm cancels a uniform gradient scale in the update, so
        # the pre-clip global norm is the only place such an error is visible
        grad_norm_ratio = grad_norm_test / grad_norm_ref
        assert 0.95 < grad_norm_ratio < 1.05, (
            f"chunked+offload grad_norm {grad_norm_test:.4f} is {grad_norm_ratio:.3f}x "
            f"the reference grad_norm {grad_norm_ref:.4f}"
        )

        init = _load_safetensors(
            snapshot_download(
                "axolotl-ai-co/tiny-mixtral-30m", allow_patterns=["*.safetensors"]
            ),
            glob="*.safetensors",
        )
        assert set(test) == set(ref), "checkpoint keys differ between the two arms"
        ratios = {}
        for key, init_w in init.items():
            init_w = init_w.float()
            delta_ref = (ref[key].float() - init_w).norm().item()
            delta_test = (test[key].float() - init_w).norm().item()
            assert test[key].shape == ref[key].shape, key
            if delta_ref == 0:
                assert delta_test == 0, f"{key} moved only under chunked+offload"
                continue
            ratios[key] = delta_test / delta_ref
        assert ratios, "no parameter moved in one SGD step"
        assert any("experts" in k for k in ratios), "no expert parameter moved"
        for key, ratio in ratios.items():
            assert 0.85 < ratio < 1.15, (
                f"{key}: chunked+offload update norm is {ratio:.3f}x the reference"
            )
        # clipping fixes the total update norm at lr * max_grad_norm, so a wrong global
        # grad norm scales every tensor and hides in the per-key band
        total_ref = sum(
            (ref[k].float() - init[k].float()).norm() ** 2 for k in ratios
        ).sqrt()
        total_test = sum(
            (test[k].float() - init[k].float()).norm() ** 2 for k in ratios
        ).sqrt()
        total_ratio = (total_test / total_ref).item()
        assert 0.995 < total_ratio < 1.005, (
            f"chunked+offload total update norm is {total_ratio:.4f}x the reference"
        )
        assert torch.isfinite(torch.tensor(grad_norm_test))
