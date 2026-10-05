"""Unit tests for GRPO colocate-mode vLLM wiring."""

from axolotl.core.trainers.grpo import GRPOStrategy
from axolotl.utils.dict import DictDefault


def _cfg(trl: dict, vllm: dict | None = None) -> DictDefault:
    return DictDefault(
        {
            "rl": "grpo",
            "context_parallel_size": 1,
            "trl": {"use_vllm": True, "max_completion_length": 64, **trl},
            "vllm": vllm if vllm is not None else {},
        }
    )


class TestColocateTrainingArgs:
    """GRPOStrategy maps the `vllm:` block onto TRL's colocate engine args."""

    def test_colocate_forwards_vllm_block(self):
        kwargs = GRPOStrategy.set_training_args_kwargs(
            _cfg(
                {"vllm_mode": "colocate", "vllm_enable_sleep_mode": True},
                {
                    "gpu_memory_utilization": 0.4,
                    "tensor_parallel_size": 2,
                    "max_model_len": 4096,
                },
            )
        )
        assert kwargs["vllm_mode"] == "colocate"
        assert kwargs["vllm_enable_sleep_mode"] is True
        assert kwargs["vllm_gpu_memory_utilization"] == 0.4
        assert kwargs["vllm_tensor_parallel_size"] == 2
        assert kwargs["vllm_max_model_length"] == 4096

    def test_colocate_unset_options_fall_through_to_trl_defaults(self):
        kwargs = GRPOStrategy.set_training_args_kwargs(_cfg({"vllm_mode": "colocate"}))
        for key in (
            "vllm_gpu_memory_utilization",
            "vllm_tensor_parallel_size",
            "vllm_max_model_length",
            "vllm_enable_sleep_mode",
        ):
            assert key not in kwargs

    def test_server_mode_does_not_forward_engine_options(self):
        kwargs = GRPOStrategy.set_training_args_kwargs(
            _cfg(
                {"vllm_mode": "server"},
                {"gpu_memory_utilization": 0.9, "max_model_len": 4096},
            )
        )
        assert "vllm_gpu_memory_utilization" not in kwargs
        assert "vllm_max_model_length" not in kwargs

    def test_server_host_port_fall_back_to_vllm_block(self):
        kwargs = GRPOStrategy.set_training_args_kwargs(
            _cfg(
                {
                    "vllm_mode": "server",
                    "vllm_server_host": None,
                    "vllm_server_port": None,
                },
                {"host": "10.0.0.5", "port": 9000},
            )
        )
        assert kwargs["vllm_server_host"] == "10.0.0.5"
        assert kwargs["vllm_server_port"] == 9000

    def test_server_host_port_unset_left_to_trl_default(self):
        kwargs = GRPOStrategy.set_training_args_kwargs(
            _cfg(
                {
                    "vllm_mode": "server",
                    "vllm_server_host": None,
                    "vllm_server_port": None,
                }
            )
        )
        assert "vllm_server_host" not in kwargs
        assert "vllm_server_port" not in kwargs
