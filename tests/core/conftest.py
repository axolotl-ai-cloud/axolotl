"""Shared fixtures for axolotl.core.builders trainer-builder tests."""

import pytest

from axolotl.loaders import ModelLoader, load_tokenizer
from axolotl.utils.config import normalize_config
from axolotl.utils.dict import DictDefault
from axolotl.utils.schemas.enums import RLType


@pytest.fixture(name="base_cfg")
def fixture_base_cfg():
    """
    Base config with all common arguments between SFT and RLHF
    """
    cfg = DictDefault(
        {
            # Model and tokenizer settings
            "base_model": "HuggingFaceTB/SmolLM2-135M-Instruct",
            "sequence_len": 2048,
            "model_config_type": "llama",  # example type
            # Basic training settings
            "micro_batch_size": 2,
            "eval_batch_size": 2,
            "num_epochs": 1,
            "gradient_accumulation_steps": 1,
            "max_steps": 100,
            "val_set_size": 0,
            # Optimizer settings
            "optimizer": "adamw_torch_fused",
            "learning_rate": 0.00005,
            "weight_decay": 0.01,
            "adam_beta1": 0.998,
            "adam_beta2": 0.9,
            "adam_epsilon": 0.00001,
            "max_grad_norm": 1.0,
            # LR scheduler settings
            "lr_scheduler": "cosine",
            "lr_scheduler_kwargs": {"foo": "bar"},
            "warmup_steps": 10,
            "warmup_ratio": None,
            "cosine_min_lr_ratio": 0.1,
            "cosine_constant_lr_ratio": 0.2,
            # Checkpointing and saving
            "save_steps": 100,
            "output_dir": "./model-out",
            "save_total_limit": 4,
            "save_only_model": False,
            # Hardware/performance settings
            "gradient_checkpointing": False,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "dataloader_num_workers": 1,
            "dataloader_pin_memory": True,
            "dataloader_prefetch_factor": 2,
            "context_parallel_size": 1,
            "tensor_parallel_size": 1,
            # Dtype
            "fp16": False,
            "bf16": False,
            "tf32": False,
            # Logging and evaluation
            "logging_steps": 10,
            "eval_steps": 50,
            "eval_strategy": "steps",
            "save_strategy": "steps",
            "include_tokens_per_second": True,
            # Other common settings
            "seed": 42,
            "remove_unused_columns": True,
            "ddp_timeout": 1800,
            "ddp_bucket_cap_mb": 25,
            "ddp_broadcast_buffers": False,
            "dataset_num_proc": 1,
        }
    )

    normalize_config(cfg)
    return cfg


@pytest.fixture(name="dpo_cfg")
def fixture_dpo_cfg(base_cfg):
    cfg = base_cfg.copy()
    cfg.update(
        {
            "rl": RLType.DPO,
            "dpo_use_weighting": True,
            "dpo_label_smoothing": 0.1,
            "beta": 0.1,  # DPO beta
            "dpo_loss_type": ["sigmoid", "sft"],
            "dpo_loss_weights": [1.0, 0.5],
        }
    )
    return cfg


@pytest.fixture(name="orpo_cfg")
def fixture_orpo_cfg(base_cfg):
    cfg = base_cfg.copy()
    cfg.update(
        {
            "rl": RLType.ORPO,
            "orpo_alpha": 0.1,
            "max_prompt_len": 512,
        }
    )
    return cfg


@pytest.fixture(name="kto_cfg")
def fixture_kto_cfg(base_cfg):
    cfg = base_cfg.copy()
    cfg.update(
        {
            "rl": RLType.KTO,
            "kto_desirable_weight": 1.0,
            "kto_undesirable_weight": 1.0,
            "max_prompt_len": 512,
        }
    )
    return cfg


@pytest.fixture(name="grpo_cfg")
def fixture_grpo_cfg(base_cfg):
    cfg = base_cfg.copy()
    cfg.update(
        {
            "rl": RLType.GRPO,
            "trl": DictDefault(
                {
                    "beta": 0.001,
                    "max_completion_length": 256,
                    "use_vllm": False,  # run on CPU
                    # "vllm_device": "auto",
                    # "vllm_gpu_memory_utilization": 0.15,
                    "num_generations": 4,
                    "reward_funcs": ["rewards.rand_reward_func"],
                }
            ),
            # Must be evenly divisible by num_generations
            "micro_batch_size": 4,
            "datasets": [
                {
                    "path": "openai/gsm8k",
                    "name": "main",
                    "split": "train[:1%]",
                }
            ],
        }
    )
    return DictDefault(cfg)


@pytest.fixture(name="ipo_cfg")
def fixture_ipo_cfg(base_cfg):
    cfg = base_cfg.copy()
    cfg.update(
        {
            "rl": RLType.DPO,
            "dpo_loss_type": ["ipo"],
            "dpo_label_smoothing": 0,
            "beta": 0.1,
        }
    )
    return cfg


@pytest.fixture(name="simpo_cfg")
def fixture_simpo_cfg(base_cfg):
    cfg = base_cfg.copy()
    cfg.update(
        {
            "rl": RLType.SIMPO,
            "rl_beta": 0.2,
            "cpo_alpha": 0.9,
            "simpo_gamma": 0.4,
        }
    )
    return cfg


@pytest.fixture(name="sft_cfg")
def fixture_sft_cfg(base_cfg):
    cfg = base_cfg.copy()
    cfg.update(
        {
            "rl": None,
            "sample_packing": False,
            "eval_sample_packing": False,
            "flash_attention": False,
        }
    )
    return cfg


@pytest.fixture(name="rm_cfg")
def fixture_rm_cfg(sft_cfg):
    cfg = sft_cfg.copy()
    cfg.update(
        DictDefault(
            {
                "reward_model": True,
                "datasets": [
                    {
                        "path": "argilla/distilabel-intel-orca-dpo-pairs",
                        "type": "bradley_terry.chat_template",
                        "split": "train[:1%]",
                    }
                ],
            }
        )
    )
    return cfg


@pytest.fixture(name="prm_cfg")
def fixture_prm_cfg(sft_cfg):
    cfg = sft_cfg.copy()
    cfg.update(
        DictDefault(
            {
                "process_reward_model": True,
                "datasets": [
                    {
                        "path": "trl-lib/math_shepherd",
                        "type": "stepwise_supervised",
                        "split": "train[:1%]",
                    }
                ],
            }
        )
    )
    return cfg


@pytest.fixture(name="tokenizer")
def fixture_tokenizer(base_cfg):
    return load_tokenizer(base_cfg)


@pytest.fixture(name="model")
def fixture_model(base_cfg, tokenizer):
    model, _ = ModelLoader(base_cfg, tokenizer).load()
    return model
