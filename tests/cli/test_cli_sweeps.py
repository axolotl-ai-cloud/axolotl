"""
unit tests for generating sweep configurations
"""

from axolotl.cli.main import generate_sweep_configs


def test_generate_sweep_configs_no_pairs():
    base_config = {
        "learning_rate": 0.1,
        "micro_batch_size": 1,
        "sample_packing": True,
    }

    sweeps_config = {"micro_batch_size": [1, 2, 4], "weight_decay": [0.0, 0.1]}

    generate_sweep_configs(base_config, sweeps_config)

    assert len(generate_sweep_configs(base_config, sweeps_config)) == 6

    cfg_1 = {
        "learning_rate": 0.1,
        "micro_batch_size": 2,
        "weight_decay": 0.0,
        "sample_packing": True,
    }

    assert any(
        cfg_1 == cfg for cfg in generate_sweep_configs(base_config, sweeps_config)
    )


def test_generate_sweep_configs_with_pairs():
    base_config = {
        "learning_rate": 0.1,
        "micro_batch_size": 1,
        "sample_packing": True,
    }

    sweeps_config = {
        "_": [
            {
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 8,
            },
            {
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 4,
            },
            {
                "micro_batch_size": 4,
                "gradient_accumulation_steps": 2,
            },
            {
                "micro_batch_size": 8,
                "gradient_accumulation_steps": 1,
            },
        ],
        "weight_decay": [0.0, 0.1],
    }

    generate_sweep_configs(base_config, sweeps_config)

    assert len(generate_sweep_configs(base_config, sweeps_config)) == 8

    assert all(
        cfg["gradient_accumulation_steps"] * cfg["micro_batch_size"] == 8
        for cfg in generate_sweep_configs(base_config, sweeps_config)
    )
