import time
import torch
import numpy as np
from tqdm import tqdm
from axolotl.cli import load_datasets
from torch.utils.data import RandomSampler
from axolotl.utils.dict import DictDefault
from axolotl.common.cli import TrainerCliArgs
from axolotl.utils.config import normalize_config
from transformers.data import default_data_collator
from axolotl.utils.dataloader import MultipackDistributedDataloader

cfg = DictDefault(
    {
        "base_model": "openaccess-ai-collective/tiny-mistral",
        "base_model_config": "openaccess-ai-collective/tiny-mistral",
        "flash_attention": True,
        "sample_packing": True,
        "sequence_len": 1024,
        "val_set_size": 0.1,
        "special_tokens": {
            "unk_token": "<unk>",
            "bos_token": "<s>",
            "eos_token": "</s>",
        },
        "datasets": [
            {
                "path": "mhenrichsen/alpaca_2k_test",
                "type": "alpaca",
            },
        ],
        "num_epochs": 2,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "output_dir": "./out",
        "eval_steps": 10,
    }
)

normalize_config(cfg)
cli_args = TrainerCliArgs()
dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

sampler = RandomSampler(dataset_meta.train_dataset)
dataloader = MultipackDistributedDataloader(
    dataset=dataset_meta.train_dataset,
    collate_fn=default_data_collator,
    seq_max_length=cfg["sequence_len"],
    batch_size=1,
    sampler=None,
    packing_efficiency_estimate=1.0,
    sample_packing_seq_len_multiplier=1,
    device_count=1,
)

# Let workers warmup
time.sleep(2)

# Measure throughput
timing = []
num_iterations = dataloader.len_w_stats()
iter_dataset = iter(dataloader)

for i in tqdm(range(num_iterations)):
    t_start = time.time()
    batch = next(iter_dataset)
    inputs_ids = batch["input_ids"]
    for _ in range(1000): torch.matmul(inputs_ids, inputs_ids.mT)
    timing.append(time.time() - t_start)

# Calculate throughput
throughput = 1 / np.median(timing)

print(f"Throughput: {throughput:.2f} batches/sec")
