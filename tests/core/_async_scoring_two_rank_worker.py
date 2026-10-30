"""Distributed worker: async EBFT with trainer-side scoring on two gloo ranks (DDP).

Each step holds one group of three completions in four rows (one sample per row,
two micro-batches of two ranks), so every step has a filler row. Scripted rewards
and truncation then give, in order:

- step 1, group 0: rewards vary, its third completion is truncated and excluded,
  so one rank's row trains no token while the step trains;
- step 2, group 1: constant rewards, a step that trains nothing on any rank;
- step 3, group 2: rewards vary, the step trains;
- groups 3 onward: constant rewards, until a pass over the 4 prompts trained
  nothing and the stall stop reaches both ranks.

Unequal completion lengths and a scoring budget below a row's sequences make the
ranks run feature forwards over chunks of different sizes, padded to one count.
"""

import itertools
import sys

import numpy as np
import torch
import torch.distributed as dist
from accelerate.state import PartialState
from async_test_utils import StubTransfer, StubWorker, reference_tree_attention
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from test_async_scoring import (
    GROUND_TRUTH,
    LAYER_FRACTIONS,
    MODEL_ID,
    PROMPTS,
    rollout_loop,
    run_group,
    tiny_model,
)
from transformers import AutoTokenizer
from trl.experimental.async_grpo import async_grpo_trainer, packing

from axolotl.core.trainers.ebft.async_trainer import (
    AxolotlAsyncEBFTConfig,
    AxolotlAsyncEBFTTrainer,
    EBFTScorer,
)

TRUNCATED_CALL = 2  # group 0's third completion
TRAINED_GROUP_CALLS = {0, 2}


def samples(tokenizer, rows):
    rng = np.random.default_rng(0)
    calls = itertools.count()

    async def sample(prompt_ids, max_tokens):
        ids = rng.integers(100, 1000, size=rng.integers(3, 9)).tolist()
        if next(calls) != TRUNCATED_CALL:
            ids.append(tokenizer.eos_token_id)
        return ids, [-1.0] * len(ids)

    loop = rollout_loop(tokenizer, rows, max_tokens=16, adaptive_max_tokens=False)
    return [
        s
        for group_id, row in enumerate((rows * 3)[:10])
        for s in run_group(loop, group_id, row, sample, 3)
    ]


def scripted_rewards():
    group_rewards = EBFTScorer.group_rewards
    calls = itertools.count()

    def rewards(self, *args, **kwargs):
        values, metrics = group_rewards(self, *args, **kwargs)
        if next(calls) not in TRAINED_GROUP_CALLS:
            values = torch.ones_like(values)
        return values, metrics

    EBFTScorer.group_rewards = rewards


def same_on_every_rank(value: float) -> bool:
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    return len(set(gathered)) == 1


def main(output_dir):
    # Gloo alone, so barriers and object collectives stay on the CPU even where an
    # accelerator (MPS) is present.
    dist.init_process_group(backend="gloo")
    state = PartialState(cpu=True)
    barrier = dist.barrier

    def cpu_barrier(*args, device_ids=None, **kwargs):
        # accelerate names a device index, which torch places on the accelerator.
        return barrier(*args, **kwargs)

    dist.barrier = cpu_barrier
    packing.tree_attention = reference_tree_attention
    async_grpo_trainer.get_peak_flops_per_device = lambda *a, **k: None
    scripted_rewards()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "right"
    rows = [
        {"prompt": prompt, "ground_truth": truth}
        for prompt, truth in zip(PROMPTS * 2, GROUND_TRUTH * 2, strict=True)
    ]
    rollouts = samples(tokenizer, rows)
    model = get_peft_model(
        tiny_model(tokenizer).train(),
        LoraConfig(r=8, target_modules=["q_proj", "v_proj"]),
    )
    args = AxolotlAsyncEBFTConfig(
        output_dir=output_dir,
        max_steps=20,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_generations=3,
        learning_rate=1e-3,
        bf16=False,
        use_cpu=True,
        token_budget=0,
        scoring_token_budget=12,
        mask_truncated_completions=True,
        report_to=[],
        save_strategy="no",
        logging_steps=1,
        gradient_checkpointing=False,
        ebft_feature_layers=LAYER_FRACTIONS,
        ebft_embed_method="mean_pooling",
        ebft_diversity_coef=0.5,
    )
    transfer = StubTransfer()
    trainer = AxolotlAsyncEBFTTrainer(
        model,
        args=args,
        train_dataset=Dataset.from_list(rows),
        processing_class=tokenizer,
        rollout_worker=StubWorker(rollouts),
        weight_transfer=transfer,
    )
    assert state.num_processes == 2, state.num_processes
    before = {
        n: p.detach().clone()
        for n, p in trainer.model.named_parameters()
        if p.requires_grad
    }

    try:
        trainer.train()
        raise AssertionError("the run did not stall")
    except RuntimeError as error:
        assert "No rollout group trained in the last 4 groups" in str(error), error

    steps = trainer.state.global_step
    assert steps >= 6 and same_on_every_rank(steps), steps
    trained = {
        n: p.detach() for n, p in trainer.model.named_parameters() if p.requires_grad
    }
    assert any(not torch.equal(before[n], p) for n, p in trained.items())
    checksum = float(sum(p.double().sum() for p in trained.values()))
    assert same_on_every_rank(checksum), "replicas diverged"
    # Steps 1 and 3 updated; the empty steps left Adam as they found it.
    assert {int(s["step"]) for s in trainer.optimizer.optimizer.state.values()} == {2}
    if state.is_main_process:
        # The cold sync and steps 1 and 3; no other step changed the weights.
        assert transfer.syncs == 3, transfer.syncs
    print(f"PASS two-rank-scoring rank={state.process_index} steps={steps}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main(sys.argv[1])
