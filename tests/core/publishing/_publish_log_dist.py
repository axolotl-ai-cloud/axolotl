"""Worker for test_publishing_log_dist: a 2-rank gloo Trainer publishing every step."""

import json
import sys
import time
from pathlib import Path

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments


class Publisher:
    def __init__(self):
        self.published: list[int] = []

    def publish(self, buffers, step):
        return {"identity": f"id-{step}", "action": "commit", "full": 0}

    def restore(self, buffers, step):
        return True

    def close(self):
        pass


class WaitForPreviousPublish(TrainerCallback):
    """Hold rank 0 at each step end until the previous step is published, so the
    publish callback has metrics to log at every step."""

    def __init__(self, publisher):
        self.publisher = publisher

    def on_step_end(self, args, state, control, **kwargs):
        deadline = time.monotonic() + 30
        while state.global_step - 1 not in self.publisher.published:
            if time.monotonic() > deadline:
                raise TimeoutError(f"step {state.global_step - 1} never published")
            time.sleep(0.01)
        return control


class Tokens(torch.utils.data.Dataset):
    def __len__(self):
        return 64

    def __getitem__(self, index):
        ids = torch.randint(0, 32, (8,), generator=torch.Generator().manual_seed(index))
        return {"input_ids": ids, "labels": ids}


def main(root: str) -> None:
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    from axolotl.core.publishing.callback import StepPublishCallback

    torch.manual_seed(0)
    model = get_peft_model(
        LlamaForCausalLM(
            LlamaConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
            )
        ),
        LoraConfig(r=2, target_modules=["q_proj"]),
    )
    args = TrainingArguments(
        output_dir=root,
        use_cpu=True,
        max_steps=3,
        per_device_train_batch_size=2,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        disable_tqdm=True,
        ddp_timeout=60,
    )
    trainer = Trainer(model=model, args=args, train_dataset=Tokens())
    publisher = Publisher()
    trainer.add_callback(WaitForPreviousPublish(publisher))
    trainer.add_callback(
        StepPublishCallback(
            lambda **kwargs: publisher,
            every=1,
            trainer=trainer,
            publish_at_start=True,
            on_published=publisher.published.append,
        )
    )
    trainer.train()
    rank = trainer.args.process_index
    Path(root, f"log-{rank}.json").write_text(json.dumps(trainer.state.log_history))


if __name__ == "__main__":
    main(sys.argv[1])
