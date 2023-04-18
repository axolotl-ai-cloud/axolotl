import math
import bitsandbytes as bnb
import transformers
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
from transformers import EarlyStoppingCallback
from transformers.trainer_pt_utils import get_parameter_names


def setup_trainer(cfg, train_dataset, eval_dataset, model, tokenizer):
    total_num_steps = int(
        math.ceil(len(train_dataset) * cfg.num_epochs / cfg.batch_size)
    )
    warmup_steps = cfg.warmup_steps if cfg.warmup_steps else min(int(0.03 * total_num_steps), 100)
    logging_steps = max(min(int(0.005 * total_num_steps), 10), 1)
    save_steps = eval_steps = cfg.save_steps if cfg.save_steps else min(int(0.05 * total_num_steps), 200)

    training_arguments_kwargs = {}
    if cfg.bf16 == "full":
        training_arguments_kwargs["bf16_full_eval"] = True
    else:
        training_arguments_kwargs["bf16"] = cfg.bf16
    training_arguments_kwargs["tf32"] = cfg.tf32
    training_arguments_kwargs["warmup_steps"] = warmup_steps
    training_arguments_kwargs["logging_steps"] = logging_steps
    if cfg.gradient_checkpointing is not None:
        training_arguments_kwargs["gradient_checkpointing"] = cfg.gradient_checkpointing

    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=cfg.micro_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_epochs,
        learning_rate=cfg.learning_rate,
        evaluation_strategy="steps" if cfg.val_set_size > 0 else "no",
        save_strategy="steps",
        eval_steps=eval_steps if cfg.val_set_size > 0 else None,
        save_steps=save_steps,
        output_dir=cfg.output_dir,
        save_total_limit=3,
        load_best_model_at_end=True if cfg.val_set_size > 0 else False,
        ddp_find_unused_parameters=False if cfg.ddp else None,
        group_by_length=cfg.group_by_length,
        report_to="wandb" if cfg.use_wandb else None,
        run_name=cfg.wandb_run_id if cfg.use_wandb else None,
        **training_arguments_kwargs,
    )

    trainer_kwargs = {}

    if cfg.load_in_8bit and not cfg.load_4bit:
        decay_parameters = get_parameter_names(model, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if n in decay_parameters],
                "weight_decay": training_args.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if n not in decay_parameters
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = bnb.optim.Adam8bit(
            optimizer_grouped_parameters,
            betas=(training_args.adam_beta1, training_args.adam_beta2),
            eps=training_args.adam_epsilon,
            lr=training_args.learning_rate,
        )

        if cfg.lr_scheduler == "one_cycle":
            lr_scheduler_kwargs = (
                cfg.lr_scheduler_kwargs if cfg.lr_scheduler_kwargs else {}
            )
            lr_scheduler = OneCycleLR(
                optimizer,
                cfg.learning_rate,
                total_steps=total_num_steps,
                **lr_scheduler_kwargs,
            )
        else:
            lr_scheduler = transformers.get_cosine_schedule_with_warmup(
                optimizer,
                training_args.warmup_steps,
                total_num_steps,
            )
        trainer_kwargs["optimizers"] = (optimizer, lr_scheduler)

    # TODO on_save callback to sync checkpoints to GCP/AWS in background
    if cfg.early_stopping_patience:
        early_stop_cb = EarlyStoppingCallback(
            cfg.early_stopping_patience,
        )
        trainer_kwargs["callbacks"] = [early_stop_cb]

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        **trainer_kwargs,
    )

    return trainer
