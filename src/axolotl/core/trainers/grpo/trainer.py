"""Axolotl GRPO trainer"""

import warnings
from collections import defaultdict
from contextlib import nullcontext
from typing import Any

import datasets
import torch
import torch.distributed as dist
from accelerate.utils import (
    broadcast_object_list,
    gather,
    gather_object,
    is_peft_model,
    set_seed,
)
from datasets import Dataset, IterableDataset
from torch import nn
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Sampler,
)
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer_utils import seed_worker
from transformers.utils import is_peft_available
from trl import GRPOTrainer
from trl.data_utils import (
    apply_chat_template,
    is_conversational,
    maybe_apply_chat_template,
)
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.extras.vllm_client import VLLMClient
from trl.import_utils import (
    is_deepspeed_available,
    is_rich_available,
    is_vllm_available,
)
from trl.models import (
    create_reference_model,
    prepare_deepspeed,
    unwrap_model_for_generation,
)
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import RewardFunc
from trl.trainer.utils import (
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
)

from axolotl.core.trainers.grpo.sampler import SequenceParallelRepeatRandomSampler
from axolotl.core.trainers.mixins import RngLoaderMixin, SchedulerMixin
from axolotl.monkeypatch.attention.ring_attn.patch import get_ring_attn_group

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_deepspeed_available():
    import deepspeed

if is_wandb_available():
    import wandb


class AxolotlGRPOTrainer(RngLoaderMixin, SchedulerMixin, GRPOTrainer):
    """Extend the base GRPOTrainer for axolotl helpers"""

    _tag_names = ["trl", "grpo", "axolotl"]

    def __init__(
        self,
        model: str | PreTrainedModel,
        reward_funcs: RewardFunc | list[RewardFunc],
        args: GRPOConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: (
            Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None
        ) = None,
        processing_class: PreTrainedTokenizerBase | None = None,
        reward_processing_classes: (
            PreTrainedTokenizerBase | list[PreTrainedTokenizerBase] | None
        ) = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[
            torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None
        ] = (None, None),
        peft_config: "PeftConfig | None" = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if (
                isinstance(torch_dtype, torch.dtype)
                or torch_dtype == "auto"
                or torch_dtype is None
            ):
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False
                if args.gradient_checkpointing
                else model_init_kwargs.get("use_cache")
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            if not is_peft_available():
                raise ImportError(
                    "PEFT is required to use `peft_config`. Run `pip install peft`."
                )
            model = get_peft_model(model, peft_config)

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Reference model
        self.beta = args.beta
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                model_id, **model_init_kwargs
            )
        elif is_peft_model(model):
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        else:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(
                model.config._name_or_path, padding_side="left"
            )

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError(
                    "The number of reward processing classes must match the number of reward functions."
                )

        for i, (reward_processing_class, reward_func) in enumerate(
            zip(reward_processing_classes, reward_funcs)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(
                        reward_func.config._name_or_path
                    )
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = (
                        reward_processing_class.eos_token
                    )
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = (
            args.max_completion_length
        )  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.use_vllm = args.use_vllm

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        self.epsilon_low = args.epsilon
        self.epsilon_high = (
            args.epsilon_high if args.epsilon_high is not None else args.epsilon
        )
        # Tracks the number of iterations (forward + backward passes), including those within a grad accum cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = [None] * args.gradient_accumulation_steps

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self.log_completions = args.log_completions

        Trainer.__init__(
            self,
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Get number of SP groups (number of processes divided by SP degree)
        num_processes = self.accelerator.num_processes
        num_sp_groups = num_processes // self.args.sequence_parallel_degree

        # Calculate batch size per SP group (not per process)
        sp_group_batch_size = self.args.per_device_train_batch_size * num_sp_groups
        possible_values = [
            n_gen
            for n_gen in range(2, sp_group_batch_size + 1)
            if (sp_group_batch_size) % n_gen == 0
        ]

        if self.num_generations not in possible_values:
            raise ValueError(
                f"The batch size per SP group ({num_sp_groups} x {self.args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                f"configuration, the valid values for the number of generations are: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            # If sequence parallelism is enabled, calculate batch size per SP group
            sp_group_eval_batch_size = args.per_device_eval_batch_size * num_sp_groups
            possible_values = [
                n_gen
                for n_gen in range(2, sp_group_eval_batch_size + 1)
                if (sp_group_eval_batch_size) % n_gen == 0
            ]

            if self.num_generations not in possible_values:
                raise ValueError(
                    f"With sequence parallelism (degree {self.args.sequence_parallel_degree}), "
                    f"the eval batch size per SP group ({num_sp_groups} x {self.args.per_device_eval_batch_size}) "
                    f"must be evenly divisible by the number of generations per prompt "
                    f"({self.num_generations}). Given the current eval batch size, "
                    f"the valid values for the number of generations are: {possible_values}."
                )

        # # Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations
        # num_processes = self.accelerator.num_processes
        # global_batch_size = args.per_device_train_batch_size * num_processes
        # possible_values = [
        #     n_gen
        #     for n_gen in range(2, global_batch_size + 1)
        #     if (global_batch_size) % n_gen == 0
        # ]
        # if self.num_generations not in possible_values:
        #     raise ValueError(
        #         f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
        #         f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
        #         f"batch size, the valid values for the number of generations are: {possible_values}."
        #     )
        # if self.args.eval_strategy != "no":
        #     global_batch_size = args.per_device_eval_batch_size * num_processes
        #     possible_values = [
        #         n_gen
        #         for n_gen in range(2, global_batch_size + 1)
        #         if (global_batch_size) % n_gen == 0
        #     ]
        #     if self.num_generations not in possible_values:
        #         raise ValueError(
        #             f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
        #             f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
        #             f"eval batch size, the valid values for the number of generations are: {possible_values}."
        #         )

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )

            if self.accelerator.is_main_process:
                self.vllm_client = VLLMClient(
                    args.vllm_server_host,
                    args.vllm_server_port,
                    connection_timeout=args.vllm_server_timeout,
                )

            # vLLM specific sampling arguments
            self.guided_decoding_regex = args.vllm_guided_decoding_regex

            self._last_loaded_step = (
                0  # tag to avoid useless loading during grad accumulation
            )

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                pad_token_id=processing_class.pad_token_id,
                bos_token_id=processing_class.bos_token_id,
                eos_token_id=processing_class.eos_token_id,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
                repetition_penalty=self.repetition_penalty,
                cache_implementation=args.cache_implementation,
            )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(
                    self.ref_model, evaluation_mode=True
                )

        if args.sync_ref_model:
            self.add_callback(
                SyncRefModelCallback(
                    ref_model=self.ref_model, accelerator=self.accelerator
                )
            )

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(
                    reward_func, evaluation_mode=True
                )

        # Initialize the SP group
        self.sp_group = get_ring_attn_group()
        self.local_rank = dist.get_rank(group=self.sp_group)
        self.local_world_size = dist.get_world_size(group=self.sp_group)

    def _get_train_sampler(self) -> Sampler:
        # Get distributed training info
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        effective_batch_size = (
            self.args.per_device_train_batch_size
            * world_size
            * self.args.gradient_accumulation_steps
        )

        return SequenceParallelRepeatRandomSampler(
            dataset=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=effective_batch_size
            // self.num_generations
            // self.args.sequence_parallel_degree,
            repeat_count=self.num_iterations,
            sequence_parallel_degree=self.args.sequence_parallel_degree,
            world_size=world_size,
            rank=rank,
            shuffle=True,
            seed=self.args.seed,
            drop_last=True,
        )

    def _create_dataloader_params(self, is_eval=False, custom_batch_size=None):
        """Create common dataloader parameters for train or eval."""
        batch_size = custom_batch_size or (
            self.args.eval_batch_size if is_eval else self._train_batch_size
        )

        params = {
            "batch_size": batch_size,
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        # Add persistent workers only for training
        if not is_eval and hasattr(self.args, "dataloader_persistent_workers"):
            params["persistent_workers"] = self.args.dataloader_persistent_workers

        # Add prefetch factor if specified
        if self.args.dataloader_prefetch_factor:
            params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return params

    def _prepare_dataloader(
        self, dataset, sampler, is_eval=False, custom_batch_size=None
    ):
        """Prepare a dataloader with the given dataset and sampler."""
        # Get base parameters
        dataloader_params = self._create_dataloader_params(is_eval, custom_batch_size)

        # Add sampler configuration
        if not isinstance(dataset, torch.utils.data.IterableDataset):
            if isinstance(sampler, BatchSampler):
                # batch_size and batch_sampler are mutually exclusive
                dataloader_params["batch_sampler"] = sampler
                del dataloader_params["batch_size"]
            else:
                dataloader_params["sampler"] = sampler
                dataloader_params["drop_last"] = self.args.dataloader_drop_last

            if not is_eval:
                dataloader_params["worker_init_fn"] = seed_worker

        # Create the dataloader
        dataloader = DataLoader(dataset, **dataloader_params)

        if self.args.sample_packing and (
            (not is_eval and not self.args.pretraining)
            or (is_eval and self.args.eval_sample_packing is not False)
        ):
            self.accelerator.even_batches = False

        # Return unprepared dataloader if using sequence parallelism
        # TODO(djsaunde): We might be able to use `accelerate`'s dataloader preparation
        # if we use `dispatch_batches` and `slice_fn_for_dispatch` properly (i.e.,
        # slice each batch along the sequence dimension).
        if self.args.sequence_parallel_degree > 1:
            return dataloader

        # Otherwise prepare with accelerator
        return self.accelerator.prepare_data_loader(dataloader)

    def get_train_dataloader(self) -> DataLoader:
        """Get dataloader for training"""
        train_dataset = self.train_dataset
        data_collator = self.data_collator  # type: ignore

        # Initialize SP group attributes if sequence parallelism is enabled
        if self.args.sequence_parallel_degree > 1:
            self.sp_group = get_ring_attn_group()
            self.local_rank = dist.get_rank(group=self.sp_group)
            self.local_world_size = dist.get_world_size(group=self.sp_group)

        # Handle dataset preprocessing
        if isinstance(train_dataset, datasets.Dataset):
            # Add debug print before any modifications
            if self.args.sample_packing and not self.args.pretraining:
                train_dataset = train_dataset.remove_columns(["length"])
            if not self.args.sample_packing or self.args.pretraining:
                train_dataset = self._remove_unused_columns(
                    train_dataset, description="training"
                )
        else:
            self.data_collator = self._get_collator_with_removed_columns(
                data_collator,
                description="training",
            )

        # Get sampler and create dataloader
        sampler = self._get_train_sampler()
        dataloader = self._prepare_dataloader(train_dataset, sampler, is_eval=False)

        return dataloader

    @profiling_decorator
    def _move_model_to_vllm(self):
        # For DeepSpeed ZeRO-3, we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        gather_if_zero3 = (
            deepspeed.zero.GatheredParameters if zero_stage_3 else nullcontext
        )

        if is_peft_model(self.model):
            # With PEFT and DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging, as merging
            # adapters in a sharded manner is not supported.
            with gather_if_zero3(list(self.model.parameters())):
                self.model.merge_adapter()

                # Update vLLM weights while parameters are gathered
                for name, param in self.model.named_parameters():
                    # When using PEFT, we need to recover the original parameter name and discard some parameters
                    name = (
                        name.removeprefix("base_model.model.")
                        .removeprefix("base_model.model.")
                        .replace(".base_layer", "")
                    )
                    if self.model.prefix in name:
                        continue
                    # When module to save, remove its prefix and discard the original module
                    if "original_module" in name:
                        continue
                    name = name.replace("modules_to_save.default.", "")

                    if self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(name, param.data)

                # Unmerge adapters while parameters are still gathered
                self.model.unmerge_adapter()
                # Parameters will automatically be repartitioned when exiting the context
        else:
            # For non-PEFT models, simply gather and update each parameter individually.
            for name, param in self.model.named_parameters():
                with gather_if_zero3([param]):
                    if self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(name, param.data)

        # Reset cache on main process
        if self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()

    def _generate_and_score_completions(
        self, inputs: dict[str | torch.Tensor | Any]
    ) -> dict[str, torch.Tensor | Any]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs
        ]
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)

        prompt_ids, prompt_mask = (
            prompt_inputs["input_ids"],
            prompt_inputs["attention_mask"],
        )

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions using either vLLM or regular generation
        if self.args.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                # prompt individually.
                # ordered_set_of_prompts = all_prompts_text[:: self.num_generations]
                ordered_set_of_prompts = all_prompts_text[
                    :: self.num_generations * self.args.sequence_parallel_degree
                ]
                with profiling_context(self, "vLLM.generate"):
                    completion_ids = self.vllm_client.generate(
                        prompts=ordered_set_of_prompts,
                        n=self.num_generations,
                        repetition_penalty=self.repetition_penalty,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=-1 if self.top_k is None else self.top_k,
                        min_p=0.0 if self.min_p is None else self.min_p,
                        max_tokens=self.max_completion_length,
                        guided_decoding_regex=self.guided_decoding_regex,
                    )
            else:
                completion_ids = [None] * (
                    len(all_prompts_text) // self.args.sequence_parallel_degree
                )

            # Broadcast the completions from the main process to all processes
            completion_ids = broadcast_object_list(completion_ids, from_process=0)

            # Determine the appropriate slice based on sequence parallelism
            if self.args.sequence_parallel_degree > 1:
                # Calculate SP group ID (which group of ranks this rank belongs to)
                sp_group_id = self.accelerator.process_index // self.local_world_size

                # Calculate the start index for this SP group
                sp_group_start = sp_group_id * len(prompts) * self.local_world_size

                # All ranks in the same SP group get the same data slice
                process_slice = slice(
                    sp_group_start,
                    sp_group_start + len(prompts),
                )
                completion_ids = completion_ids[process_slice]
            else:
                # Original behavior for non-sequence parallel case
                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                completion_ids = completion_ids[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [
                torch.tensor(ids, device=device) for ids in completion_ids
            ]
            completion_ids = pad(
                completion_ids, padding_value=self.processing_class.pad_token_id
            )
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # Regular generation path
            with unwrap_model_for_generation(
                self.model_wrapped,
                self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation,
            ) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(
                    prompt_ids,
                    attention_mask=prompt_mask,
                    generation_config=self.generation_config,
                )

            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full(
            (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(
            is_eos.size(0), -1
        )
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)
        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens

        with torch.no_grad():
            if self.args.sequence_parallel_degree > 1:
                # Pad sequence to be divisible by SP degree if needed
                total_seq_len = prompt_completion_ids.shape[1]
                if total_seq_len % self.local_world_size != 0:
                    pad_len = self.local_world_size - (
                        total_seq_len % self.local_world_size
                    )
                    pad_token_id = self.processing_class.pad_token_id or 0

                    # Pad input_ids and attention_mask
                    padding = torch.full(
                        (prompt_completion_ids.shape[0], pad_len),
                        pad_token_id,
                        dtype=prompt_completion_ids.dtype,
                        device=prompt_completion_ids.device,
                    )
                    prompt_completion_ids = torch.cat(
                        [prompt_completion_ids, padding], dim=1
                    )

                    attn_padding = torch.zeros(
                        (attention_mask.shape[0], pad_len),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    attention_mask = torch.cat([attention_mask, attn_padding], dim=1)

                    total_seq_len += pad_len
                    logits_to_keep += pad_len

                # Split the sequence
                slice_size = total_seq_len // self.local_world_size
                start = self.local_rank * slice_size
                end = start + slice_size

                # Get our slice
                prompt_completion_ids = prompt_completion_ids[:, start:end]
                attention_mask = attention_mask[:, start:end]

                # Calculate how many completion tokens each rank should process
                prompt_len = prompt_ids.size(1)
                completion_len = completion_ids.size(
                    1
                )  # This is equal to logits_to_keep

                # Calculate where our slice starts and ends relative to the completion tokens
                if start >= prompt_len:
                    # Slice starts within the completion section
                    start_in_completion = start - prompt_len
                    end_in_completion = min(end - prompt_len, completion_len)
                    logits_to_keep = end_in_completion - start_in_completion
                    completion_mask = completion_mask[
                        :, start_in_completion:end_in_completion
                    ]
                elif end <= prompt_len:
                    # Slice is entirely within the prompt section (no completion tokens)
                    logits_to_keep = 0
                    completion_mask = torch.zeros(
                        (completion_mask.size(0), 0), device=completion_mask.device
                    )
                else:
                    # Slice contains the boundary between prompt and completion
                    start_in_completion = 0
                    end_in_completion = min(end - prompt_len, completion_len)
                    logits_to_keep = end_in_completion - start_in_completion
                    completion_mask = completion_mask[
                        :, start_in_completion:end_in_completion
                    ]

            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                print(f"{dist.get_rank()}: prompt_completion_ids.shape: {prompt_completion_ids.shape}")
                print(f"{dist.get_rank()}: attention_mask.shape: {attention_mask.shape}")
                print(f"{dist.get_rank()}: logits_to_keep: {logits_to_keep}")

                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                    )

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = (
                    prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                )
                completions.append(
                    [{"role": "assistant", "content": bootstrap + completion}]
                )
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(
            len(prompts), len(self.reward_funcs), device=device
        )
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(
                reward_func, nn.Module
            ):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = (
                    f"reward {reward_func.config._name_or_path.split('/')[-1]}"
                )
            else:
                reward_func_name = reward_func.__name__
            with profiling_context(self, reward_func_name):
                if isinstance(
                    reward_func, nn.Module
                ):  # Module instead of PretrainedModel for compat with compiled models
                    if is_conversational(inputs[0]):
                        messages = [
                            {"messages": p + c} for p, c in zip(prompts, completions)
                        ]
                        texts = [
                            apply_chat_template(x, reward_processing_class)["text"]
                            for x in messages
                        ]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        text=texts,
                        return_tensors="pt",
                        padding=True,
                        padding_side="right",
                        add_special_tokens=False,
                    )
                    reward_inputs = Trainer._prepare_inputs(self, reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[
                            :, 0
                        ]  # Shape (B*G,)
                else:
                    # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                    keys = [
                        key for key in inputs[0] if key not in ["prompt", "completion"]
                    ]
                    reward_kwargs = {
                        key: [example[key] for example in inputs] for key in keys
                    }
                    output_reward_func = reward_func(
                        prompts=prompts, completions=completions, **reward_kwargs
                    )
                    # Convert None values to NaN
                    output_reward_func = [
                        reward if reward is not None else torch.nan
                        for reward in output_reward_func
                    ]

                    rewards_per_func[:, i] = torch.tensor(
                        output_reward_func, dtype=torch.float32, device=device
                    )

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = (
                torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            )
            row_reward_kwargs = {
                key: value[nan_row_idx] for key, value in reward_kwargs.items()
            }
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)

        # Apply weights to each reward function's output and sum
        rewards = (
            rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
        ).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        advantages = rewards - mean_grouped_rewards
        if self.args.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if mode == "train":
            self._total_train_tokens += (
                self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
            )
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        completion_length = (
            self.accelerator.gather_for_metrics(completion_mask.sum(1))
            .float()
            .mean()
            .item()
        )
        self._metrics[mode]["completion_length"].append(completion_length)

        # Calculate mean reward per function, but only for samples where the function was applied
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(
                reward_func, nn.Module
            ):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            # Only calculate mean for samples where this reward function was applied (non-NaN values)
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        if (
            self.log_completions
            and self.state.global_step % self.args.logging_steps == 0
        ):
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)
            rewards_to_log = rewards.tolist()

            if self.accelerator.is_main_process:
                if is_rich_available():
                    print_prompt_completions_sample(
                        prompts_to_log,
                        completions_to_log,
                        rewards_to_log,
                        self.state.global_step,
                    )
                if (
                    self.args.report_to
                    and "wandb" in self.args.report_to
                    and wandb.run is not None
                ):
                    import pandas as pd

                    # For logging
                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards.tolist(),
                    }
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }

    @profiling_decorator
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        # Unpack inputs
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        if self.args.sequence_parallel_degree > 1:
            # Pad sequence to be divisible by SP degree if needed
            total_seq_len = input_ids.shape[1]
            if total_seq_len % self.local_world_size != 0:
                pad_len = self.local_world_size - (
                    total_seq_len % self.local_world_size
                )
                pad_token_id = self.processing_class.pad_token_id or 0

                # Pad input_ids and attention_mask
                padding = torch.full(
                    (input_ids.shape[0], pad_len),
                    pad_token_id,
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
                input_ids = torch.cat([input_ids, padding], dim=1)

                attn_padding = torch.zeros(
                    (attention_mask.shape[0], pad_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([attention_mask, attn_padding], dim=1)

                total_seq_len += pad_len
                logits_to_keep += pad_len

            # Split the sequence
            slice_size = total_seq_len // self.local_world_size
            start = self.local_rank * slice_size
            end = start + slice_size

            # Get our slice
            input_ids_slice = input_ids[:, start:end]
            attention_mask_slice = attention_mask[:, start:end]

            # Calculate how many completion tokens each rank should process
            prompt_len = prompt_ids.size(1)
            completion_len = completion_ids.size(1)  # This is equal to logits_to_keep

            # Calculate where our slice starts and ends relative to the completion tokens
            if start >= prompt_len:
                # Slice starts within the completion section
                start_in_completion = start - prompt_len
                end_in_completion = min(end - prompt_len, completion_len)
                local_logits_to_keep = end_in_completion - start_in_completion
                completion_mask = completion_mask[
                    :, start_in_completion:end_in_completion
                ]
            elif end <= prompt_len:
                # Slice is entirely within the prompt section (no completion tokens)
                local_logits_to_keep = 0
                completion_mask = torch.zeros(
                    (completion_mask.size(0), 0), device=completion_mask.device
                )
            else:
                # Slice contains the boundary between prompt and completion
                start_in_completion = 0
                end_in_completion = min(end - prompt_len, completion_len)
                local_logits_to_keep = end_in_completion - start_in_completion
                completion_mask = completion_mask[
                    :, start_in_completion:end_in_completion
                ]

            # Run model on our slice
            if local_logits_to_keep > 0:
                # Get logits with enough context to compute log probs
                logits = model(
                    input_ids=input_ids_slice,
                    attention_mask=attention_mask_slice,
                    logits_to_keep=local_logits_to_keep + 1,
                ).logits

                # Remove the last prediction token on the last rank
                # if self.local_rank == self.local_world_size - 1:
                #     logits = logits[:, :-1, :]
                logits = logits[:, :-1, :]

                # Compute log probabilities on our local slice
                local_input_ids = input_ids_slice[:, -local_logits_to_keep:]
                logits = logits / self.temperature
                per_token_logps = selective_log_softmax(logits, local_input_ids)
            else:
                # This rank doesn't have any tokens to keep
                per_token_logps = torch.zeros(
                    (input_ids.shape[0], 0),
                    dtype=torch.float32,
                    device=input_ids.device,
                )
        else:
            per_token_logps = super()._get_per_token_logps(
                model, input_ids, attention_mask, logits_to_keep
            )

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip its computation
        # and use per_token_logps.detach() instead.
        old_per_token_logps = (
            inputs["old_per_token_logps"]
            if self.num_iterations > 1
            else per_token_logps.detach()
        )
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if dist.get_rank() == 0:
            import ipdb

            ipdb.set_trace()
        dist.barrier()
        if dist.get_rank() == 1:
            import ipdb

            ipdb.set_trace()
        dist.barrier()

        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        # Log metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(
                self.accelerator.gather_for_metrics(mean_kl).mean().item()
            )

        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(
            self.accelerator.gather_for_metrics(clip_ratio).mean().item()
        )

        return loss

    # def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
    #     # if self.args.sequence_parallel_degree > 1:
    #     if False:
    #         # Handle padding to make sequence length divisible by world size
    #         total_seq_len = input_ids.shape[1]
    #         if total_seq_len % self.local_world_size != 0:
    #             # Pad to make divisible
    #             pad_len = self.local_world_size - (
    #                 total_seq_len % self.local_world_size
    #             )
    #             pad_token_id = self.processing_class.pad_token_id or 0

    #             # Pad input_ids
    #             padding = torch.full(
    #                 (input_ids.shape[0], pad_len),
    #                 pad_token_id,
    #                 dtype=input_ids.dtype,
    #                 device=input_ids.device,
    #             )
    #             input_ids = torch.cat([input_ids, padding], dim=1)

    #             # Pad attention mask
    #             if attention_mask is not None:
    #                 attn_padding = torch.zeros(
    #                     (attention_mask.shape[0], pad_len),
    #                     dtype=attention_mask.dtype,
    #                     device=attention_mask.device,
    #                 )
    #                 attention_mask = torch.cat([attention_mask, attn_padding], dim=1)

    #             total_seq_len += pad_len
    #             logits_to_keep += pad_len

    #         # Share logits_to_keep across ranks to ensure consistency
    #         lt_keep = torch.tensor([logits_to_keep], device=input_ids.device)
    #         dist.broadcast(lt_keep, src=0, group=self.sp_group)
    #         logits_to_keep = lt_keep.item()

    #         # Split the sequence across ranks
    #         slice_size = total_seq_len // self.local_world_size
    #         start = self.local_rank * slice_size
    #         end = start + slice_size

    #         # Slice for this rank
    #         input_ids_slice = input_ids[:, start:end]
    #         attention_mask_slice = (
    #             attention_mask[:, start:end] if attention_mask is not None else None
    #         )

    #         # Calculate how many tokens this rank needs to keep
    #         tokens_before_slice = self.local_rank * slice_size
    #         local_logits_to_keep = 0

    #         if tokens_before_slice < logits_to_keep:
    #             # This rank has tokens we need to keep
    #             local_logits_to_keep = min(
    #                 slice_size, logits_to_keep - tokens_before_slice
    #             )

    #         # Run the model on our slice
    #         if local_logits_to_keep > 0:
    #             logits = model(
    #                 input_ids=input_ids_slice,
    #                 attention_mask=attention_mask_slice,
    #                 logits_to_keep=local_logits_to_keep + 1,
    #             ).logits
    #             if self.local_rank == self.local_world_size - 1:
    #                 logits = logits[:, :-1, :]

    #             # Get the relevant input_ids for computing log probs
    #             # Ensure this is the correct slice that corresponds to the logits
    #             relevant_input_ids = input_ids_slice[:, -local_logits_to_keep:]
    #         else:
    #             # Create empty logits with correct shape if we don't need to keep any
    #             vocab_size = model.config.vocab_size
    #             logits = torch.zeros(
    #                 (input_ids.shape[0], 0, vocab_size),
    #                 dtype=torch.float32,
    #                 device=input_ids.device,
    #             )
    #             relevant_input_ids = torch.zeros(
    #                 (input_ids.shape[0], 0),
    #                 dtype=torch.float32,
    #                 device=input_ids.device,
    #             )

    #         # Temperature scaling
    #         logits = logits / self.temperature

    #         print(f"{dist.get_rank()}: logits.shape: {logits.shape}")
    #         print(
    #             f"{dist.get_rank()}: relevant_input_ids.shape: {relevant_input_ids.shape}"
    #         )

    #         return selective_log_softmax(logits, relevant_input_ids)

    #         # All-gather results across SP group with proper shape handling
    #         # local_shape = torch.tensor([logits.shape[1]], device=logits.device)
    #         # all_shapes = [
    #         #     torch.zeros_like(local_shape) for _ in range(self.local_world_size)
    #         # ]
    #         # dist.all_gather(all_shapes, local_shape, group=self.sp_group)

    #         # # Create full tensor to hold the complete result
    #         # full_logits = torch.zeros(
    #         #     (input_ids.shape[0], logits_to_keep, model.config.vocab_size),
    #         #     dtype=torch.float32,
    #         #     device=input_ids.device,
    #         # )

    #         # # Calculate positions for each rank's contribution
    #         # position = 0
    #         # for i in range(self.local_world_size):
    #         #     shape = all_shapes[i].item()
    #         #     if i < self.local_rank:
    #         #         position += shape

    #         # # Add our contribution to the full tensor
    #         # if local_logits_to_keep > 0:
    #         #     # Make sure we're not exceeding bounds
    #         #     end_pos = min(position + local_logits_to_keep, logits_to_keep)
    #         #     copy_size = end_pos - position

    #         #     if dist.get_rank() == 0:
    #         #         import ipdb; ipdb.set_trace()
    #         #     dist.barrier()
    #         #     if dist.get_rank() == 1:
    #         #         import ipdb; ipdb.set_trace()
    #         #     dist.barrier()

    #         #     if copy_size > 0:
    #         #         full_logits[:, position:end_pos, :] = logits[:, :copy_size, :]

    #         # # Combine results via all-reduce
    #         # dist.all_reduce(full_logits, op=dist.ReduceOp.SUM, group=self.sp_group)

    #         # # Remove the last prediction token
    #         # full_logits = full_logits[:, :-1, :]

    #         # # Get the relevant input_ids for computing log probs
    #         # # Ensure this is the correct slice that corresponds to the logits
    #         # relevant_input_ids = input_ids[:, -logits_to_keep:]

    #         # # Temperature scaling
    #         # full_logits = full_logits / self.temperature

    #         # return selective_log_softmax(full_logits, relevant_input_ids)
    #     else:
    #         return super()._get_per_token_logps(
    #             model, input_ids, attention_mask, logits_to_keep
    #         )
