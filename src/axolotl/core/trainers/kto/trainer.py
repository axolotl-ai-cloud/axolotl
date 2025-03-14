"""
KTO trainer implementation for Axolotl.
"""

from __future__ import annotations

import inspect
import logging
import warnings
from contextlib import nullcontext
from typing import Any, Callable, Literal, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (
    AutoModelForCausalLM,
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import EvalLoopOutput
from trl import KTOTrainer
from trl.trainer.kto_config import KTOConfig
from trl.trainer.utils import KTODataCollatorWithPadding, pad_to_length

from axolotl.core.trainers.base import SchedulerMixin

# Check if PEFT is available
try:
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training, peft_module_casting_to_bf16
    is_peft_available = True
except ImportError:
    is_peft_available = False

LOG = logging.getLogger("axolotl.core.trainers.kto")


class AxolotlKTOTrainer(SchedulerMixin, Trainer):
    """
    Extend the base KTOTrainer for axolotl helpers
    """

    tag_names = ["axolotl", "kto"]

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module, str] = None,
        args: KTOConfig = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        data_collator: Optional[DataCollator] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional[dict] = None,
        compute_metrics: Optional[Callable[[EvalLoopOutput], dict]] = None,
        dataset_tags=None,
        model_adapter_name: Optional[str] = None,
        ref_adapter_name: Optional[str] = None,
    ):
        self.dataset_tags = dataset_tags
        self._tag_names = ["trl", "kto"]
        if hasattr(self, "tag_names"):
            self._tag_names.extend(self.tag_names)

        if type(args) is TrainingArguments:
            raise ValueError("Please use `KTOConfig` instead TrainingArguments.")

        if args.model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError("You passed model_kwargs to the KTOTrainer. But your model is already instantiated.")
        else:
            model_init_kwargs = args.model_init_kwargs
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if torch_dtype is not None:
                # Convert to `torch.dtype` if an str is passed
                if isinstance(torch_dtype, str) and torch_dtype != "auto":
                    torch_dtype = getattr(torch, torch_dtype)
                if torch_dtype != "auto" and not isinstance(torch_dtype, torch.dtype):
                    raise ValueError(
                        f"Invalid `torch_dtype` passed to the KTOConfig. Expected a string with either `torch.dtype` or 'auto', but got {torch_dtype}."
                    )
                model_init_kwargs["torch_dtype"] = torch_dtype

        if args.ref_model_init_kwargs is None:
            ref_model_init_kwargs = {}
        elif not isinstance(ref_model, str):
            raise ValueError(
                "You passed ref_model_kwargs to the KTOTrainer. But your ref_model is already instantiated."
            )
        else:
            ref_model_init_kwargs = args.ref_model_init_kwargs
            torch_dtype = ref_model_init_kwargs.get("torch_dtype")
            if torch_dtype is not None:
                # Convert to `torch.dtype` if an str is passed
                if isinstance(torch_dtype, str) and torch_dtype != "auto":
                    torch_dtype = getattr(torch, torch_dtype)
                if torch_dtype != "auto" and not isinstance(torch_dtype, torch.dtype):
                    raise ValueError(
                        f"Invalid `torch_dtype` passed to the KTOConfig. Expected a string with either `torch.dtype` or 'auto', but got {torch_dtype}."
                    )
                ref_model_init_kwargs["torch_dtype"] = torch_dtype

        if isinstance(model, str):
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        if isinstance(ref_model, str):
            ref_model = AutoModelForCausalLM.from_pretrained(ref_model, **ref_model_init_kwargs)

        # Initialize this variable to False. This helps tracking the case when `peft_module_casting_to_bf16`
        # has been called in order to properly call autocast if needed.
        self._peft_has_been_casted_to_bf16 = False

        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it with `pip install peft` to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            # if model is a peft model and we have a peft_config, we merge and unload it first
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()

            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                _support_gc_kwargs = hasattr(
                    args, "gradient_checkpointing_kwargs"
                ) and "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )

                prepare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

                if _support_gc_kwargs:
                    prepare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

                model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)
            elif getattr(args, "gradient_checkpointing", False):
                # For backward compatibility with older versions of transformers
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:

                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)

                    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

            # get peft model with the given config
            model = get_peft_model(model, peft_config)
            if args.bf16 and getattr(model, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(model)
                # If args.bf16 we need to explicitly call `generate` with torch amp autocast context manager
                self._peft_has_been_casted_to_bf16 = True

        # For models that use gradient_checkpointing, we need to attach a hook that enables input
        # to explicitly have `requires_grad=True`, otherwise training will either silently
        # fail or completely fail.
        elif getattr(args, "gradient_checkpointing", False):
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if args.generate_during_eval and not (is_wandb_available() or is_comet_available()):
            raise ValueError(
                "`generate_during_eval=True` requires Weights and Biases or Comet to be installed."
                " Please install `wandb` or `comet-ml` to resolve."
            )

        if model is not None:
            self.is_encoder_decoder = model.config.is_encoder_decoder
        elif args.is_encoder_decoder is None:
            raise ValueError("When no model is provided, you need to pass the parameter is_encoder_decoder.")
        else:
            self.is_encoder_decoder = args.is_encoder_decoder

        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)
        self.model_adapter_name = model_adapter_name
        self.ref_adapter_name = ref_adapter_name

        if ref_model:
            self.ref_model = ref_model
        elif self.is_peft_model or args.precompute_ref_log_probs:
            # The `model` with adapters turned off will be used as the reference model
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(model)

        if processing_class is None:
            raise ValueError(
                "max_length or a processing_class must be specified when using the default DPODataCollatorWithPadding"
            )
        if args.max_length is None:
            warnings.warn(
                "When using DPODataCollatorWithPadding, you should set `max_length` in the KTOTrainer's init"
                " it will be set to `512` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_length = 512
        if args.max_length is not None:
            max_length = args.max_length

        if args.max_prompt_length is None:
            warnings.warn(
                "When using DPODataCollatorWithPadding, you should set `max_prompt_length` in the KTOTrainer's init"
                " it will be set to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_prompt_length = 128
        if args.max_prompt_length is not None:
            max_prompt_length = args.max_prompt_length

        max_completion_length = None
        if args.max_completion_length is None and self.is_encoder_decoder:
            warnings.warn(
                "When using DPODataCollatorWithPadding with an encoder decoder architecture, you should set `max_completion_length` in the KTOTrainer's init"
                " it will be set to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_completion_length = 128
        if args.max_completion_length is not None and self.is_encoder_decoder:
            max_completion_length = args.max_completion_length

        if data_collator is None:
            data_collator = DPODataCollatorWithPadding(
                pad_token_id=processing_class.pad_token_id,
                label_pad_token_id=args.label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                # warn users
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your KTOConfig"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        # Disable dropout in the model and reference model
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        self.loss_type = args.loss_type
        self.max_length = max_length
        self.generate_during_eval = args.generate_during_eval
        self.label_pad_token_id = args.label_pad_token_id
        self.padding_value = args.padding_value if args.padding_value is not None else processing_class.pad_token_id
        self.max_prompt_length = max_prompt_length
        self.truncation_mode = args.truncation_mode
        self.max_completion_length = max_completion_length
        self.processing_class = processing_class
        self.precompute_ref_log_probs = args.precompute_ref_log_probs

        # Not all losses require a KL calculation
        self.calculate_KL = True
        if self.loss_type in ["apo_zero_unpaired"]:
            self.calculate_KL = False

        # Since ref_logs are precomputed on the first call to get_train/eval_dataloader
        # keep track of first called to avoid computation of future calls
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False

        # metric
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # KTO parameter
        self.beta = args.beta
        self.desirable_weight = args.desirable_weight
        self.undesirable_weight = args.undesirable_weight
        self.aux_loss_enabled = getattr(model.config, "output_router_logits", False)
        self.aux_loss_coef = getattr(model.config, "router_aux_loss_coef", 0.0)
        if self.aux_loss_enabled and self.aux_loss_coef == 0.0:
            warnings.warn(
                "You set `output_router_logits` to `True` in the model config, but `router_aux_loss_coef` is set to "
                "`0.0`, meaning the auxiliary loss will not be used. Either set `router_aux_loss_coef` to a value "
                "greater than `0.0`, or set `output_router_logits` to `False` if you don't want to use the auxiliary "
                "loss.",
                UserWarning,
            )

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in KTO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys are "prompt_input_ids" and "completion_input_ids". As a result,
        # the trainer issues the warning: "Could not estimate the number of tokens of the input, floating-point
        # operations will not be computed." To suppress this warning, we set the "estimate_tokens" key in the model's
        # "warnings_issued" dictionary to True. This acts as a flag to indicate that the warning has already been
        # issued.
        model.warnings_issued["estimate_tokens"] = True

        # Compute that only on the main process for faster data processing.
        # see: https://github.com/huggingface/trl/pull/1255
        with PartialState().local_main_process_first():
            # Extract the prompt if needed
            train_dataset = train_dataset.map(
                maybe_extract_prompt, num_proc=args.dataset_num_proc, desc="Extracting prompt from train dataset"
            )
            # Unpair the dataset if needed
            train_dataset = maybe_unpair_preference_dataset(
                train_dataset, args.dataset_num_proc, desc="Unpairing train dataset"
            )
            # Apply the chat template if needed
            train_dataset = train_dataset.map(
                maybe_apply_chat_template,
                fn_kwargs={"tokenizer": processing_class},
                num_proc=args.dataset_num_proc,
                desc="Applying chat template to train dataset",
            )
            if eval_dataset is not None:
                eval_dataset = eval_dataset.map(
                    maybe_extract_prompt, num_proc=args.dataset_num_proc, desc="Extracting prompt from eval dataset"
                )
                eval_dataset = maybe_unpair_preference_dataset(
                    eval_dataset, args.dataset_num_proc, desc="Unpairing eval dataset"
                )
                eval_dataset = eval_dataset.map(
                    maybe_apply_chat_template,
                    fn_kwargs={"tokenizer": processing_class},
                    num_proc=args.dataset_num_proc,
                    desc="Applying chat template to eval dataset",
                )

            # Tokenize and prepare the training datasets
            train_dataset = train_dataset.map(
                _tokenize,
                batched=True,
                fn_kwargs={"tokenizer": self.processing_class},
                num_proc=args.dataset_num_proc,
                desc="Tokenizing train dataset",
            )

            fn_kwargs = {
                "prefix": "",
                "is_encoder_decoder": self.is_encoder_decoder,
                "tokenizer": self.processing_class,
                "max_length": self.max_length,
                "truncation_mode": self.truncation_mode,
                "label_pad_token_id": self.label_pad_token_id,
                "max_prompt_length": self.max_prompt_length,
                "max_completion_length": self.max_completion_length,
            }

            train_dataset = train_dataset.map(
                _process_tokens,
                fn_kwargs=fn_kwargs,
                num_proc=args.dataset_num_proc,
                desc="Processing tokenized train dataset",
            )

            # Tokenize and prepare the eval datasets
            if eval_dataset is not None:
                eval_dataset = eval_dataset.map(
                    _tokenize,
                    fn_kwargs={"tokenizer": self.processing_class},
                    batched=True,
                    num_proc=args.dataset_num_proc,
                    desc="Tokenizing eval dataset",
                )

                eval_dataset = eval_dataset.map(
                    _process_tokens,
                    fn_kwargs=fn_kwargs,
                    num_proc=args.dataset_num_proc,
                    desc="Processing tokenized eval dataset",
                )

            # Get KL datasets if needed
            if self.calculate_KL:
                if args.per_device_train_batch_size <= 1:
                    raise ValueError(
                        "Actual (not effective) batch size must be > 1. KTO will not work properly because the KL term will be equivalent to the implied reward."
                    )

                # create pairs for estimating the KL term by flipping the matched pairs in each batch of size total_batch_size
                # i.e., (x_1, y_1), ..., (x_n, y_n) --> (x_1, y_n), ..., (x_n, y_1) = (x'_1, y'_1), ..., (x'_n, y'_n)
                train_kl_dataset = train_dataset.map(
                    _get_kl_dataset,
                    batched=True,
                    batch_size=args.per_device_train_batch_size,
                    num_proc=args.dataset_num_proc,
                    desc="Extracting KL train dataset",
                )

                fn_kwargs["prefix"] = "KL_"
                train_kl_dataset = train_kl_dataset.map(
                    _process_tokens,
                    fn_kwargs=fn_kwargs,
                    num_proc=args.dataset_num_proc,
                    remove_columns=[c for c in train_kl_dataset.column_names if c in train_dataset.column_names],
                    desc="Processing tokenized train KL dataset",
                )

                # merge the datasets
                train_dataset = concatenate_datasets([train_dataset, train_kl_dataset], axis=1)

                if eval_dataset is not None:
                    # Get KL dataset
                    eval_kl_dataset = eval_dataset.map(
                        _get_kl_dataset,
                        batched=True,
                        batch_size=args.per_device_train_batch_size,
                        num_proc=args.dataset_num_proc,
                        desc="Extracting eval KL dataset",
                    )

                    eval_kl_dataset = eval_kl_dataset.map(
                        _process_tokens,
                        fn_kwargs=fn_kwargs,
                        num_proc=args.dataset_num_proc,
                        remove_columns=[c for c in eval_kl_dataset.column_names if c in eval_dataset.column_names],
                        desc="Processing tokenized eval KL dataset",
                    )

                    # merge the datasets
                    eval_dataset = concatenate_datasets([eval_dataset, eval_kl_dataset], axis=1)

            # calculate dataset desirability balance
            num_desirable = max(sum(train_dataset["label"]), 1)
            num_undesirable = max(len(train_dataset["label"]) - num_desirable, 1)  # "label" is binary

            if num_desirable != num_undesirable:
                # The lower and upper bounds come from Eq. (8) of https://huggingface.co/papers/2402.01306
                des_weight_lower_bound = round((num_undesirable * self.undesirable_weight / num_desirable) * 1, 2)
                des_weight_upper_bound = round((num_undesirable * self.undesirable_weight / num_desirable) * 1.33, 2)
                und_weight_lower_bound = round((num_desirable * self.desirable_weight / num_undesirable) / 1.33, 2)
                und_weight_upper_bound = round((num_desirable * self.desirable_weight / num_undesirable) / 1, 2)

                des_weight_in_range = des_weight_lower_bound <= self.desirable_weight <= des_weight_upper_bound
                und_weight_in_range = und_weight_lower_bound <= self.undesirable_weight <= und_weight_upper_bound

                if not (des_weight_in_range or und_weight_in_range):
                    warnings.warn(
                        "You have different amounts of desirable/positive and undesirable/negative examples but the "
                        "weights on the desirable and undesirable losses don't seem to be in an ideal range. Based "
                        f"on your data, we recommend EITHER "
                        f"desirable_weight in [{des_weight_lower_bound}, {des_weight_upper_bound}] or "
                        f"undesirable_weight in [{und_weight_lower_bound}, {und_weight_upper_bound}] (but NOT BOTH). "
                        "See the documentation on how to optimally set these weights.",
                        UserWarning,
                    )

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

        # Deepspeed Zero-3 does not support precompute_ref_log_probs
        if self.is_deepspeed_enabled:
            if self.accelerator.state.deepspeed_plugin.zero_stage == 3 and self.precompute_ref_log_probs:
                raise ValueError(
                    "You cannot use `precompute_ref_log_probs=True` with Deepspeed ZeRO-3. Please set `precompute_ref_log_probs=False`."
                )

        if self.ref_model is None:
            if not (self.is_peft_model or self.precompute_ref_log_probs):
                raise ValueError(
                    "No reference model and model is not a Peft model. Try setting `precompute_ref_log_probs=True`"
                )
        else:
            if self.is_deepspeed_enabled:
                self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
