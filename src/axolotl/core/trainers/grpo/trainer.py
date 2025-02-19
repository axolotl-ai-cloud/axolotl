"""
Axolotl GRPO trainer
"""

from contextlib import contextmanager, nullcontext
from accelerate.utils import is_peft_model
from accelerate.utils.other import is_compiled_module
import torch
from transformers import PreTrainedModel
from trl import GRPOConfig, GRPOTrainer
from trl.models import unwrap_model_for_generation

from axolotl.core.trainers.base import SchedulerMixin
from transformers.utils import is_liger_kernel_available

if is_liger_kernel_available():
    from liger_kernel.chunked_loss.grpo_loss import LigerFusedLinearGRPOLoss


# mypy: ignore-errors
class AxolotlGRPOTrainer(SchedulerMixin, GRPOTrainer):
    """
    Extend the base GRPOTrainer for axolotl helpers
    """

    _tag_names = ["trl", "grpo", "axolotl"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Import Liger loss if enabled
        if self.args.use_liger_loss:
            if not is_liger_kernel_available():
                raise ValueError(
                    "You set `use_liger_loss=True` but the liger kernel is not available. "
                    "Please install liger-kernel first: `pip install liger-kernel`"
                )
            self.grpo_loss_fn = LigerFusedLinearGRPOLoss(
                beta=self.beta,
                compiled=is_compiled_module(self.model),
                use_ref_model=True,
                num_generations=self.args.num_generations,
            )
        # pylint: disable=access-member-before-definition
        # Enable gradient checkpointing if requested
        if kwargs["args"].gradient_checkpointing:
            # Ensure use_cache is disabled
            if hasattr(self.model, "config"):
                self.model.config.use_cache = False

            # Enable gradient checkpointing on the base model for PEFT
            if is_peft_model(self.model) and hasattr(self.model.base_model, "gradient_checkpointing_enable"):
                self.model.base_model.gradient_checkpointing_enable()
            # Enable gradient checkpointing for non-PEFT models
            elif hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            self.model = self._enable_gradient_checkpointing(self.model, kwargs["args"])
        # pylint: enable=access-member-before-definition

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # pylint: disable=unused-argument,redefined-builtin
        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        return model
        # pylint: enable=unused-argument,redefined-builtin

    def _move_model_to_vllm(self):
        with unwrap_model_for_generation(
            self.model,
            self.accelerator,
            gather_deepspeed3_params=self.args.ds3_gather_for_generation,
        ) as unwrapped_model:
            if is_compiled_module(unwrapped_model):
                unwrapped_model = unwrapped_model._orig_mod  # pylint: disable=protected-access
            if is_peft_model(unwrapped_model):
                unwrapped_model.merge_adapter()
                state_dict = unwrapped_model.state_dict()
                unwrapped_model.unmerge_adapter()
                # Remove base_model and base_layer prefixes
                state_dict = {
                    k.removeprefix("base_model.model.").removeprefix("base_model.model.").replace(".base_layer", ""): v
                    for k, v in state_dict.items()
                }
                # Remove values with adapter prefix (example: "_lora")
                state_dict = {k: v for k, v in state_dict.items() if unwrapped_model.prefix not in k}
                # When module to save, remove its prefix and discard the original module
                state_dict = {
                    k.replace("modules_to_save.default.", ""): v
                    for k, v in state_dict.items()
                    if "original_module" not in k
                }
            else:
                state_dict = unwrapped_model.state_dict()
        if self.accelerator.is_main_process:
            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
            llm_model.load_weights(state_dict.items())

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.args.use_liger_loss:
            if return_outputs:
                raise ValueError("The GRPOTrainer does not support returning outputs")

            device = self.accelerator.device
            prompts = [x["prompt"] for x in inputs]
            prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
            prompt_inputs = self.processing_class(
                prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
            )
            prompt_inputs = super()._prepare_inputs(prompt_inputs)

            if self.max_prompt_length is not None:
                prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length :]
                prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length :]

            # Generate completions using either vLLM or regular generation
            if self.args.use_vllm:
                # First, have main process load weights if needed
                if self.state.global_step != self._last_loaded_step:
                    with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                        state_dict = unwrapped_model.state_dict()
                    if self.accelerator.is_main_process:
                        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                        llm_model.load_weights(state_dict.items())
                    self._last_loaded_step = self.state.global_step

                # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
                all_prompts_text = gather_object(prompts_text)
                if self.accelerator.is_main_process:
                    outputs = self.llm.generate(all_prompts_text, sampling_params=self.sampling_params, use_tqdm=False)
                    completion_ids = [out.token_ids for completions in outputs for out in completions.outputs]
                else:
                    completion_ids = [None] * len(all_prompts_text) * self.num_generations

                # Broadcast the completions from the main process to all processes, ensuring each process receives its
                # corresponding slice.
                completion_ids = broadcast_object_list(completion_ids, from_process=0)
                process_slice = slice(
                    self.accelerator.process_index * len(prompts) * self.num_generations,
                    (self.accelerator.process_index + 1) * len(prompts) * self.num_generations,
                )
                completion_ids = completion_ids[process_slice]

                # Pad the completions, and concatenate them with the prompts
                completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
                completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
                prompt_inputs_repeated = torch.repeat_interleave(
                    prompt_inputs["input_ids"], self.num_generations, dim=0
                )
                prompt_completion_ids = torch.cat([prompt_inputs_repeated, completion_ids], dim=1)
            else:
                # Regular generation path
                with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                    prompt_completion_ids = unwrapped_model.generate(
                        **prompt_inputs, generation_config=self.generation_config
                    )

            prompt_length = prompt_inputs["input_ids"].size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]

            # Get the per-token log probabilities for the completions for the model and the reference model
            def get_per_token_logps(model, input_ids, num_logits_to_keep):
                # We add 1 to `num_logits_to_keep` because the last logits of the sequence is later excluded
                outputs = model(input_ids, num_logits_to_keep=num_logits_to_keep + 1)
                hidden_states = outputs.last_hidden_state[:, :-1]
                logits = outputs.logits  # (B, L, V)
                logits = logits[
                    :, :-1, :
                ]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

                # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
                per_token_logps = []
                for logits_row, input_ids_row in zip(logits, input_ids[:, -num_logits_to_keep:]):
                    log_probs = logits_row.log_softmax(dim=-1)
                    token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
                    per_token_logps.append(token_log_prob)
                return torch.stack(per_token_logps), hidden_states

            num_logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
            per_token_logps, hidden_states = get_per_token_logps(model, prompt_completion_ids, num_logits_to_keep)

            with torch.inference_mode():
                if self.ref_model is not None:
                    ref_per_token_logps, ref_hidden_states = get_per_token_logps(
                        self.ref_model, prompt_completion_ids, num_logits_to_keep
                    )
                else:
                    with self.accelerator.unwrap_model(model).disable_adapter():
                        ref_per_token_logps, ref_hidden_states = get_per_token_logps(model, prompt_completion_ids, num_logits_to_keep)

            # done in liger
            # Compute the KL divergence between the model and the reference model
            # per_token_kl = (
            #     torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            # )

            # Mask everything after the first EOS token
            is_eos = completion_ids == self.processing_class.eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

            # Decode the generated completions
            completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
            if is_conversational(inputs[0]):
                completions = [[{"role": "assistant", "content": completion}] for completion in completions]

            # Compute the rewards
            prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]

            rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
            for i, (reward_func, reward_processing_class) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes)
            ):
                if isinstance(reward_func, PreTrainedModel):
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                else:
                    # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                    reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                    for key in reward_kwargs:
                        for example in inputs:
                            # Repeat each value in the column for `num_generations` times
                            reward_kwargs[key].extend([example[key]] * self.num_generations)
                    output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

            # Sum the rewards from all reward functions
            rewards = rewards_per_func.sum(dim=1)

            # done in liger
            # # Compute grouped-wise rewards
            # mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
            # std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

            # done in liger
            # # Normalize the rewards to compute the advantages
            # mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            # std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            # advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
            
            # done in liger
            # x - x.detach() allows for preserving gradients from x
            # per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
            # per_token_loss = -(per_token_loss - self.beta * per_token_kl)
            # loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

            # Log the metrics
            completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
            self._metrics["completion_length"].append(completion_length)

            reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
            for i, reward_func in enumerate(self.reward_funcs):
                if isinstance(reward_func, PreTrainedModel):
                    reward_func_name = reward_func.config._name_or_path.split("/")[-1]
                else:
                    reward_func_name = reward_func.__name__
                self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

            self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())

            
            lm_head = model.get_output_embeddings()

            if self.ref_model is not None:
                ref_lm_head = self.ref_model.get_output_embeddings()
            else:
                with self.null_ref_context():
                    ref_lm_head = model.get_output_embeddings()
            ref_weight = ref_lm_head.weight
            ref_bias = ref_lm_head.bias if hasattr(ref_lm_head, "bias") else None

            loss, metrics = self.grpo_loss_fn(
                lm_head,
                hidden_states,  # this is the hidden states from the model
                completion_mask,
                rewards,
                bias=lm_head.bias if hasattr(lm_head, "bias") else None,
                ref_input=ref_hidden_states,  # this is the hidden states from the ref model
                ref_weight=ref_weight,
                ref_bias=ref_bias,
            )
        else:
            super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

    @contextmanager
    def null_ref_context(self):
        """Context manager for handling null reference model (that is, peft adapter manipulation)."""
        with (
            self.accelerator.unwrap_model(self.model).disable_adapter()
            if self.is_peft_model and not self.ref_adapter_name
            else nullcontext()
        ):
            if self.ref_adapter_name:
                self.model.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                self.model.set_adapter(self.model_adapter_name or "default")
