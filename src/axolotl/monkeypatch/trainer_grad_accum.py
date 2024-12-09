"""
fix for FSDP gradient accumulation
see https://github.com/huggingface/transformers/pull/35128
"""
import inspect
import logging

from transformers import LlamaForCausalLM, Trainer

from axolotl.monkeypatch.unsloth_ import detab_code

LOG = logging.getLogger("axolotl.monkeypatch.trainer_grad_accum")

ORIGINAL_CONTEXT_CODE = """
    with self.compute_loss_context_manager():
        if self.model_accepts_loss_kwargs:
            loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
"""

PATCHED_CONTEXT_CODE = """
    with self.compute_loss_context_manager():
        if self.model_accepts_loss_kwargs:
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
        else:
            loss = self.compute_loss(model, inputs)
"""

ORIGINAL_LLAMA_FCLM_CODE = """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs[0]
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

    loss = None
    if labels is not None:
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
"""

PATCHED_LLAMA_FCLM_CODE = """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # remove num_items_in_batch otherwise self.model attempts to pass it to flash_attention
    num_items_in_batch = kwargs.pop("num_items_in_batch", None)

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        **kwargs,
    )
    hidden_states = outputs[0]
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

    loss = None
    if labels is not None:
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, num_items_in_batch=num_items_in_batch, **kwargs)
"""


def get_training_step_code() -> str:
    training_step = inspect.getsource(
        Trainer.training_step  # pylint: disable=protected-access
    )
    return training_step


def check_training_step_is_patchable() -> bool:
    training_step = get_training_step_code()
    training_step, _ = detab_code(training_step)
    return ORIGINAL_CONTEXT_CODE in training_step


def patch_training_step_for_ga():
    """
    monkeypatch for fixing the training loop for gradient accumulation
    """

    try:
        training_step = get_training_step_code()
    except OSError:
        return
    Trainer._original_training_step = training_step  # pylint: disable=protected-access
    training_step, _ = detab_code(training_step)
    if ORIGINAL_CONTEXT_CODE not in training_step:
        return
    # assert (
    #     ORIGINAL_CONTEXT_CODE in training_step
    # ), "Original training_step code not found"

    training_step = training_step.replace(ORIGINAL_CONTEXT_CODE, PATCHED_CONTEXT_CODE)
    training_step = training_step.replace(
        "def training_step(",
        "def _fixed_training_step(",
        1,
    )

    # load imports necessary
    import transformers.trainer

    items_to_import = []
    for item in dir(transformers.trainer):
        if item in training_step:
            items_to_import.append(item)

    exec(  # pylint: disable=exec-used  # nosec B102
        "from transformers.trainer import ("
        + ", ".join(x for x in items_to_import)
        + ")",
        globals(),
    )
    exec(training_step, globals())  # pylint: disable=exec-used  # nosec B102
    LOG.info("patching training_step")
    Trainer.training_step = (  # pylint: disable=protected-access
        _fixed_training_step  # pylint: disable=undefined-variable  # noqa: F821
    )


def get_model_forward_code() -> str:
    forward = inspect.getsource(
        LlamaForCausalLM.forward  # pylint: disable=protected-access
    )
    return forward


def check_forward_is_patchable() -> bool:
    forward = get_model_forward_code()
    forward, _ = detab_code(forward)
    return ORIGINAL_LLAMA_FCLM_CODE in forward


def patch_forward_for_ga():
    """
    monkeypatch for fixing the training loop for gradient accumulation
    """

    try:
        forward = get_model_forward_code()
    except OSError:
        return
    LlamaForCausalLM._original_forward = forward  # pylint: disable=protected-access
    forward, _ = detab_code(forward)
    if ORIGINAL_LLAMA_FCLM_CODE not in forward:
        return
    # assert ORIGINAL_LLAMA_FCLM_CODE in forward, "Original forward code not found"

    forward = forward.replace(ORIGINAL_LLAMA_FCLM_CODE, PATCHED_LLAMA_FCLM_CODE)
    forward = forward.replace(
        "def forward(",
        "def _fixed_forward(",
        1,
    )

    # load imports necessary
    import transformers.models.llama.modeling_llama

    items_to_import = []
    for item in dir(transformers.models.llama.modeling_llama):
        if item in forward:
            items_to_import.append(item)

    exec(  # pylint: disable=exec-used  # nosec B102
        "from transformers.models.llama.modeling_llama import ("
        + ", ".join(x for x in items_to_import)
        + ")",
        globals(),
    )
    exec(forward, globals())  # pylint: disable=exec-used  # nosec B102
    LOG.info("patching forward")
    LlamaForCausalLM.forward = (  # pylint: disable=protected-access
        _fixed_forward  # pylint: disable=undefined-variable  # noqa: F821
    )


ORIGINAL_TRAINER_CODE = """
                context = (
                    functools.partial(self.accelerator.no_sync, model=model)
                    if i != len(batch_samples) - 1
                    else contextlib.nullcontext
                )
                with context():
                    tr_loss_step = self.training_step(model, inputs, num_items_in_batch)
"""

PATCHED_TRAINER_CODE = """
                disable_deepspeed_no_sync = (
                        self.accelerator.distributed_type == DistributedType.DEEPSPEED
                        # and self.accelerator.deepspeed_engine_wrapped.engine.zero_optimization_partition_gradients()
                )
                context = (
                    functools.partial(self.accelerator.no_sync, model=model)
                    if i != len(batch_samples) - 1 and not disable_deepspeed_no_sync
                    else contextlib.nullcontext
                )
                with context():
                    tr_loss_step = self.training_step(model, inputs, num_items_in_batch)
"""


def get_training_loop_code() -> str:
    training_loop = inspect.getsource(
        Trainer._inner_training_loop  # pylint: disable=protected-access
    )
    return training_loop


def check_training_loop_is_patchable() -> bool:
    training_loop = get_training_loop_code()
    training_loop, _ = detab_code(training_loop)
    return ORIGINAL_TRAINER_CODE in training_loop


def patch_training_loop_for_deepspeed_0_16_x():
    """
    monkeypatch for fixing the training loop for deepspeed GA

    see https://github.com/huggingface/transformers/pull/35157
    """

    try:
        training_loop = get_training_loop_code()
    except OSError:
        return
    Trainer._original_inner_training_loop = (  # pylint: disable=protected-access
        training_loop
    )
    training_loop, _ = detab_code(training_loop)
    if ORIGINAL_TRAINER_CODE not in training_loop:
        return

    training_loop = training_loop.replace(ORIGINAL_TRAINER_CODE, PATCHED_TRAINER_CODE)
    training_loop = training_loop.replace(
        "def _inner_training_loop(",
        "def _fixed_inner_training_loop(",
        1,
    )

    # load imports necessary
    import transformers.trainer

    items_to_import = []
    for item in dir(transformers.trainer):
        if item in training_loop:
            items_to_import.append(item)

    exec(  # pylint: disable=exec-used  # nosec B102
        "from transformers.trainer import ("
        + ", ".join(x for x in items_to_import)
        + ")",
        globals(),
    )
    exec(training_loop, globals())  # pylint: disable=exec-used  # nosec B102
    LOG.info("patching _inner_training_loop for fsdp optimizer save")
    Trainer._inner_training_loop = (  # pylint: disable=protected-access
        _fixed_inner_training_loop  # pylint: disable=undefined-variable  # noqa: F821
    )
