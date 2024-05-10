"""Module for models and model loading"""

# pylint: disable=too-many-lines

import logging
import math
import os
import types
from typing import Any, Dict, Optional, Tuple, Union  # noqa: F401

import addict
import bitsandbytes as bnb
import torch
import transformers
import transformers.modeling_utils
from accelerate import init_empty_weights
from bitsandbytes.nn import Params4bit
from peft import (
    LoftQConfig,
    PeftConfig,
    PeftModel,
    PeftModelForCausalLM,
    prepare_model_for_kbit_training,
)
from peft.tuners.lora import QuantLinear
from torch import nn
from transformers import (  # noqa: F401
    AddedToken,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GPTQConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

from axolotl.models.mamba import fix_mamba_attn_for_loss
from axolotl.monkeypatch.multipack import (
    SUPPORTED_MULTIPACK_MODEL_TYPES,
    patch_for_multipack,
)
from axolotl.prompt_tokenizers import LLAMA_DEFAULT_EOS_TOKEN
from axolotl.utils.bench import log_gpu_memory_usage
from axolotl.utils.chat_templates import chat_templates
from axolotl.utils.dict import DictDefault
from axolotl.utils.distributed import zero_only
from axolotl.utils.gradient_checkpointing import hf_grad_checkpoint_unsloth_wrapper
from axolotl.utils.lora_embeddings import get_linear_embedding_layers
from axolotl.utils.model_shard_quant import load_sharded_model, load_sharded_model_quant

LOG = logging.getLogger("axolotl")


# copied from accelerator.FullyShardedDataParallelPlugin
def get_module_class_from_name(module, name):
    """
    Gets a class from a module by its name.

    Args:
        module (`torch.nn.Module`): The module to get the class from.
        name (`str`): The name of the class.
    """
    modules_children = list(module.children())
    if module.__class__.__name__ == name:
        return module.__class__

    if len(modules_children) == 0:
        return None

    for child_module in modules_children:
        module_class = get_module_class_from_name(child_module, name)
        if module_class is not None:
            return module_class

    return None


def check_model_config(cfg: DictDefault, model_config: Union[AutoConfig, DictDefault]):
    quant_config_exists = (
        hasattr(model_config, "quantization_config")
        and model_config.quantization_config
    )
    quant_config_method_is_gptq = (
        quant_config_exists
        and "quant_method" in model_config.quantization_config
        and model_config.quantization_config["quant_method"] == "gptq"
    )

    if cfg.gptq and not quant_config_method_is_gptq:
        raise ValueError(
            "model_config.quantization_config is not set or quant_method is not set to gptq. "
            "Please make sure to point to a GPTQ model."
        )

    if not cfg.gptq and quant_config_exists:
        raise ValueError(
            "model_config.quantization_config is set but `gptq` flag is not. "
            "Please use the `gptq` flag to train quantized model or point to a non-quantized model."
        )

    lora_modules_to_save = get_linear_embedding_layers(model_config.model_type)
    if (
        cfg.adapter
        and cfg.tokens
        and (
            not cfg.lora_modules_to_save
            or not all(x in cfg.lora_modules_to_save for x in lora_modules_to_save)
        )
    ):
        lora_modules_to_save = ", ".join(map(lambda x: f"`{x}`", lora_modules_to_save))
        raise ValueError(
            f"`lora_modules_to_save` not properly set when adding new tokens. Please include [{lora_modules_to_save}] in `lora_modules_to_save`."
        )


def load_model_config(cfg):
    model_config_name = cfg.base_model_config or cfg.base_model
    if not model_config_name and cfg.tokenizer_config:
        model_config_name = cfg.tokenizer_config
    trust_remote_code = cfg.trust_remote_code is True
    config_kwargs = {}
    if cfg.revision_of_model:
        config_kwargs["revision"] = cfg.revision_of_model

    try:
        model_config = AutoConfig.from_pretrained(
            model_config_name,
            trust_remote_code=trust_remote_code,
            **config_kwargs,
        )
    except ValueError as err:
        if "mamba" in model_config_name:
            return addict.Dict(
                {
                    "model_type": "mamba",
                }
            )
        raise err

    if cfg.overrides_of_model_config:
        for key, val in cfg.overrides_of_model_config.items():
            setattr(model_config, key, val)

    check_model_config(cfg, model_config)

    return model_config


def load_tokenizer(cfg):
    model_config = load_model_config(cfg)
    tokenizer_kwargs = {}
    use_fast = True  # this is the default

    if cfg.tokenizer_use_fast is not None:
        use_fast = cfg.tokenizer_use_fast
    if cfg.tokenizer_legacy is not None:
        # True is the default w/ https://github.com/huggingface/transformers/pull/25224
        tokenizer_kwargs["legacy"] = cfg.tokenizer_legacy

    tokenizer_cls = AutoTokenizer
    if cfg.tokenizer_type:
        tokenizer_cls = getattr(transformers, cfg.tokenizer_type)

    tokenizer = tokenizer_cls.from_pretrained(
        cfg.tokenizer_config,
        trust_remote_code=cfg.trust_remote_code or False,
        use_fast=use_fast,
        **tokenizer_kwargs,
    )

    if (
        tokenizer.__class__.__name__
        in [
            "LlamaTokenizer",
            "LlamaTokenizerFast",
            "CodeLlamaTokenizer",
            "CodeLlamaTokenizerFast",
        ]
        and hasattr(tokenizer, "pad_token")
        and not tokenizer.pad_token
    ):
        # set a pad_token, but use eos_token so we don't add a new token
        tokenizer.pad_token = LLAMA_DEFAULT_EOS_TOKEN

    if tokenizer.__class__.__name__ == "GPTNeoXTokenizerFast":
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Mistral's official FA implementation requires left padding
    if cfg.is_mistral_derived_model and cfg.flash_attention and not cfg.sample_packing:
        tokenizer.padding_side = "left"

    # Qwen base only has single token, so we need to set the special tokens
    if cfg.is_qwen_derived_model:
        token_ids = ["bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"]
        for attr_name in token_ids:
            if getattr(tokenizer, attr_name) is None:
                setattr(tokenizer, attr_name, tokenizer.eod_id)

        token_names = ["bos_token", "eos_token", "pad_token", "unk_token"]
        for attr_name in token_names:
            if getattr(tokenizer, attr_name) is None:
                setattr(tokenizer, attr_name, "<|endoftext|>")

    additional_special_tokens = None
    if cfg.special_tokens:
        special_tokens = cfg.special_tokens.to_dict()
        additional_special_tokens = special_tokens.pop(
            "additional_special_tokens", None
        )
        lora_modules_to_save = get_linear_embedding_layers(model_config.model_type)
        for k, val in special_tokens.items():
            # check if new special token is not already in tokenizer and
            # is adapter training to make sure lora_modules_to_save is set
            # pylint: disable=too-many-boolean-expressions
            if (
                (getattr(tokenizer, k) is None or getattr(tokenizer, k) != val)
                and (len(tokenizer.encode(val, add_special_tokens=False)) > 2)
                and cfg.adapter
                and (
                    not cfg.lora_modules_to_save
                    or not all(
                        x in cfg.lora_modules_to_save for x in lora_modules_to_save
                    )
                )
            ):
                lora_modules_to_save = ", ".join(
                    [f"`{x}`" for x in lora_modules_to_save]
                )
                raise ValueError(
                    f"Please set lora_modules_to_save to [{lora_modules_to_save}] when using an adapter and changing the special tokens."
                )

            tokenizer.add_special_tokens(
                {k: AddedToken(val, rstrip=False, lstrip=False, normalized=False)}
            )

        # If we add bos_token and eos_token, we need to update the post processor to
        # handle them correctly.
        # https://github.com/huggingface/transformers/pull/24132
        bos_or_eos_in_special_tokens = (
            "bos_token" in cfg.special_tokens and "eos_token" in cfg.special_tokens
        )
        if (
            tokenizer.__class__.__name__
            in (
                "LlamaTokenizerFast",
                "CodeLlamaTokenizerFast",
            )
            and bos_or_eos_in_special_tokens
        ):
            tokenizer.update_post_processor()

    if cfg.tokens:
        tokenizer.add_tokens(
            [
                AddedToken(token, rstrip=False, lstrip=False, normalized=False)
                for token in cfg.tokens
            ]
        )

    # Additional special tokens are a List, and need to be treated differently than regular special
    # tokens. We add them after we have called `add_tokens` in case these additional special tokens
    # are new tokens.
    #
    # Usage:
    #
    # ```py
    # special_tokens:
    #   additional_special_tokens: ["<|im_start|>", "<|im_end|>"]
    # ```
    if additional_special_tokens is not None:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": additional_special_tokens}
        )

    with zero_only():
        LOG.debug(f"EOS: {tokenizer.eos_token_id} / {tokenizer.eos_token}")
        LOG.debug(f"BOS: {tokenizer.bos_token_id} / {tokenizer.bos_token}")
        LOG.debug(f"PAD: {tokenizer.pad_token_id} / {tokenizer.pad_token}")
        LOG.debug(f"UNK: {tokenizer.unk_token_id} / {tokenizer.unk_token}")

    if cfg.chat_template:
        chat_template_string = chat_templates(cfg.chat_template)
        if cfg.default_system_message and cfg.chat_template == "chatml":
            chat_template_string = chat_template_string.replace(
                "You are a helpful assistant.", cfg.default_system_message
            )

        tokenizer.chat_template = chat_template_string
    else:
        LOG.info(
            "No Chat template selected. Consider adding a chat template for easier inference."
        )
    return tokenizer


def load_model(
    cfg: DictDefault,
    tokenizer: PreTrainedTokenizerBase,
    inference: bool = False,
    reference_model: bool = False,
) -> Tuple[PreTrainedModel, Optional[PeftConfig]]:
    """
    Load a model for a given configuration and tokenizer.
    """
    base_model = cfg.base_model
    model_type = cfg.type_of_model
    model_config = load_model_config(cfg)

    # TODO refactor as a kwarg
    load_in_8bit = cfg.load_in_8bit

    if cfg.gradient_checkpointing == "unsloth":
        transformers.modeling_utils.checkpoint = hf_grad_checkpoint_unsloth_wrapper

    if hasattr(model_config, "model_type") and model_config.model_type == "btlm":
        if cfg.flash_attention:
            from axolotl.monkeypatch.btlm_attn_hijack_flash import (
                replace_btlm_attn_with_flash_attn,
            )

            replace_btlm_attn_with_flash_attn(cfg.base_model)

    if (
        hasattr(model_config, "model_type")
        and model_config.model_type == "stablelm_epoch"
    ):
        if cfg.flash_attention and cfg.sample_packing:
            from axolotl.monkeypatch.stablelm_attn_hijack_flash import (
                replace_stablelm_attn_with_flash_attn,
            )

            replace_stablelm_attn_with_flash_attn(cfg.base_model)

    if cfg.sample_packing and cfg.s2_attention:
        raise ValueError(
            "Received `sample_packing=true` and `s2_attention=true`; however, \
        shifted-sparse attention does not currently support sample packing."
        )

    if (
        cfg.model_config_type in SUPPORTED_MULTIPACK_MODEL_TYPES
        and cfg.flash_attention
        and cfg.sample_packing
    ):
        patch_for_multipack(cfg.model_config_type, model_name=cfg.base_model)
    elif cfg.is_llama_derived_model:
        # Modify all llama derived models in one block

        if cfg.flash_attention:
            from axolotl.monkeypatch.llama_attn_hijack_flash import (
                replace_llama_attn_with_flash_attn,
            )

            if cfg.sample_packing:
                if cfg.device not in ["mps", "cpu"] and not inference:
                    LOG.info("patching with flash attention for sample packing")
                    replace_llama_attn_with_flash_attn(
                        packed=True,
                        cross_entropy=cfg.flash_attn_cross_entropy,
                        rms_norm=cfg.flash_attn_rms_norm,
                    )
            elif cfg.s2_attention:
                LOG.info("patching w/ flash-enabled, shifted-sparse attention")
                replace_llama_attn_with_flash_attn(
                    packed=False,
                    cross_entropy=cfg.flash_attn_cross_entropy,
                    rms_norm=cfg.flash_attn_rms_norm,
                    use_shifted_sparse_attn=True,
                )
        elif cfg.xformers_attention:
            from axolotl.monkeypatch.llama_attn_hijack_xformers import (
                hijack_llama_attention,
            )

            LOG.info("patching with xformers attention")
            hijack_llama_attention()
        elif cfg.sample_packing:
            from axolotl.monkeypatch.llama_patch_multipack import (
                hijack_llama_prepare_4d_mask,
            )

            LOG.info("patching llama _prepare_4d_causal_attention_mask*")
            hijack_llama_prepare_4d_mask()
        elif cfg.s2_attention:
            raise NotImplementedError(
                "Shifted-sparse attention not currently implemented without flash attention."
            )

        if cfg.unsloth_cross_entropy_loss:
            from axolotl.monkeypatch.unsloth_ import integrate_cross_entropy_loss_patch

            integrate_cross_entropy_loss_patch()

    # Modify mistral derived models
    if (
        cfg.model_config_type == "mistral"
        and cfg.flash_attention
        and cfg.sample_packing
    ):
        from axolotl.monkeypatch.mistral_attn_hijack_flash import (
            replace_mistral_attn_with_flash_attn,
        )

        LOG.info("patching mistral with flash attention")
        replace_mistral_attn_with_flash_attn(packed=cfg.sample_packing)

    if cfg.is_llama_derived_model and cfg.sample_packing and not inference:
        from axolotl.monkeypatch.llama_expand_mask import hijack_expand_mask

        LOG.info("patching _expand_mask")
        hijack_expand_mask()

    model_kwargs: Dict[str, Any] = {}

    if cfg.model_kwargs:
        for key, val in cfg.model_kwargs.items():
            model_kwargs[key] = val

    max_memory = cfg.max_memory
    device_map = cfg.device_map

    if cfg.gpu_memory_limit:
        gpu_memory_limit = (
            str(cfg.gpu_memory_limit) + "GiB"
            if isinstance(cfg.gpu_memory_limit, int)
            else cfg.gpu_memory_limit
        )

        max_memory = {}
        for i in range(torch.cuda.device_count()):
            max_memory[i] = gpu_memory_limit
        max_memory["cpu"] = "256GiB"  # something sufficiently large to fit anything

    if max_memory is not None:
        # Based on https://github.com/togethercomputer/OpenChatKit/blob/main/inference/bot.py
        from accelerate import infer_auto_device_map

        with init_empty_weights():
            model_canvas = AutoModelForCausalLM.from_config(
                model_config, trust_remote_code=cfg.trust_remote_code or False
            )
        model_canvas.tie_weights()
        device_map = infer_auto_device_map(
            model_canvas,
            max_memory=max_memory,
            dtype=cfg.torch_dtype,
        )
        # We can discard max_memory now as we have a device map set up for us
        max_memory = None

    model_kwargs["device_map"] = device_map
    model_kwargs["torch_dtype"] = cfg.torch_dtype

    if torch.backends.mps.is_available():
        model_kwargs["device_map"] = "mps:0"

    # TODO can we put the reference model on it's own gpu? I think we have to move logits around to calculate loss
    # if cfg.rl:
    #     if torch.cuda.device_count() > 1:
    #         if reference_model:
    #             model_kwargs["device_map"] = "cuda:" + str(
    #                 torch.cuda.current_device() + 1
    #             )
    #         else:
    #             model_kwargs["device_map"] = "cuda:" + str(torch.cuda.current_device())

    if is_deepspeed_zero3_enabled():
        del model_kwargs["device_map"]

    if cfg.revision_of_model:
        model_kwargs["revision"] = cfg.revision_of_model

    if cfg.gptq:
        if not hasattr(model_config, "quantization_config"):
            LOG.warning("model config does not contain quantization_config information")
        else:
            if cfg.gptq_disable_exllama is not None:
                model_config.quantization_config[
                    "disable_exllama"
                ] = cfg.gptq_disable_exllama
            model_kwargs["quantization_config"] = GPTQConfig(
                **model_config.quantization_config
            )
    if cfg.adapter == "qlora" and cfg.load_in_4bit:
        bnb_config = {
            "load_in_4bit": True,
            "llm_int8_threshold": 6.0,
            "llm_int8_has_fp16_weight": False,
            "bnb_4bit_compute_dtype": cfg.torch_dtype,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_quant_storage": torch.bfloat16,
        }
        if cfg.model_config_type in ["jamba", "qwen2_moe"] and not cfg.deepspeed:
            # for some reason, this causes the loss to be off by an order of magnitude
            # but deepspeed needs this still in bfloat16
            bnb_config["bnb_4bit_quant_storage"] = torch.float32

        if cfg.bnb_config_kwargs:
            bnb_config.update(cfg.bnb_config_kwargs)

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            **bnb_config,
        )
    elif cfg.adapter == "lora" and cfg.load_in_8bit:
        bnb_config = {
            "load_in_8bit": True,
        }
        # Exclude mamba blocks from int8 quantization for jamba
        if cfg.model_config_type == "jamba":
            bnb_config["llm_int8_skip_modules"] = ["mamba"]
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            **bnb_config,
        )

    if cfg.load_in_8bit and cfg.adapter is not None:
        model_kwargs["load_in_8bit"] = True
    if cfg.load_in_4bit and cfg.adapter is not None:
        model_kwargs["load_in_4bit"] = True

    # no longer needed per https://github.com/huggingface/transformers/pull/26610
    if "quantization_config" in model_kwargs or cfg.gptq:
        if "load_in_8bit" in model_kwargs:
            del model_kwargs["load_in_8bit"]
        if "load_in_4bit" in model_kwargs:
            del model_kwargs["load_in_4bit"]

    # sample packing uses custom FA2 patch
    if cfg.flash_attention:
        if not cfg.sample_packing:
            if cfg.s2_attention:
                pass
            # most other models support flash attention, we can define exceptions as they come up
            model_kwargs["attn_implementation"] = "flash_attention_2"
            model_config._attn_implementation = (  # pylint: disable=protected-access
                "flash_attention_2"
            )
        else:
            if model_config.model_type in SUPPORTED_MULTIPACK_MODEL_TYPES:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                model_config._attn_implementation = (  # pylint: disable=protected-access
                    "flash_attention_2"
                )
            else:
                model_kwargs["attn_implementation"] = "eager"
                model_config._attn_implementation = (  # pylint: disable=protected-access
                    "eager"
                )
    elif cfg.sdp_attention:
        model_kwargs["attn_implementation"] = "sdpa"
        model_config._attn_implementation = "sdpa"  # pylint: disable=protected-access
    elif cfg.eager_attention:
        model_kwargs["attn_implementation"] = "eager"
        model_config._attn_implementation = "eager"  # pylint: disable=protected-access

    if cfg.low_cpu_mem_usage:
        model_kwargs["low_cpu_mem_usage"] = True

    qlora_fsdp = cfg.fsdp and cfg.adapter == "qlora"

    try:
        skip_move_to_device = False
        if (
            cfg.fsdp and cfg.fsdp_config.fsdp_cpu_ram_efficient_loading
        ) and not qlora_fsdp:
            model = load_sharded_model(
                base_model,
                model_config,
                cfg,
                torch_dtype=cfg.torch_dtype,
            )
            skip_move_to_device = True
        elif (
            qlora_fsdp
            and cfg.fsdp_config.fsdp_cpu_ram_efficient_loading
            and cfg.model_config_type == "dbrx"
        ):
            quant_storage = cfg.torch_dtype
            model = load_sharded_model_quant(
                base_model,
                model_config,
                cfg,
                quant_storage=quant_storage,
            )
            skip_move_to_device = True
        elif (
            model_config.model_type == "llama"
            and not cfg.trust_remote_code
            and not cfg.gptq
        ):
            from transformers import LlamaForCausalLM

            model = LlamaForCausalLM.from_pretrained(
                base_model,
                config=model_config,
                **model_kwargs,
            )

            if cfg.flash_attention and not inference:
                from axolotl.monkeypatch.llama_attn_hijack_flash import (
                    is_xformers_swiglu_available,
                    replace_llama_mlp_with_swiglu,
                    replace_llama_qkv_with_fused,
                )

                if cfg.flash_attn_fuse_mlp and is_xformers_swiglu_available():
                    LOG.info("patching with SwiGLU")
                    replace_llama_mlp_with_swiglu(model)

                if cfg.flash_attn_fuse_qkv:
                    LOG.info("patching with fused QKV")
                    replace_llama_qkv_with_fused(model)
        elif model_type == "MambaLMHeadModel":
            # FIXME this is janky at best and hacked together to make it work
            MambaLMHeadModel = fix_mamba_attn_for_loss()  # pylint: disable=invalid-name

            model_kwargs["dtype"] = model_kwargs["torch_dtype"]
            model_kwargs["device"] = torch.cuda.current_device()
            del model_kwargs["torch_dtype"]
            del model_kwargs["device_map"]

            model = MambaLMHeadModel.from_pretrained(
                base_model,
                **model_kwargs,
            )
        elif model_type and not cfg.trust_remote_code:
            if cfg.gptq:
                model = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    config=model_config,
                    trust_remote_code=cfg.trust_remote_code or False,
                    **model_kwargs,
                )
            else:
                model = getattr(transformers, model_type).from_pretrained(
                    base_model,
                    config=model_config,
                    trust_remote_code=cfg.trust_remote_code or False,
                    **model_kwargs,
                )
        else:
            # Shouldn't be a problem most of the time. will obviously error if the model doesn't support this
            # when training starts
            if (
                hasattr(model_config, "max_seq_len")
                and model_config.max_seq_len
                and cfg.sequence_len > model_config.max_seq_len
            ):
                model_config.max_seq_len = cfg.sequence_len
                LOG.warning(f"increasing context length to {cfg.sequence_len}")
            elif (
                hasattr(model_config, "max_sequence_length")
                and model_config.max_sequence_length
                and cfg.sequence_len > model_config.max_sequence_length
            ):
                model_config.max_sequence_length = cfg.sequence_len
                LOG.warning(f"increasing context length to {cfg.sequence_len}")
            if cfg.gptq:
                model = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    config=model_config,
                    trust_remote_code=cfg.trust_remote_code or False,
                    **model_kwargs,
                )
            else:
                if qlora_fsdp and cfg.fsdp_config.fsdp_cpu_ram_efficient_loading:
                    skip_move_to_device = True
                    if "device_map" in model_kwargs:
                        del model_kwargs["device_map"]

                model = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    config=model_config,
                    trust_remote_code=cfg.trust_remote_code or False,
                    **model_kwargs,
                )
    except Exception as err:  # pylint: disable=broad-exception-caught
        LOG.exception(err)
        raise err

    if isinstance(model, (PeftModel, PeftModelForCausalLM)) and not qlora_fsdp:
        model = model.merge_and_unload()

    embeddings_len = (
        math.ceil(len(tokenizer) / 32) * 32
        if cfg.resize_token_embeddings_to_32x
        else len(tokenizer)
    )
    if (
        hasattr(model, "get_input_embeddings")
        and model.get_input_embeddings().num_embeddings < embeddings_len
    ):
        model.resize_token_embeddings(embeddings_len)
    else:
        model.tie_weights()

    if (
        hasattr(model, "config")
        and hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings
        and cfg.sequence_len > model.config.max_position_embeddings
    ):
        LOG.warning(
            f"increasing model.config.max_position_embeddings from {model.config.max_position_embeddings} to {cfg.sequence_len}"
        )
        model.config.max_position_embeddings = cfg.sequence_len

    if (
        hasattr(model, "config")
        and hasattr(model.config, "bos_token_id")
        and model.config.bos_token_id
        and model.config.bos_token_id != tokenizer.bos_token_id
    ):
        model.config.bos_token_id = tokenizer.bos_token_id

    if (
        hasattr(model, "config")
        and hasattr(model.config, "eos_token_id")
        and model.config.eos_token_id
        and model.config.eos_token_id != tokenizer.eos_token_id
    ):
        model.config.eos_token_id = tokenizer.eos_token_id

    if hasattr(model, "device") and model.device.type in ("cuda", "mps"):
        log_gpu_memory_usage(LOG, "after model load", model.device)

    # make sure these are fp32 per Ramesh et al. (2021)
    embedding_modules = get_linear_embedding_layers(cfg.model_config_type)
    if not cfg.fsdp:
        # FSDP doesn't like mixed Float and BFloat16
        for name, module in model.named_modules():
            if "norm" in name or name.endswith(".gate"):
                module.to(torch.float32)
            if model_config.model_type == "btlm":
                # don't upcast lm_head for btlm
                continue
            if any(m in name for m in embedding_modules):
                if hasattr(module, "weight"):
                    module.to(torch.float32)

    needs_fa2_dtype = cfg.adapter or cfg.fsdp
    skip_prepare_model_for_kbit_training = False

    if is_deepspeed_zero3_enabled():
        from deepspeed.utils import (  # pylint: disable=no-name-in-module
            set_z3_leaf_modules,
        )

        if cfg.model_config_type == "mixtral":
            moe_block = get_module_class_from_name(model, "MixtralSparseMoeBlock")
            set_z3_leaf_modules(model, [moe_block])
        elif cfg.model_config_type == "dbrx":
            moe_block = get_module_class_from_name(model, "DbrxFFN")
            set_z3_leaf_modules(model, [moe_block])

    if cfg.model_config_type == "qwen" and cfg.adapter == "lora":
        # Qwen doesn't play nicely with LoRA if this is enabled
        skip_prepare_model_for_kbit_training = True

    loftq_bits = cfg.peft and cfg.peft.loftq_config and cfg.peft.loftq_config.loftq_bits
    if cfg.adapter == "lora" and loftq_bits:
        skip_prepare_model_for_kbit_training = True

    if qlora_fsdp or (cfg.fsdp and cfg.fsdp_config.fsdp_cpu_ram_efficient_loading):
        # make sure everything is in the same dtype
        skip_prepare_model_for_kbit_training = True

    if cfg.adapter in ["lora", "qlora"]:
        if cfg.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=cfg.gradient_checkpointing_kwargs
            )
        if (
            cfg.load_in_8bit or cfg.load_in_4bit
        ) and not skip_prepare_model_for_kbit_training:
            LOG.info("converting PEFT model w/ prepare_model_for_kbit_training")
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=cfg.gradient_checkpointing
            )
        needs_fa2_dtype = True

    # LlamaRMSNorm layers are in fp32 after kbit_training or full finetune, so we need to
    # convert them back to fp16/bf16 for flash-attn compatibility.
    if (needs_fa2_dtype or cfg.flash_attention) and not qlora_fsdp:
        LOG.info("converting modules to %s for flash attention", cfg.torch_dtype)
        for name, module in model.named_modules():
            if "norm" in name:
                module.to(cfg.torch_dtype)
            if any(m in name for m in embedding_modules):
                if hasattr(module, "weight"):
                    module.to(cfg.torch_dtype)

    lora_config = None
    if not reference_model or cfg.lora_model_dir:
        # if we're not loading the reference model, then we're loading the model for training
        # then the dpo trainer doesn't want the peft model loaded over it, it just wants the lora/peft config
        if cfg.adapter and cfg.rl in ["dpo", "ipo", "kto_pair"] and not cfg.merge_lora:
            _, lora_config = load_lora(model, cfg, inference=False, config_only=True)
        else:
            model, lora_config = load_adapter(model, cfg, cfg.adapter)

    if (
        cfg.ddp
        and not load_in_8bit
        and not (cfg.rl and cfg.load_in_4bit)
        and not skip_move_to_device
    ):
        # TODO revaldate this conditional
        model.to(f"cuda:{cfg.local_rank}")

    if torch.cuda.device_count() > 1 and int(os.getenv("WORLD_SIZE", "1")) == 1:
        setattr(model, "is_parallelizable", True)
        setattr(model, "model_parallel", True)

    requires_grad = []
    for name, param in model.named_parameters(recurse=True):
        if param.requires_grad:
            requires_grad.append(f"{name}: {param.requires_grad}")
    if len(requires_grad) == 0:
        LOG.warning("there are no parameters that require gradient updates")
    if hasattr(model, "config"):
        model.config.use_cache = False

    if cfg.flash_optimum:
        from optimum.bettertransformer import BetterTransformer

        model = BetterTransformer.transform(model)

    if cfg.adapter is not None:
        log_gpu_memory_usage(LOG, "after adapters", model.device)

    if cfg.unsloth_lora_mlp:
        from axolotl.monkeypatch.unsloth_ import integrate_lora_mlp_patch

        integrate_lora_mlp_patch(model)
    if cfg.unsloth_lora_qkv:
        from axolotl.monkeypatch.unsloth_ import integrate_lora_qkv_patch

        integrate_lora_qkv_patch(model)
    if cfg.unsloth_lora_o:
        from axolotl.monkeypatch.unsloth_ import integrate_lora_o_patch

        integrate_lora_o_patch(model)

    # TODO resume_from_checkpoint handling
    return model, lora_config


def load_adapter(model, cfg, adapter, inference=False):
    # type: (PreTrainedModel, DictDefault, Optional[str], bool) -> Tuple[PreTrainedModel, Optional[PeftConfig]]

    if adapter is None:
        return model, None
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if adapter in ["lora", "qlora"]:
        return load_lora(model, cfg, inference=inference)
    if adapter == "llama-adapter":
        return load_llama_adapter(model, cfg)

    raise NotImplementedError(f"{adapter} peft adapter not available")


def load_llama_adapter(model, cfg):
    # type: (PreTrainedModel, DictDefault) -> Tuple[PreTrainedModel, Optional[PeftConfig]]
    from peft import AdaptionPromptConfig, get_peft_model

    peft_config = AdaptionPromptConfig(
        adapter_layers=cfg.peft_adapter.layers,  # layers (L)
        adapter_len=cfg.peft_adapter.len,  # prompt length (K)
        task_type="CAUSAL_LM",
    )

    if cfg.lora_model_dir:
        LOG.debug("Loading pretrained PEFT - llama_adapter")
        model = PeftModel.from_pretrained(
            model,
            cfg.lora_model_dir,
            torch_dtype=torch.float16,
        )
    else:
        model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    return model, peft_config


def find_all_linear_names(model):
    cls = (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt, torch.nn.Linear, QuantLinear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if (
            isinstance(module, cls)
            or "Linear" in module.__class__.__name__
            and module.__class__.__name__ not in ("LlamaLinearScalingRotaryEmbedding",)
        ):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    embedding_modules = get_linear_embedding_layers(model.config.model_type)
    output_embedding = embedding_modules[1]
    if output_embedding in lora_module_names:  # needed for 16-bit
        lora_module_names.remove(output_embedding)

    return list(lora_module_names)


def setup_quantized_meta_for_peft(model: nn.Module):
    """Replaces `quant_state.to` with a dummy function to prevent PEFT from moving `quant_state` to meta device"""

    def temp_to_method(self, *args, **kwargs):  # pylint: disable=unused-argument
        return self

    for param in model.parameters():
        if isinstance(param, Params4bit):
            param.quant_state._orig_to = (  # pylint: disable=protected-access
                param.quant_state.to
            )
            param.quant_state.to = types.MethodType(temp_to_method, param.quant_state)


def setup_quantized_peft_meta_for_training(model: nn.Module):
    """Replaces dummy `quant_state.to` method with the original function to allow training to continue"""
    for param in model.parameters():
        if isinstance(param, Params4bit) and hasattr(param.quant_state, "_orig_to"):
            param.quant_state.to = (
                param.quant_state._orig_to  # pylint: disable=protected-access
            )
            param.quant_state._orig_to = None  # pylint: disable=protected-access


def load_lora(model, cfg, inference=False, config_only=False):
    # type: (PreTrainedModel, DictDefault, bool, bool) -> Tuple[Optional[PreTrainedModel], Optional[PeftConfig]]

    from peft import LoraConfig, get_peft_model

    lora_target_modules = list(cfg.lora_target_modules or [])

    if cfg.lora_target_linear:
        linear_names = find_all_linear_names(model)
        LOG.info(f"found linear modules: {repr(linear_names)}")
        lora_target_modules = list(set(lora_target_modules + linear_names))

    lora_config_kwargs = {}
    loftq_bits = cfg.peft and cfg.peft.loftq_config and cfg.peft.loftq_config.loftq_bits
    if loftq_bits:
        lora_config_kwargs["loftq_config"] = LoftQConfig(loftq_bits=loftq_bits)
        lora_config_kwargs["init_lora_weights"] = "loftq"
    if cfg.peft_use_dora:
        lora_config_kwargs["use_dora"] = cfg.peft_use_dora
    if cfg.peft_use_rslora:
        lora_config_kwargs["use_rslora"] = cfg.peft_use_rslora
    if cfg.peft_layer_replication:
        lora_config_kwargs["layer_replication"] = cfg.peft_layer_replication

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=lora_target_modules,
        layers_to_transform=cfg.peft_layers_to_transform,
        lora_dropout=cfg.lora_dropout,
        fan_in_fan_out=cfg.lora_fan_in_fan_out,
        modules_to_save=cfg.lora_modules_to_save if cfg.lora_modules_to_save else None,
        bias="none",
        task_type="CAUSAL_LM",
        **lora_config_kwargs,
    )

    if config_only:
        return None, lora_config

    rank = int(os.environ.get("LOCAL_RANK", 0))

    if (
        cfg.fsdp
        and cfg.adapter
        and cfg.fsdp_config.fsdp_cpu_ram_efficient_loading
        and rank != 0
    ):
        setup_quantized_meta_for_peft(model)

    if cfg.lora_model_dir:
        LOG.debug("Loading pretrained PEFT - LoRA")
        model_kwargs: Any = {}
        if cfg.lora_on_cpu:
            model_kwargs["max_memory"] = {"cpu": "256GiB"}
            model_kwargs["device_map"] = {"": "cpu"}
        model = PeftModel.from_pretrained(
            model,
            cfg.lora_model_dir,
            is_trainable=(not inference),
            **model_kwargs,
        )
    else:
        model = get_peft_model(model, lora_config)

    if rank == 0:
        try:
            model.print_trainable_parameters()
        except AttributeError as exc:
            LOG.warning(
                "Exception caught during model.print_trainable_parameters(): %s", exc
            )
    elif (
        cfg.fsdp
        and cfg.adapter
        and cfg.fsdp_config.fsdp_cpu_ram_efficient_loading
        and rank != 0
    ):
        setup_quantized_peft_meta_for_training(model)

    return model, lora_config


def ensure_dtype(model, dtype=torch.bfloat16):
    for name, module in model.named_modules():
        try:
            if module.weight.dtype != dtype:
                print(f"Converting module {name}: {module.weight.dtype} -> {dtype}")
                module.to(dtype)
        except AttributeError:
            pass
