"""Pydantic models for datasets-related configuration"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from axolotl.utils.schemas.enums import ChatTemplate
from axolotl.utils.schemas.utils import handle_legacy_message_fields_logic


class UserDefinedPrompterType(BaseModel):
    """Structure for user defined prompt types"""

    system_prompt: str | None = Field(
        default=None,
        json_schema_extra={"description": "Custom user instruction prompt"},
    )
    system_format: str | None = Field(
        default=None,
        json_schema_extra={"description": "Use {system} as key to be replaced"},
    )
    field_system: str | None = None
    field_instruction: str | None = None
    field_input: str | None = None
    field_output: str | None = None

    format: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Customizable to be single line or multi-line. Use {instruction}/{input} as key to be replaced. 'format' can include {input}"
        },
    )
    no_input_format: str | None = Field(
        default=None,
        json_schema_extra={"description": "'no_input_format' cannot include {input}"},
    )


class SFTDataset(BaseModel):
    """SFT configuration subset"""

    path: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "HuggingFace dataset repo | s3:// | gs:// | path to local file or directory"
        },
    )
    split: str | None = Field(
        default=None,
        json_schema_extra={"description": "name of dataset split to load from"},
    )
    type: str | UserDefinedPrompterType | None = Field(
        default=None,
        json_schema_extra={
            "description": "The type of prompt to use for training. [alpaca, gpteacher, oasst, reflection]"
        },
    )
    input_transform: str | None = None
    shards: int | None = Field(
        default=None,
        json_schema_extra={
            "description": "split dataset into N pieces (use with shards_idx)"
        },
    )
    shards_idx: int | None = Field(
        default=None,
        json_schema_extra={"description": "the index of sharded dataset to use"},
    )
    preprocess_shards: int | None = Field(
        default=None,
        json_schema_extra={
            "description": "process dataset in N sequential chunks for memory efficiency (exclusive with `shards`)"
        },
    )
    conversation: str | None = None
    # Do not make this too strict or it will break the validator to choose different dataset class
    chat_template: ChatTemplate | str | None = Field(
        default=None,
        json_schema_extra={
            "description": "The name of the chat template to use for training, following values are supported: tokenizer_default: Uses the chat template that is available in the tokenizer_config.json. If the chat template is not available in the tokenizer, it will raise an error. This is the default. alpaca/inst/chatml/gemma/cohere/llama3/phi_3/deepseek_v2/jamba: These chat templates are available in the axolotl codebase at src/axolotl/utils/chat_templates.py. tokenizer_default_fallback_*: where * is the name of the chat template to fallback to if the tokenizer does not have a chat template else default to tokenizer. E.g. tokenizer_default_fallback_chatml. jinja: Uses a custom jinja template for the chat template. The custom jinja template should be provided in the chat_template_jinja field."
        },
    )
    chat_template_jinja: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Custom jinja chat template or path to jinja file. Used only if `chat_template: jinja` or empty."
        },
    )
    data_files: str | list[str] | None = Field(
        default=None, json_schema_extra={"description": "path to source data files"}
    )
    input_format: str | None = None
    name: str | None = Field(
        default=None,
        json_schema_extra={"description": "name of dataset configuration to load"},
    )
    ds_type: str | None = Field(
        default=None,
        json_schema_extra={"description": "defines the datatype when path is a file"},
    )
    field: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "For `completion` datasets only, uses the provided field instead of `text` column"
        },
    )
    field_human: str | None = None
    field_model: str | None = None
    field_messages: str | None = Field(
        default=None,
        json_schema_extra={
            "description": 'Key containing the messages (default: "messages")'
        },
    )
    field_tools: str | None = Field(
        default=None,
        json_schema_extra={
            "description": 'Key containing the tools (default: "tools"). Must be a list[dict] and follow [JSON schema](https://json-schema.org/learn/getting-started-step-by-step).'
        },
    )
    field_thinking: str | None = Field(
        default=None,
        json_schema_extra={
            "description": 'Key containing the reasoning trace (default: "reasoning_content").'
        },
    )
    template_thinking_key: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "The key the chat template expects that indicates the reasoning trace."
        },
    )
    # deprecated, use message_property_mappings
    message_field_role: str | None = None
    # deprecated, use message_property_mappings
    message_field_content: str | None = None
    message_property_mappings: dict[str, str] | None = Field(
        default=None,
        json_schema_extra={
            "description": "Mapping of properties from the input dataset to the chat template. (default: message_property_mappings={'role':'role', 'content':'content'}) If a property exists in the template but not in this mapping, the system will attempt to load it directly from the message using the property name as the key. Example: In the mapping below, 'from' is loaded from input dataset and used as 'role', while 'value' is loaded and used as 'content' in the chat template."
        },
    )
    message_field_training: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "The key in the message turn that indicates via boolean whether tokens of a turn should be considered for training. Useful to selectively train on certain turns besides the `roles_to_train`."
        },
    )
    message_field_training_detail: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "The key in the message turn that contains the training details. Useful to selectively train on certain tokens in a turn. The value of the key is a List[Dict] containing `begin_offset` (start character index in content), `end_offset` (end character index in content), and `train` (boolean whether to train)."
        },
    )
    split_thinking: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "(for Qwen3 template only) Whether to split the assistant content based on a reasoning trace inside delimited tags"
        },
    )
    logprobs_field: str | None = None
    temperature: float | None = None
    roles_to_train: list[str] | None = Field(
        default=None,
        json_schema_extra={
            "description": "Roles to train on. The tokens from these roles will be considered for the loss."
        },
    )
    train_on_eos: Literal["all", "turn", "last"] | None = Field(
        default=None,
        json_schema_extra={
            "description": "Which EOS tokens to train on in the conversation. Possible values are: all: train on all EOS tokens, turn (default): train on the EOS token at the end of each trainable turn, last: train on the last EOS token in the conversation"
        },
    )
    roles: dict[str, list[str]] | None = Field(
        default=None,
        json_schema_extra={
            "description": 'Roles mapping in the messages. The format is {target_role: [source_roles]}. All source roles will be mapped to the target role. The default is: user: ["human", "user"], assistant: ["gpt", "assistant"], system: ["system"], tool: ["tool"]'
        },
    )
    drop_system_message: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Whether to drop the system turn from the dataset. Only works with chat_template. This does not drop the default system message from chat_template if it exists. If you wish to, we recommend using a custom jinja template with the default system message removed or adding a system turn with empty content."
        },
    )
    trust_remote_code: bool | None = Field(
        default=False,
        json_schema_extra={"description": "Trust remote code for untrusted source"},
    )
    revision: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "The specific revision of the dataset to use when loading from the Hugging Face Hub. This can be a commit hash, tag, or branch name. If not specified, the latest version will be used. This parameter is ignored for local datasets."
        },
    )

    @model_validator(mode="before")
    @classmethod
    def handle_legacy_message_fields(cls, data):
        """Handle backwards compatibility between legacy message field mapping and new property mapping system."""
        return handle_legacy_message_fields_logic(data)

    @model_validator(mode="before")
    @classmethod
    def check_chat_template_config(cls, data):
        if isinstance(data, BaseModel):
            data = data.model_dump()

        # Set chat_template to tokenizer_default if not set
        if data.get("type") == "chat_template" and not data.get("chat_template"):
            data["chat_template"] = ChatTemplate.tokenizer_default

        # if chat_template is set to jinja, chat_template_jinja is required
        if data.get("chat_template") == ChatTemplate.jinja and not data.get(
            "chat_template_jinja"
        ):
            raise ValueError(
                "chat_template_jinja is required when chat_template is set to jinja"
            )

        # If chat_template_jinja is set, set chat_template to jinja
        if data.get("chat_template_jinja") and not data.get("chat_template"):
            data["chat_template"] = ChatTemplate.jinja

        return data


class PretrainingDataset(BaseModel):
    """Pretraining dataset configuration subset"""

    name: str | None = None
    path: str | None = None
    split: str | None = "train"
    text_column: str | None = "text"
    type: str | None = "pretrain"
    trust_remote_code: bool | None = False
    data_files: str | None = None
    skip: int | None = None


class UserDefinedDPOType(BaseModel):
    """User defined typing for DPO"""

    field_system: str | None = None
    field_prompt: str | None = None
    field_chosen: str | None = None
    field_rejected: str | None = None
    prompt_format: str | None = None
    chosen_format: str | None = None
    rejected_format: str | None = None


class DPODataset(BaseModel):
    """DPO configuration subset"""

    path: str | None = None
    split: str | None = None
    type: UserDefinedDPOType | str | None = None
    data_files: list[str] | None = None
    revision: str | None = None
    field_messages: str | None = None


class StepwiseSupervisedDataset(BaseModel):
    """Stepwise supervised dataset configuration subset"""

    path: str | None = None
    split: str | None = None
    data_files: list[str] | None = None
    revision: str | None = None
    step_separator: str | None = None
    max_completion_length: int | None = None
    train_on_last_step_only: bool | None = None


class UserDefinedKTOType(BaseModel):
    """User defined typing for KTO"""

    field_system: str | None = None
    field_prompt: str | None = None
    field_completion: str | None = None
    field_label: bool | None = None
    prompt_format: str | None = None
    completion_format: str | None = None


class KTODataset(BaseModel):
    """KTO configuration subset"""

    path: str | None = None
    split: str | None = None
    type: UserDefinedKTOType | str | None = None
    data_files: list[str] | None = None
    trust_remote_code: bool | None = False
    revision: str | None = None


DatasetConfig = SFTDataset | DPODataset | KTODataset | StepwiseSupervisedDataset
