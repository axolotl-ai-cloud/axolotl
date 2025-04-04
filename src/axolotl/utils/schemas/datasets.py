"""Pydantic models for datasets-related configuration"""

from pydantic import BaseModel, model_validator

from axolotl.utils.schemas.enums import ChatTemplate
from axolotl.utils.schemas.utils import handle_legacy_message_fields_logic


class UserDefinedPrompterType(BaseModel):
    """Structure for user defined prompt types"""

    system_prompt: str | None = None
    system_format: str | None = None
    field_system: str | None = None
    field_instruction: str | None = None
    field_input: str | None = None
    field_output: str | None = None

    format: str | None = None
    no_input_format: str | None = None
    field: str | None = None


class SFTDataset(BaseModel):
    """SFT configuration subset"""

    path: str | None = None
    split: str | None = None
    type: str | UserDefinedPrompterType | None = None
    input_transform: str | None = None
    shards: int | None = None
    shards_idx: int | None = None
    preprocess_shards: int | None = None
    conversation: str | None = None
    # Do not make this too strict or it will break the validator to choose different dataset class
    chat_template: ChatTemplate | str | None = None
    chat_template_jinja: str | None = None
    data_files: str | list[str] | None = None
    input_format: str | None = None
    name: str | None = None
    ds_type: str | None = None
    field: str | None = None
    field_human: str | None = None
    field_model: str | None = None
    field_messages: str | None = None
    # deprecated, use message_property_mappings
    message_field_role: str | None = None
    # deprecated, use message_property_mappings
    message_field_content: str | None = None
    message_property_mappings: dict[str, str] | None = None
    message_field_training: str | None = None
    message_field_training_detail: str | None = None
    logprobs_field: str | None = None
    temperature: float | None = None
    roles_to_train: list[str] | None = None
    train_on_eos: str | None = None
    roles: dict[str, list[str]] | None = None
    drop_system_message: bool | None = None
    trust_remote_code: bool | None = False
    revision: str | None = None

    @model_validator(mode="before")
    @classmethod
    def handle_legacy_message_fields(cls, data):
        """Handle backwards compatibility between legacy message field mapping and new property mapping system."""
        return handle_legacy_message_fields_logic(data)

    @model_validator(mode="before")
    @classmethod
    # pylint: disable=duplicate-code
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
