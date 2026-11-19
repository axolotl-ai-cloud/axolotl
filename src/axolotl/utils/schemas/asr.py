"""Pydantic models for speech recognition (ASR) fine-tuning configuration"""

from pydantic import BaseModel, Field


class AsrConfig(BaseModel):
    """Dataset columns and decoder prompt options for speech seq2seq training"""

    audio_column: str = Field(
        default="audio",
        json_schema_extra={
            "description": "Dataset column holding the audio: a `datasets.Audio` feature, a {'array', 'sampling_rate'} dict, raw bytes, or a file path."
        },
    )
    text_column: str = Field(
        default="text",
        json_schema_extra={"description": "Dataset column holding the transcript."},
    )
    language: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Language code used in the decoder prompt for rows without a `language_column` value, e.g. 'en'."
        },
    )
    language_column: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Optional dataset column holding a per-row language code. Falls back to `language` when empty."
        },
    )
    punctuation: bool = Field(
        default=True,
        json_schema_extra={
            "description": "Whether the decoder prompt requests punctuated, cased output."
        },
    )
