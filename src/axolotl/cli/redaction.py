"""Utils for redaction of sensitive information in config."""

from pathlib import Path
from typing import Any

import yaml

# NOTE: Borrowed from the telemetry logic. Should be unified with it once merged.
WHITELIST_PATH = str(Path(__file__).parent / "redaction_whitelist.yaml")

with open(WHITELIST_PATH, encoding="utf-8") as f:
    WHITELIST = yaml.safe_load(f)

    # Send org strings to lowercase since model names are case insensitive
    WHITELIST["organizations"] = {org.lower() for org in WHITELIST["organizations"]}


# NOTE: Need to keep these up to date with any config schema changes.
FIELDS_TO_REDACT = {
    "base_model",
    "tokenizer_config",
    "base_model_config",
    "pretraining_dataset",  # NOTE: this field may be a string or a dictionary.
    "resume_from_checkpoint",
    "hub_model_id",
}
PREFIXES_TO_REDACT = {"wandb_", "comet_", "mlflow_", "gradio_"}
PATH_INDICATORS = {"path", "dir"}


def is_whitelisted(value: str) -> bool:
    """
    Check if model / dataset / etc. org is in whitelist.

    This logic is borrowed from the telemetry logic. Should be unified with it once
    merged.

    Args:
        value: Value for one of `FIELDS_WITH_ORGS` ("base_model", etc.).

    Returns:
        Boolean indicating whitelist membership.
    """
    # NOTE: This membership-checking logic can be improved.
    # What happens when a local model path matches a whitelisted org?
    parts = value.split("/")
    if len(parts) < 2:
        return False
    org = parts[0]
    whitelisted = org.lower() in WHITELIST["organizations"]

    return whitelisted


def redact_sensitive_info(properties: dict[str, Any]) -> dict[str, Any]:
    """
    Redact properties to remove any paths, API keys, etc., so as to avoid collecting
    private or personally identifiable information (PII).

    This logic is borrowed from the telemetry logic. It can be unified with it once
    merged.

    Args:
        properties: Dictionary of properties to redact.

    Returns:
        Properties dictionary with redaction applied.
    """
    if not properties:
        return {}

    def redact_value(value: Any, key: str = "") -> Any:
        """Recursively sanitize values, redacting those with path-like keys"""
        if isinstance(key, str) and isinstance(value, str):
            # Other redaction special cases
            if (
                key in FIELDS_TO_REDACT
                or any(prefix in key for prefix in PREFIXES_TO_REDACT)
                or any(indicator in key.lower() for indicator in PATH_INDICATORS)
            ):
                # Fields with whitelisted orgs don't need to be redacted
                if not is_whitelisted(value):
                    return "[REDACTED]"

        # Handle nested values
        if isinstance(value, dict):
            return {k: redact_value(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [redact_value(item) for item in value]

        return value

    # Create new dict with redacted values
    redacted = {k: redact_value(v, k) for k, v in properties.items()}

    return redacted
