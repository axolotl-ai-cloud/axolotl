"""Utilities for Axolotl Pydantic models"""

import logging

LOG = logging.getLogger(__name__)


def handle_legacy_message_fields_logic(data: dict) -> dict:
    """
    Handle backwards compatibility between legacy message field mapping and new property mapping system.

    Previously, the config only supported mapping 'role' and 'content' fields via dedicated config options:
    - message_field_role: Mapped to the role field
    - message_field_content: Mapped to the content field

    The new system uses message_property_mappings to support arbitrary field mappings:
    message_property_mappings:
        role: source_role_field
        content: source_content_field
        additional_field: source_field

    Args:
        data: Dictionary containing configuration data

    Returns:
        Updated dictionary with message field mappings consolidated

    Raises:
        ValueError: If there are conflicts between legacy and new mappings
    """
    data = data.copy()  # Create a copy to avoid modifying the original

    if data.get("message_property_mappings") is None:
        data["message_property_mappings"] = {}

    # Check for conflicts and handle role
    if "message_field_role" in data:
        LOG.warning(
            "message_field_role is deprecated, use message_property_mappings instead. "
            f"Example: message_property_mappings: {{role: {data['message_field_role']}}}"
        )
        if (
            "role" in data["message_property_mappings"]
            and data["message_property_mappings"]["role"] != data["message_field_role"]
        ):
            raise ValueError(
                f"Conflicting message role fields: message_field_role='{data['message_field_role']}' "
                f"conflicts with message_property_mappings.role='{data['message_property_mappings']['role']}'"
            )
        data["message_property_mappings"]["role"] = data["message_field_role"] or "role"

        del data["message_field_role"]
    elif "role" not in data["message_property_mappings"]:
        data["message_property_mappings"]["role"] = "role"

    # Check for conflicts and handle content
    if "message_field_content" in data:
        LOG.warning(
            "message_field_content is deprecated, use message_property_mappings instead. "
            f"Example: message_property_mappings: {{content: {data['message_field_content']}}}"
        )
        if (
            "content" in data["message_property_mappings"]
            and data["message_property_mappings"]["content"]
            != data["message_field_content"]
        ):
            raise ValueError(
                f"Conflicting message content fields: message_field_content='{data['message_field_content']}' "
                f"conflicts with message_property_mappings.content='{data['message_property_mappings']['content']}'"
            )
        data["message_property_mappings"]["content"] = (
            data["message_field_content"] or "content"
        )

        del data["message_field_content"]
    elif "content" not in data["message_property_mappings"]:
        data["message_property_mappings"]["content"] = "content"

    return data
