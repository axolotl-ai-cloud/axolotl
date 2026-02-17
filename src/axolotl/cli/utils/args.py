"""Utilities for axolotl CLI args."""

import dataclasses
from functools import wraps
from types import NoneType, UnionType
from typing import Any, Callable, Type, Union, get_args, get_origin

import click
from pydantic import BaseModel


def _strip_optional_type(field_type: type | str | None):
    """
    Extracts the non-`None` type from an `Optional` / `Union` type.

    Args:
        field_type: Type of field for Axolotl CLI command.

    Returns:
        If the input type is `Union[T, None]` or `Optional[T]`, returns `T`. Otherwise
            returns the input type unchanged.
    """
    is_union = get_origin(field_type) is Union or isinstance(field_type, UnionType)
    if is_union and type(None) in get_args(field_type):
        field_type = next(
            t for t in get_args(field_type) if not isinstance(t, NoneType)
        )

    return field_type


def filter_none_kwargs(func: Callable) -> Callable:
    """
    Wraps function to remove `None`-valued `kwargs`.

    Args:
        func: Function to wrap.

    Returns:
        Wrapped function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> Callable:
        """Filters out `None`-valued `kwargs`."""
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}

        return func(*args, **filtered_kwargs)

    return wrapper


def add_options_from_dataclass(config_class: Type[Any]) -> Callable:
    """
    Create Click options from the fields of a dataclass.

    Args:
        config_class: Dataclass with fields to parse from the CLI.

    Returns:
        Function decorator for Axolotl CLI command.
    """

    def decorator(function: Callable) -> Callable:
        # Process dataclass fields in reverse order for correct option ordering
        for field in reversed(dataclasses.fields(config_class)):
            field_type = _strip_optional_type(field.type)

            if field_type is bool:
                field_name = field.name.replace("_", "-")
                option_name = f"--{field_name}/--no-{field_name}"
                function = click.option(
                    option_name,
                    default=field.default,
                    help=field.metadata.get("description"),
                )(function)
            else:
                option_name = f"--{field.name.replace('_', '-')}"
                function = click.option(
                    option_name,
                    type=field_type,
                    default=field.default,
                    help=field.metadata.get("description"),
                )(function)

        return function

    return decorator


def _is_pydantic_model(field_type: type) -> bool:
    """Check if a type is a Pydantic BaseModel subclass."""
    try:
        return isinstance(field_type, type) and issubclass(field_type, BaseModel)
    except TypeError:
        return False


def _get_field_description(field) -> str | None:
    """Get description from a Pydantic field, checking both .description and json_schema_extra."""
    if field.description:
        return field.description
    if field.json_schema_extra and isinstance(field.json_schema_extra, dict):
        return field.json_schema_extra.get("description")
    return None


def _add_nested_model_options(
    function: Callable, parent_name: str, model_class: Type[BaseModel]
) -> Callable:
    """
    Add Click options for all fields of a nested Pydantic model using dot-notation.

    Args:
        function: Click command function to add options to.
        parent_name: Parent field name (e.g., "trl").
        model_class: Nested Pydantic model class.

    Returns:
        Function with added Click options.
    """
    for sub_name, sub_field in reversed(model_class.model_fields.items()):
        sub_type = _strip_optional_type(sub_field.annotation)
        # Use dot notation: --parent.sub_field
        cli_name = f"{parent_name}.{sub_name}".replace("_", "-")
        # The kwarg name uses double-underscore as separator
        param_name = f"{parent_name}__{sub_name}"
        description = _get_field_description(sub_field)

        if sub_type is bool:
            option_name = f"--{cli_name}/--no-{cli_name}"
            function = click.option(
                option_name, param_name, default=None, help=description
            )(function)
        else:
            option_name = f"--{cli_name}"
            function = click.option(
                option_name, param_name, default=None, help=description
            )(function)

    return function


def add_options_from_config(config_class: Type[BaseModel]) -> Callable:
    """
    Create Click options from the fields of a Pydantic model.

    For fields whose type is itself a Pydantic BaseModel, dot-notation CLI options are
    generated for each sub-field (e.g., ``--trl.beta=0.1``).

    Args:
        config_class: PyDantic model with fields to parse from the CLI

    Returns:
        Function decorator for Axolotl CLI command.
    """

    def decorator(function: Callable) -> Callable:
        # Process model fields in reverse order for correct option ordering
        for name, field in reversed(config_class.model_fields.items()):
            field_type = _strip_optional_type(field.annotation)

            # Handle nested Pydantic models with dot-notation options
            if _is_pydantic_model(field_type):
                function = _add_nested_model_options(function, name, field_type)
                continue

            if field_type is bool:
                field_name = name.replace("_", "-")
                option_name = f"--{field_name}/--no-{field_name}"
                function = click.option(
                    option_name, default=None, help=field.description
                )(function)
            else:
                option_name = f"--{name.replace('_', '-')}"
                function = click.option(
                    option_name, default=None, help=field.description
                )(function)

        return function

    return decorator
