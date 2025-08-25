"""Utilities for axolotl CLI args."""

import dataclasses
from functools import wraps
from types import NoneType
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
    if get_origin(field_type) is Union and type(None) in get_args(field_type):
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


def add_options_from_config(config_class: Type[BaseModel]) -> Callable:
    """
    Create Click options from the fields of a Pydantic model.

    Args:
        config_class: PyDantic model with fields to parse from the CLI

    Returns:
        Function decorator for Axolotl CLI command.
    """

    def decorator(function: Callable) -> Callable:
        # Process model fields in reverse order for correct option ordering
        for name, field in reversed(config_class.model_fields.items()):
            field_type = _strip_optional_type(field.annotation)

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
