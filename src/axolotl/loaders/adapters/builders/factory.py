from typing import Dict, Type

from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

from .base import BaseAdapterBuilder
from .lora import LoraAdapterBuilder

LOG = get_logger(__name__)


class AdapterBuilderFactory:
    """Factory for creating adapter builders based on adapter type."""

    _builders: Dict[str, Type[BaseAdapterBuilder]] = {
        "lora": LoraAdapterBuilder,
        "qlora": LoraAdapterBuilder,
    }

    @classmethod
    def register_builder(
        cls, adapter_type: str, builder_class: Type[BaseAdapterBuilder]
    ):
        """
        Register a new adapter builder.

        Args:
            adapter_type: Type of adapter (e.g., 'lora', 'qlora')
            builder_class: Builder class that extends BaseAdapterBuilder
        """
        cls._builders[adapter_type] = builder_class
        LOG.info(
            f"Registered adapter builder for '{adapter_type}': {builder_class.__name__}"
        )

    @classmethod
    def create_builder(cls, adapter_type: str, cfg: DictDefault) -> BaseAdapterBuilder:
        """
        Create an adapter builder for the specified type.

        Args:
            adapter_type: Type of adapter to create builder for
            cfg: Configuration object

        Returns:
            BaseAdapterBuilder: Configured adapter builder

        Raises:
            ValueError: If adapter type is not supported
        """
        if adapter_type not in cls._builders:
            available_types = list(cls._builders.keys())
            raise ValueError(
                f"Unsupported adapter type: {adapter_type}. "
                f"Available types: {available_types}"
            )

        builder_class = cls._builders[adapter_type]
        LOG.info(f"Creating {builder_class.__name__} for adapter type '{adapter_type}'")
        return builder_class(cfg)

    @classmethod
    def get_supported_adapters(cls) -> list[str]:
        """
        Get list of supported adapter types.

        Returns:
            list[str]: List of supported adapter type names
        """
        return list(cls._builders.keys())
