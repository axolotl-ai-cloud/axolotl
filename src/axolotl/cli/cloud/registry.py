"""Discover cloud launchers without importing provider or training dependencies."""

from importlib.metadata import EntryPoint, entry_points

from axolotl.cli.cloud.base import Cloud

ENTRY_POINT_GROUP = "axolotl.cloud_providers"

BUILTIN_PROVIDERS = {
    "modal": "axolotl.integrations.modal.cloud:ModalCloud",
    "baseten": "axolotl.integrations.baseten.cloud:BasetenCloud",
}


def load_cloud_provider(config: dict) -> Cloud:
    """Instantiate the selected provider with its cloud configuration.

    Entry points must resolve to a Cloud subclass accepting a configuration dict.
    Built-in names are reserved. Only the selected provider is imported.
    """
    name = config.get("provider")
    if name is None or name == "":
        name = "modal"
    if not isinstance(name, str):
        raise ValueError("Cloud provider must be a string")
    if name in BUILTIN_PROVIDERS:
        entry_point = EntryPoint(
            name=name, value=BUILTIN_PROVIDERS[name], group=ENTRY_POINT_GROUP
        )
    else:
        matches = [
            entry_point
            for entry_point in entry_points(group=ENTRY_POINT_GROUP)
            if entry_point.name == name
        ]
        if not matches:
            raise ValueError(
                f"Unsupported cloud provider: {name}. "
                "Install a package registering it under axolotl.cloud_providers."
            )
        if len(matches) > 1:
            raise ValueError(f"Multiple cloud providers registered as {name!r}")
        entry_point = matches[0]

    provider = entry_point.load()
    if not isinstance(provider, type) or not issubclass(provider, Cloud):
        raise TypeError(f"Cloud provider {name!r} must be a Cloud subclass")
    return provider(dict(config))
