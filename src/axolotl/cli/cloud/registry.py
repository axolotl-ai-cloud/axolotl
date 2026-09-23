"""Discover cloud launchers without importing provider or training dependencies."""

from importlib.metadata import EntryPoint, entry_points
from pathlib import Path

from axolotl.cli.cloud.base import CloudLauncher

ENTRY_POINT_GROUP = "axolotl.cloud_providers"

BUILTIN_PROVIDERS = {
    "nebius": "axolotl.integrations.nebius.cloud:NebiusCloud",
    "modal": "axolotl.integrations.modal.cloud:ModalCloud",
    "baseten": "axolotl.integrations.baseten.cloud:BasetenCloud",
}


def load_cloud_provider(
    config: dict, *, config_dir: Path | None = None
) -> CloudLauncher:
    """Instantiate the selected whole-job launcher with its cloud configuration.

    Resolve installed entry points or an explicit ``module:class`` target.
    Built-in names are reserved; their targets also work from source checkouts
    with missing or stale package metadata. Only the selected provider is imported.
    """
    name = config.get("provider")
    if name is None or name == "":
        name = "modal"
    if not isinstance(name, str):
        raise ValueError("Cloud provider must be a string")
    if ":" in name:
        entry_point = EntryPoint(name=name, value=name, group=ENTRY_POINT_GROUP)
    else:
        matches = [
            point
            for point in entry_points(group=ENTRY_POINT_GROUP)
            if point.name == name
        ]
        builtin = BUILTIN_PROVIDERS.get(name)
        if builtin:
            matches = [
                point
                for point in matches
                if point.value == builtin
                and point.dist is not None
                and point.dist.name == "axolotl"
            ]
        if len(matches) > 1:
            raise ValueError(f"Multiple cloud providers registered as {name!r}")
        if matches:
            entry_point = matches[0]
        elif builtin:
            entry_point = EntryPoint(name=name, value=builtin, group=ENTRY_POINT_GROUP)
        else:
            raise ValueError(
                f"Unsupported cloud provider: {name}. "
                "Install a package registering it under axolotl.cloud_providers "
                "or set provider to '<module>:<class>'."
            )

    provider = entry_point.load()
    if not isinstance(provider, type) or not issubclass(provider, CloudLauncher):
        raise TypeError(f"Cloud provider {name!r} must be a CloudLauncher subclass")
    return provider.from_config(dict(config), config_dir=config_dir)
