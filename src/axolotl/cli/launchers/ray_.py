"""Ray launcher backend: in-cluster Ray Train driver and Ray Jobs submission.

Named ray_ to avoid shadowing the ray package.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from axolotl.utils.schemas.runtime import RuntimeConfig


def launch_ray_training(
    cfg_file: str,
    kwargs: dict,
    runtime: "RuntimeConfig | None" = None,
    env: dict[str, str] | None = None,
) -> None:
    """Run training through Ray Train (in-process driver) or the Ray Jobs API."""
    from axolotl.cli.train import do_cli

    do_cli(config=cfg_file, **kwargs)
