# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI

"""Soft dependency helpers for Arctic Platform."""

from __future__ import annotations

_INSTALL_HINT = (
    "Arctic SFT requires the arctic-platform package.\n\n"
    "Install with:\n"
    "  pip install arctic-platform\n\n"
    "Or from a checkout of the arctic-platform repo:\n"
    "  git clone https://github.com/Snowflake-AI-Research/arctic-platform\n"
    "  cd arctic-platform\n"
    "  pip install -e .\n"
)


def require_arctic_platform():
    """Import ``arctic_platform`` or raise a helpful ``ImportError``.

    Soft dependency: Axolotl itself does not hard-require arctic_platform; the
    ArcticSFT plugin checks at the point it needs the client.
    """
    try:
        import arctic_platform
    except ImportError as err:
        raise ImportError(f"{_INSTALL_HINT}\nOriginal error: {err}") from err
    return arctic_platform


def require_arctic_sft_client():
    """Import ``ArcticSFTClient`` / ``ArcticSFTClientConfig`` or raise helpfully."""
    require_arctic_platform()
    try:
        from arctic_platform.sft import ArcticSFTClient
        from arctic_platform.sft import ArcticSFTClientConfig
    except ImportError as err:
        raise ImportError(f"{_INSTALL_HINT}\nOriginal error: {err}") from err
    return ArcticSFTClient, ArcticSFTClientConfig
