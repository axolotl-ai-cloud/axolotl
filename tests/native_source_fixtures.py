"""Pinned source-only remote-code fixtures for native diffusion tests."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

NATIVE_SOURCE_FIXTURES_ENV = "AXOLOTL_NATIVE_SOURCE_FIXTURES"


@dataclass(frozen=True)
class NativeSourceFixture:
    repository: str
    revision: str
    files: dict[str, str]


_FIXTURES = {
    "dream": NativeSourceFixture(
        repository="Dream-org/Dream-v0-Instruct-7B",
        revision="05334cb9faaf763692dcf9d8737c642be2b2a6ae",
        files={
            "config.json": "11afc7fe7a1881e73dcd70cca6868aaaf8eb89ad45ea5fc6426c3fb82dec9340",
            "configuration_dream.py": "24d038962ccf16361595494461000ec0c1b043653c7494d7fc835c96f121f0b9",
            "generation_config.json": "e4fa42a31ef22740804534395c7a18f335a3c1f9abd9db83a6941b1c8748b2ee",
            "generation_utils.py": "7f8ad01484898946b3c9c9d5ebc85e5ac726061d0fa7b4a522c4b8110c13a921",
            "modeling_dream.py": "3166e789f0d69beb1f4fbdee2317953d1c1c3b9bc21479b109afc5dd059de5b3",
        },
    ),
    "nemotron": NativeSourceFixture(
        repository="nvidia/Nemotron-Labs-Diffusion-3B",
        revision="0d51902da1f8869f83413ce642fab402fa5641e0",
        files={
            "config.json": "fc6ec1011e5f4ff858a104e8e04599e931e45dc874c71b9e73182a7f2dbdec44",
            "configuration_nemotron_labs_diffusion.py": "2b18216e1b4e0d89b728c1c871744088a28004564f99009809294b39ec677b57",
            "modeling_ministral.py": "021afd8a72e23c5bf3a309f9dc5f0a7771db2424c148be5ce2a2737bf4cb7d65",
            "modeling_nemotron_labs_diffusion.py": "29d73c5709e90e3be3c7e537edb61a84fbb3dc1c286b1eaed42d899a3a4e4760",
        },
    ),
}


def _fixture(name: str) -> NativeSourceFixture:
    try:
        return _FIXTURES[name]
    except KeyError as error:
        raise ValueError(f"unknown native source fixture: {name}") from error


def validate_native_source_fixture(name: str, directory: str | Path) -> Path:
    """Return a fixture directory only after every downloaded file is verified."""

    fixture = _fixture(name)
    directory = Path(directory)
    for filename, expected_hash in fixture.files.items():
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"native {name} source fixture is missing {filename}: {directory}"
            )
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(
                f"native {name} source fixture hash mismatch for {filename}"
            )
    return directory


def native_source_fixture_path(name: str) -> Path | None:
    """Return a configured, verified local fixture or ``None`` when unavailable."""

    root = os.environ.get(NATIVE_SOURCE_FIXTURES_ENV)
    if not root:
        return None
    return validate_native_source_fixture(name, Path(root) / name)


def prepare_native_source_fixtures(
    destination: str | Path, *, offline: bool = False
) -> dict[str, Path]:
    """Download only audited remote-code/config files and verify them before use."""

    destination = Path(destination)
    prepared: dict[str, Path] = {}
    for name, fixture in _FIXTURES.items():
        directory = destination / name
        try:
            prepared[name] = validate_native_source_fixture(name, directory)
            continue
        except FileNotFoundError:
            if offline:
                raise
        except ValueError:
            if offline:
                raise
        if offline:
            raise FileNotFoundError(
                f"native {name} source fixture is unavailable offline: {directory}"
            )
        from huggingface_hub import snapshot_download

        snapshot_download(
            fixture.repository,
            revision=fixture.revision,
            allow_patterns=sorted(fixture.files),
            local_dir=directory,
            force_download=True,
        )
        prepared[name] = validate_native_source_fixture(name, directory)
    return prepared
