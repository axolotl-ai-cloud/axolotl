"""
dynamic requirements for axolotl
"""

import platform
import re
from importlib.metadata import PackageNotFoundError, version

from setuptools.command.build_py import build_py as _build_py


def parse_requirements():
    _install_requires = []
    _dependency_links = []
    with open("./requirements.txt", encoding="utf-8") as requirements_file:
        lines = [r.strip() for r in requirements_file.readlines()]
        for line in lines:
            is_extras = (
                "flash-attn" in line
                or "flash-attention" in line
                or "deepspeed" in line
                or "mamba-ssm" in line
                or "lion-pytorch" in line
            )
            if line.startswith("--extra-index-url"):
                # Handle custom index URLs
                _, url = line.split()
                _dependency_links.append(url)
            elif not is_extras and line and line[0] != "#":
                # Handle standard packages
                _install_requires.append(line)

    try:
        xformers_version = next((r for r in _install_requires if "xformers" in r), None)
        torchao_version = next((r for r in _install_requires if "torchao" in r), None)
        autoawq_version = next((r for r in _install_requires if "autoawq" in r), None)

        sys_platform = platform.system()
        # On macOS/Windows skip xformers adjustments entirely (build friction); leave if user later adds manually.
        if sys_platform in ("Darwin", "Windows") and xformers_version and xformers_version in _install_requires:
            _install_requires.remove(xformers_version)
            xformers_version = None

        # Always pin torch to already-installed version (or default) to avoid clobber.
        try:
            torch_version = version("torch")
        except PackageNotFoundError:
            torch_version = "2.5.1"
        if not any(req.startswith("torch==") for req in _install_requires):
            _install_requires.append(f"torch=={torch_version}")

        # If xformers not present in requirements list, nothing else to massage.
        if not xformers_version:
            return _install_requires, _dependency_links

        version_match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", torch_version)
        if version_match:
            major, minor, patch = version_match.groups()
            major, minor = int(major), int(minor)
            patch = int(patch) if patch is not None else 0
        else:
            raise ValueError("Invalid version format")

        def replace_xformers(new_spec: str):
            if xformers_version in _install_requires:
                _install_requires.remove(xformers_version)
            _install_requires.append(new_spec)

        if (major, minor) >= (2, 5):
            replace_xformers("xformers==0.0.28.post2" if patch == 0 else "xformers==0.0.28.post3")
            if autoawq_version and autoawq_version in _install_requires:
                _install_requires.remove(autoawq_version)
        elif (major, minor) >= (2, 4):
            replace_xformers("xformers>=0.0.27" if patch == 0 else "xformers==0.0.28.post1")
        elif (major, minor) >= (2, 3):
            if torchao_version and torchao_version in _install_requires:
                _install_requires.remove(torchao_version)
            replace_xformers("xformers>=0.0.26.post1" if patch == 0 else "xformers>=0.0.27")
        elif (major, minor) >= (2, 2):
            if torchao_version and torchao_version in _install_requires:
                _install_requires.remove(torchao_version)
            replace_xformers("xformers>=0.0.25.post1")
        else:
            if torchao_version and torchao_version in _install_requires:
                _install_requires.remove(torchao_version)
            replace_xformers("xformers>=0.0.23.post1")

    except PackageNotFoundError:
        pass
    except Exception as exc:
        print(f"[axolotl setup dynamic] Non-fatal dependency resolution issue: {exc}")
    return _install_requires, _dependency_links


class BuildPyCommand(_build_py):
    """
    custom build_py command to parse dynamic requirements
    """

    def finalize_options(self):
        super().finalize_options()
        install_requires, _ = parse_requirements()
        self.distribution.install_requires = install_requires
