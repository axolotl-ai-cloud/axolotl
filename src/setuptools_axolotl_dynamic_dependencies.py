"""
dynamic requirements for axolotl
"""

import platform
import re
from importlib.metadata import PackageNotFoundError, version

from setuptools.command.build_py import build_py as _build_py


# pylint: disable=duplicate-code
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
        xformers_version = [req for req in _install_requires if "xformers" in req][0]
        torchao_version = [req for req in _install_requires if "torchao" in req][0]
        autoawq_version = [req for req in _install_requires if "autoawq" in req][0]

        if "Darwin" in platform.system():
            # don't install xformers on MacOS
            _install_requires.pop(_install_requires.index(xformers_version))
        else:
            # detect the version of torch already installed
            # and set it so dependencies don't clobber the torch version
            try:
                torch_version = version("torch")
            except PackageNotFoundError:
                torch_version = "2.5.1"
            _install_requires.append(f"torch=={torch_version}")

            version_match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", torch_version)
            if version_match:
                major, minor, patch = version_match.groups()
                major, minor = int(major), int(minor)
                patch = (
                    int(patch) if patch is not None else 0
                )  # Default patch to 0 if not present
            else:
                raise ValueError("Invalid version format")

            if (major, minor) >= (2, 5):
                _install_requires.pop(_install_requires.index(xformers_version))
                if patch == 0:
                    _install_requires.append("xformers==0.0.28.post2")
                else:
                    _install_requires.append("xformers==0.0.28.post3")
                _install_requires.pop(_install_requires.index(autoawq_version))
            elif (major, minor) >= (2, 4):
                if patch == 0:
                    _install_requires.pop(_install_requires.index(xformers_version))
                    _install_requires.append("xformers>=0.0.27")
                else:
                    _install_requires.pop(_install_requires.index(xformers_version))
                    _install_requires.append("xformers==0.0.28.post1")
            elif (major, minor) >= (2, 3):
                _install_requires.pop(_install_requires.index(torchao_version))
                if patch == 0:
                    _install_requires.pop(_install_requires.index(xformers_version))
                    _install_requires.append("xformers>=0.0.26.post1")
                else:
                    _install_requires.pop(_install_requires.index(xformers_version))
                    _install_requires.append("xformers>=0.0.27")
            elif (major, minor) >= (2, 2):
                _install_requires.pop(_install_requires.index(torchao_version))
                _install_requires.pop(_install_requires.index(xformers_version))
                _install_requires.append("xformers>=0.0.25.post1")
            else:
                _install_requires.pop(_install_requires.index(torchao_version))
                _install_requires.pop(_install_requires.index(xformers_version))
                _install_requires.append("xformers>=0.0.23.post1")

    except PackageNotFoundError:
        pass
    return _install_requires, _dependency_links


class BuildPyCommand(_build_py):
    """
    custom build_py command to parse dynamic requirements
    """

    def finalize_options(self):
        super().finalize_options()
        install_requires, _ = parse_requirements()
        self.distribution.install_requires = install_requires
