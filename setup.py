"""setup.py for axolotl"""

import ast
import os
import platform
import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from setuptools import find_packages, setup


def parse_requirements(extras_require_map):
    _install_requires = []
    _dependency_links = []
    with open("./requirements.txt", encoding="utf-8") as requirements_file:
        lines = [r.strip() for r in requirements_file.readlines()]
        for line in lines:
            is_extras = "deepspeed" in line or "mamba-ssm" in line
            if line.startswith("--extra-index-url"):
                # Handle custom index URLs
                _, url = line.split()
                _dependency_links.append(url)
            elif not is_extras and line and line[0] != "#":
                # Handle standard packages
                _install_requires.append(line)
    try:
        xformers_version = [req for req in _install_requires if "xformers" in req][0]
        if "Darwin" in platform.system():
            # skip packages not compatible with OSX
            skip_packages = [
                "bitsandbytes",
                "triton",
                "mamba-ssm",
                "xformers",
                "liger-kernel",
            ]
            _install_requires = [
                req
                for req in _install_requires
                if re.split(r"[>=<]", req)[0].strip() not in skip_packages
            ]
            print(
                _install_requires, [req in skip_packages for req in _install_requires]
            )
        else:
            # detect the version of torch already installed
            # and set it so dependencies don't clobber the torch version
            try:
                torch_version = version("torch")
                _install_requires.append(f"torch=={torch_version}")
            except PackageNotFoundError:
                # Use flexible version range to support FlexAI torch 2.8
                _install_requires.append("torch>=2.6.0,<=2.8.1")
                torch_version = "2.8.0"  # default to torch 2.8 for FlexAI

            version_match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", torch_version)
            if version_match:
                major, minor, patch = version_match.groups()
                major, minor = int(major), int(minor)
                patch = (
                    int(patch) if patch is not None else 0
                )  # Default patch to 0 if not present
            else:
                raise ValueError("Invalid version format")

            if (major, minor) >= (2, 8):
                pass
            elif (major, minor) >= (2, 7):
                _install_requires.pop(_install_requires.index(xformers_version))
                if patch == 0:
                    _install_requires.append("xformers==0.0.30")
                    # vllm 0.9.x is incompatible with latest transformers
                    extras_require_map.pop("vllm")
                else:
                    _install_requires.append("xformers==0.0.31")
                    extras_require_map["vllm"] = ["vllm>=0.10.0"]
            elif (major, minor) >= (2, 6):
                _install_requires.pop(_install_requires.index(xformers_version))
                _install_requires.append("xformers==0.0.29.post3")
                # since we only support 2.6.0+cu126
                _dependency_links.append("https://download.pytorch.org/whl/cu126")
                extras_require_map.pop("vllm")
            elif (major, minor) >= (2, 5):
                _install_requires.pop(_install_requires.index(xformers_version))
                if patch == 0:
                    _install_requires.append("xformers==0.0.28.post2")
                else:
                    _install_requires.append("xformers>=0.0.28.post3")
                extras_require_map.pop("vllm")
            elif (major, minor) >= (2, 4):
                extras_require_map.pop("vllm")
                if patch == 0:
                    _install_requires.pop(_install_requires.index(xformers_version))
                    _install_requires.append("xformers>=0.0.27")
                else:
                    _install_requires.pop(_install_requires.index(xformers_version))
                    _install_requires.append("xformers==0.0.28.post1")
            else:
                raise ValueError("axolotl requires torch>=2.4")

    except PackageNotFoundError:
        pass
    return _install_requires, _dependency_links, extras_require_map


def get_package_version():
    with open(
        Path(os.path.dirname(os.path.abspath(__file__)))
        / "src"
        / "axolotl"
        / "__init__.py",
        "r",
        encoding="utf-8",
    ) as fin:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", fin.read(), re.MULTILINE)
    version_ = ast.literal_eval(version_match.group(1))
    return version_


extras_require = {
    "flash-attn": ["flash-attn==2.8.3"],
    "ring-flash-attn": [
        "flash-attn==2.8.3",
        "ring-flash-attn>=0.1.7",
    ],
    "deepspeed": [
        "deepspeed==0.17.5",
        "deepspeed-kernels",
    ],
    "mamba-ssm": [
        "mamba-ssm==1.2.0.post1",
        "causal_conv1d",
    ],
    "auto-gptq": [
        "auto-gptq==0.5.1",
    ],
    "mlflow": [
        "mlflow",
    ],
    "galore": [
        "galore_torch",
    ],
    "apollo": [
        "apollo-torch",
    ],
    "optimizers": [
        "galore_torch",
        "apollo-torch",
        "lomo-optim==0.1.1",
        "torch-optimi==0.2.1",
        "came_pytorch==0.1.3",
    ],
    "ray": [
        "ray[train]",
    ],
    "vllm": [
        "vllm==0.10.0",
    ],
    "llmcompressor": [
        "llmcompressor==0.5.1",
    ],
    "fbgemm-gpu": ["fbgemm-gpu-genai>=1.2.0"],
    "opentelemetry": [
        "opentelemetry-api",
        "opentelemetry-sdk",
        "opentelemetry-exporter-prometheus",
        "prometheus-client",
    ],
}
install_requires, dependency_links, extras_require_build = parse_requirements(
    extras_require
)

setup(
    version=get_package_version(),
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=install_requires,
    dependency_links=dependency_links,
    entry_points={
        "console_scripts": [
            "axolotl=axolotl.cli.main:main",
        ],
    },
    extras_require=extras_require_build,
)
