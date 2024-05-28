"""setup.py for axolotl"""

import platform
import re
from importlib.metadata import PackageNotFoundError, version

from setuptools import find_packages, setup


def extract_requirements_info(requirement_line):
    pattern = r"(?P<egg>\w+) @ git\+https://github.com/(?P<namespace>[\w-]+)/(?P<repo>[\w-]+)\.git@(?P<gitsha>[a-f0-9]{40})"
    match = re.match(pattern, requirement_line)
    if match:
        info = match.groupdict()
        info["namespace/repo"] = f"{info.pop('namespace')}/{info.pop('repo')}"
        return info
    raise ValueError("The requirement line is not in the expected format")


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
                if " @ " in line:
                    req_data = extract_requirements_info(line)
                    _dependency_links.append(
                        f"https://github.com/{req_data['namespace/repo']}/tarball/{req_data['gitsha']}#egg={req_data['egg']}"
                    )
                else:
                    # Handle standard packages
                    _install_requires.append(line)

    try:
        if "Darwin" in platform.system():
            # don't install xformers on MacOS
            _install_requires.pop(_install_requires.index("xformers==0.0.26.post1"))
        else:
            # detect the version of torch already installed
            # and set it so dependencies don't clobber the torch version
            torch_version = version("torch")
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

            if (major, minor) >= (2, 3):
                pass
            elif (major, minor) >= (2, 2):
                _install_requires.pop(_install_requires.index("xformers==0.0.26.post1"))
                _install_requires.append("xformers>=0.0.25.post1")
            else:
                _install_requires.pop(_install_requires.index("xformers==0.0.26.post1"))
                _install_requires.append("xformers>=0.0.23.post1")

    except PackageNotFoundError:
        pass

    return _install_requires, _dependency_links


install_requires, dependency_links = parse_requirements()


setup(
    name="axolotl",
    version="0.4.1",
    description="LLM Trainer",
    long_description="Axolotl is a tool designed to streamline the fine-tuning of various AI models, offering support for multiple configurations and architectures.",
    package_dir={"": "src"},
    packages=find_packages(),
    install_requires=install_requires,
    dependency_links=dependency_links,
    extras_require={
        "flash-attn": [
            "flash-attn==2.5.8",
        ],
        "fused-dense-lib": [
            "fused-dense-lib  @ git+https://github.com/Dao-AILab/flash-attention@v2.5.8#subdirectory=csrc/fused_dense_lib",
        ],
        "deepspeed": [
            "deepspeed==0.14.2",
            "deepspeed-kernels",
        ],
        "mamba-ssm": [
            "mamba-ssm==1.2.0.post1",
        ],
        "auto-gptq": [
            "auto-gptq==0.5.1",
        ],
        "mlflow": [
            "mlflow",
        ],
        "lion-pytorch": [
            "lion-pytorch==0.1.2",
        ],
        "galore": [
            "galore_torch",
        ],
    },
)
