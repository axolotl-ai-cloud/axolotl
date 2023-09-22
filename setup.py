"""setup.py for axolotl"""

from setuptools import find_packages, setup


def parse_requirements():
    _install_requires = []
    _dependency_links = []
    with open("./requirements.txt", encoding="utf-8") as requirements_file:
        lines = [r.strip() for r in requirements_file.readlines()]
        for line in lines:
            if line.startswith("--extra-index-url"):
                # Handle custom index URLs
                _, url = line.split()
                _dependency_links.append(url)
            elif (
                "flash-attn" not in line
                and "deepspeed" not in line
                and line
                and line[0] != "#"
            ):
                # Handle standard packages
                _install_requires.append(line)
    return _install_requires, _dependency_links


install_requires, dependency_links = parse_requirements()


setup(
    name="axolotl",
    version="0.3.0",
    description="LLM Trainer",
    long_description="Axolotl is a tool designed to streamline the fine-tuning of various AI models, offering support for multiple configurations and architectures.",
    package_dir={"": "src"},
    packages=find_packages(),
    install_requires=install_requires,
    dependency_links=dependency_links,
    extras_require={
        "flash-attn": [
            "flash-attn>=2.2.1",
        ],
        "deepspeed": [
            "deepspeed",
        ],
    },
)
