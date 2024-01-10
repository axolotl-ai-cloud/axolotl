"""setup.py for axolotl"""

from importlib.metadata import PackageNotFoundError, version

from setuptools import find_packages, setup


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
            )
            if line.startswith("--extra-index-url"):
                # Handle custom index URLs
                _, url = line.split()
                _dependency_links.append(url)
            elif not is_extras and line and line[0] != "#":
                # Handle standard packages
                _install_requires.append(line)

    try:
        torch_version = version("torch")
        if torch_version.startswith("2.1.1"):
            _install_requires.pop(_install_requires.index("xformers==0.0.22"))
            _install_requires.append("xformers==0.0.23")
    except PackageNotFoundError:
        pass

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
            "flash-attn==2.3.3",
        ],
        "fused-dense-lib": [
            "fused-dense-lib  @ git+https://github.com/Dao-AILab/flash-attention@v2.3.3#subdirectory=csrc/fused_dense_lib",
        ],
        "deepspeed": [
            "deepspeed",
        ],
        "mamba-ssm": [
            "mamba-ssm==1.0.1",
        ],
        "auto-gptq": [
            "auto-gptq==0.5.1",
        ],
    },
)
