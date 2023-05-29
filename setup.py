"""setup.py for axolotl"""

from setuptools import setup, find_packages

install_requires = []
with open("./requirements.txt", encoding="utf-8") as requirements_file:
    # don't include peft yet until we check the int4
    # need to manually install peft for now...
    reqs = [r.strip() for r in requirements_file.readlines() if "peft" not in r]
    reqs = [r for r in reqs if r and r[0] != "#"]
    for r in reqs:
        install_requires.append(r)

setup(
    name="axolotl",
    version="0.1",
    description="You know you're going to axolotl questions",
    package_dir={"": "src"},
    packages=find_packages(),
    install_requires=install_requires,
    extras_require={
        "int4": [
            "alpaca_lora_4bit @ git+https://github.com/winglian/alpaca_lora_4bit.git@setup_pip",
        ],
        "int4_triton": [
            "alpaca_lora_4bit[triton] @ git+https://github.com/winglian/alpaca_lora_4bit.git@setup_pip",
        ],
        "extras": [
            "flash-attn",
            "deepspeed",
        ],
    },
)
