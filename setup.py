import sys
from setuptools import setup, find_packages

install_requires = []
with open("./requirements.txt", "r") as requirements_file:
    # don't include peft yet until we check the int4
    reqs = [r.strip() for r in requirements_file.readlines() if "peft" not in r]
    reqs = [r for r in reqs if r[0] != "#"]
    for r in reqs:
        install_requires.append(r)

setup(
    name='axolotl',
    version='0.1',
    description="You know you're going to axolotl questions",
    package_dir={'': 'src'},
    packages=find_packages(),
    install_requires=install_requires,
    extras_require={
        None: [
            "peft @ git+https://github.com/huggingface/peft.git",
        ],
        'int4_cuda': [
            "alpaca_lora_4bit[cuda] @ git+https://github.com/winglian/alpaca_lora_4bit.git@setup_pip#egg=alpaca_lora_4bit[cuda]",
        ],
        'int4_triton': [
            "alpaca_lora_4bit[triton] @ git+https://github.com/winglian/alpaca_lora_4bit.git@setup_pip#egg=alpaca_lora_4bit[triton]",
        ],
    },
)
