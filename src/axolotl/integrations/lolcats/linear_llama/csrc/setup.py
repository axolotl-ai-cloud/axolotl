#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

import subprocess  # nosec

import torch
from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension


def get_last_arch_torch():
    arch = torch.cuda.get_arch_list()[-1]
    print(f"Found arch: {arch} from existing torch installation")
    return arch


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True  # nosec
    )
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]
    return raw_output, bare_metal_major, bare_metal_minor


def append_nvcc_threads(nvcc_extra_args):
    _, bare_metal_major, bare_metal_minor = get_cuda_bare_metal_version(CUDA_HOME)
    if int(bare_metal_major) >= 11 and int(bare_metal_minor) >= 2:
        return nvcc_extra_args + ["--threads", "4"]
    return nvcc_extra_args


arch = get_last_arch_torch()
sm_num = arch[-2:]
cc_flag = ["--generate-code=arch=compute_90,code=compute_90"]  # for H100
# cc_flag = ['--generate-code=arch=compute_80,code=compute_80']  # for A100
# cc_flag = ['--generate-code=arch=compute_89,code=compute_89']  # for RTX 6000, 4090
# cc_flag = ['--generate-code=arch=compute_86,code=compute_86']  # for A6000, 3090
# cc_flag = ['--generate-code=arch=compute_75,code=compute_75']

setup(
    name="causal_attention_cuda_cpp",
    ext_modules=[
        CUDAExtension(
            "causal_attention_cuda",
            [
                # 'causal_attention.cpp',
                "causal_attention_cuda.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": append_nvcc_threads(
                    ["-O3", "-lineinfo", "--use_fast_math", "-std=c++17"] + cc_flag
                ),
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
