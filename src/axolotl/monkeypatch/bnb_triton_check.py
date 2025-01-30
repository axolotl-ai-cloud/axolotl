"""
monkey patch bnb fix from https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1497
"""
import functools


@functools.lru_cache(None)
def is_triton_available():
    try:
        # torch>=2.2.0
        from torch.utils._triton import has_triton, has_triton_package

        return has_triton_package() and has_triton()
    except ImportError:
        from torch._inductor.utils import has_triton

        return has_triton()


def patch_is_triton_available():
    from bitsandbytes.triton import triton_utils

    triton_utils.is_triton_available = is_triton_available
