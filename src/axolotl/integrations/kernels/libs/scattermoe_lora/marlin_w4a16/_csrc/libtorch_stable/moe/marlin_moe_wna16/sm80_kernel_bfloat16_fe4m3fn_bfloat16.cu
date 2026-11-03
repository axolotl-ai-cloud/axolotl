// fp8 (block-fp8 W8A16) instantiations: b = fe4m3fn weights, bf16 group-128 scales
// (group_blocks = 8). Mirrors the fe2m1f file's config sweep; combos match upstream
// generate_kernels.py's FP8 entry (group_blocks in {-1, 8} — we emit 8, the block-fp8 case).
// clang-format off

#include "kernel.h"
#include "marlin_template.h"

namespace MARLIN_NAMESPACE_NAME {


template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 1, 8, 8, true, 4, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 1, 8, 4, true, 4, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 1, 4, 8, true, 4, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 1, 8, 8, false, 4, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 1, 8, 4, false, 4, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 1, 4, 8, false, 4, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 2, 16, 4, false, 4, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 2, 8, 4, false, 4, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 2, 4, 8, false, 4, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 3, 16, 4, false, 4, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 3, 8, 4, false, 4, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 3, 4, 8, false, 4, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 256, 4, 16, 4, false, 4, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 4, 8, 4, false, 4, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), 128, 4, 4, 8, false, 4, 8, false>( MARLIN_KERNEL_PARAMS );

}
