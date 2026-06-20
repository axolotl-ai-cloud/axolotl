/*
 * Standalone extraction of vLLM's NVFP4 Marlin MoE GEMM (W4A16, bf16 act/out).
 * Ported from vllm csrc .../moe/marlin_moe_wna16/ops.cu:
 *   - kept the marlin_mm launcher + helpers verbatim (launcher_body.inc),
 *   - ported the moe_wna16_marlin_gemm wrapper from torch::stable::Tensor to
 *     regular torch::Tensor,
 *   - replaced STABLE_TORCH_LIBRARY_IMPL registration with a PYBIND11_MODULE.
 */

#ifndef MARLIN_NAMESPACE_NAME
  #define MARLIN_NAMESPACE_NAME marlin_moe_wna16
#endif

#include "kernel.h"

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

// STD_TORCH_CHECK comes from the header-only torch ABI; the kernel headers and
// the launcher use it. Keep it available for launcher_body.inc.
#include <torch/headeronly/util/Exception.h>

#define STATIC_ASSERT_SCALAR_TYPE_VALID(scalar_t)               \
  static_assert(std::is_same<scalar_t, half>::value ||          \
                    std::is_same<scalar_t, nv_bfloat16>::value, \
                "only float16 and bfloat16 is supported");

// ---- launcher + helpers (verbatim from upstream ops.cu lines 42..541) ----
#include "launcher_body.inc"

// -------------------------------------------------------------------------
// Ported wrapper: regular torch::Tensor signature.
// b_q_type is received as an int64 ScalarType id (vllm::ScalarTypeId) and
// reconstructed via vllm::ScalarType::from_id().
// -------------------------------------------------------------------------

torch::Tensor moe_wna16_marlin_gemm(
    torch::Tensor& a, std::optional<torch::Tensor> c_or_none,
    torch::Tensor& b_q_weight,
    std::optional<torch::Tensor> const& b_bias_or_none,
    torch::Tensor& b_scales,
    std::optional<torch::Tensor> const& a_scales_or_none,
    std::optional<torch::Tensor> const& global_scale_or_none,
    std::optional<torch::Tensor> const& b_zeros_or_none,
    std::optional<torch::Tensor> const& g_idx_or_none,
    std::optional<torch::Tensor> const& perm_or_none,
    torch::Tensor& workspace, torch::Tensor& sorted_token_ids,
    torch::Tensor& expert_ids, torch::Tensor& num_tokens_past_padded,
    torch::Tensor& topk_weights, int64_t moe_block_size, int64_t top_k,
    bool mul_topk_weights, int64_t b_type_id, int64_t size_m, int64_t size_n,
    int64_t size_k, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce,
    bool is_zp_float, int64_t thread_k, int64_t thread_n,
    int64_t blocks_per_sm) {
  vllm::ScalarTypeId a_type_id, c_type_id, s_type_id;

  auto c_dtype = a.scalar_type();
  if (a.scalar_type() == at::ScalarType::Half) {
    a_type_id = vllm::kFloat16.id();
    c_type_id = vllm::kFloat16.id();
  } else if (a.scalar_type() == at::ScalarType::BFloat16) {
    a_type_id = vllm::kBFloat16.id();
    c_type_id = vllm::kBFloat16.id();
  } else {
    c_dtype = b_scales.scalar_type();
    if (b_scales.scalar_type() == at::ScalarType::Half) {
      c_type_id = vllm::kFloat16.id();
    } else if (b_scales.scalar_type() == at::ScalarType::BFloat16) {
      c_type_id = vllm::kBFloat16.id();
    } else {
      c_type_id = vllm::kBFloat16.id();

      TORCH_CHECK(c_or_none.has_value(), "c must be passed for W4A8-FP4");
      torch::Tensor c = c_or_none.value();
      c_dtype = c.scalar_type();

      if (c.scalar_type() == at::ScalarType::Half) {
        c_type_id = vllm::kFloat16.id();
      } else if (c.scalar_type() == at::ScalarType::BFloat16) {
        c_type_id = vllm::kBFloat16.id();
      } else {
        TORCH_CHECK(false, "unsupported c dtype");
      }
    }

    if (a.scalar_type() == at::ScalarType::Float8_e4m3fn) {
      a_type_id = vllm::kFE4M3fn.id();
    } else if (a.scalar_type() == at::ScalarType::Char) {
      a_type_id = vllm::kS8.id();
    } else {
      TORCH_CHECK(false, "unsupported `a` scalar_type");
    }
  }

  s_type_id = c_type_id;
  if (b_type_id == vllm::kFE2M1f.id()) {
    if (b_scales.scalar_type() == at::ScalarType::Float8_e4m3fn) {
      s_type_id = vllm::kFE4M3fn.id();
    } else if (b_scales.scalar_type() == at::ScalarType::Float8_e8m0fnu) {
      s_type_id = vllm::kFE8M0fnu.id();
    } else {
      TORCH_CHECK(false,
                  "When b_type = float4_e2m1f, b_scale scalar type must be",
                  "float8_e4m3fn (for NVFP4) or float8_e8m0fnu (for MXFP4).");
    }
  } else if (b_type_id == vllm::kFE4M3fn.id() &&
             b_scales.scalar_type() == at::ScalarType::Float8_e8m0fnu) {
    s_type_id = vllm::kFE8M0fnu.id();
  }

  vllm::ScalarType a_type = vllm::ScalarType::from_id(a_type_id);
  vllm::ScalarType b_type = vllm::ScalarType::from_id(b_type_id);
  vllm::ScalarType c_type = vllm::ScalarType::from_id(c_type_id);
  vllm::ScalarType s_type = vllm::ScalarType::from_id(s_type_id);

  int pack_factor = 32 / b_type.size_bits();
  int num_experts = b_q_weight.size(0);

  if (moe_block_size != 8) {
    TORCH_CHECK(moe_block_size % 16 == 0,
                "unsupported moe_block_size=", moe_block_size);
    TORCH_CHECK(moe_block_size >= 16 && moe_block_size <= 64,
                "unsupported moe_block_size=", moe_block_size);
  }

  // Verify A
  TORCH_CHECK(a.size(0) == size_m, "Shape mismatch: a.size(0) = ", a.size(0),
              ", size_m = ", size_m);
  TORCH_CHECK(a.size(1) == size_k, "Shape mismatch: a.size(1) = ", a.size(1),
              ", size_k = ", size_k);

  // Verify B
  TORCH_CHECK(
      size_k % MARLIN_NAMESPACE_NAME::tile_size == 0, "size_k = ", size_k,
      " is not divisible by tile_size = ", MARLIN_NAMESPACE_NAME::tile_size);
  TORCH_CHECK(
      (size_k / MARLIN_NAMESPACE_NAME::tile_size) == b_q_weight.size(1),
      "Shape mismatch: b_q_weight.size(1) = ", b_q_weight.size(1),
      ", size_k = ", size_k,
      ", tile_size = ", MARLIN_NAMESPACE_NAME::tile_size);
  TORCH_CHECK(
      b_q_weight.size(2) % MARLIN_NAMESPACE_NAME::tile_size == 0,
      "b_q_weight.size(2) = ", b_q_weight.size(2),
      " is not divisible by tile_size = ", MARLIN_NAMESPACE_NAME::tile_size);
  int actual_size_n =
      (b_q_weight.size(2) / MARLIN_NAMESPACE_NAME::tile_size) * pack_factor;
  TORCH_CHECK(size_n == actual_size_n, "size_n = ", size_n,
              ", actual_size_n = ", actual_size_n);

  // Verify device and strides
  TORCH_CHECK(a.device().is_cuda(), "A is not on GPU");
  TORCH_CHECK(a.is_contiguous(), "A is not contiguous");

  TORCH_CHECK(b_q_weight.device().is_cuda(), "b_q_weight is not on GPU");
  TORCH_CHECK(b_q_weight.is_contiguous(), "b_q_weight is not contiguous");

  TORCH_CHECK(b_scales.device().is_cuda(), "b_scales is not on GPU");
  TORCH_CHECK(b_scales.is_contiguous(), "b_scales is not contiguous");

  torch::Tensor a_scales;
  auto opts_f32 = a.options().dtype(at::kFloat);

  if (a_scales_or_none.has_value()) {
    a_scales = a_scales_or_none.value();
    TORCH_CHECK(a_type.size_bits() == 8,
                "a_scales can only be used for 8bit activation.");
  } else {
    a_scales = torch::empty({0}, opts_f32);
    TORCH_CHECK(a_type.size_bits() != 8,
                "the a_scales parameter must be passed for 8bit activation.");
  }

  // sms: number of SMs to use for the kernel
  int sms = -1;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, a.get_device());

  // Alloc buffers
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  torch::Tensor c;
  if (c_or_none.has_value()) {
    c = c_or_none.value();
    TORCH_CHECK(c.device().is_cuda(), "c is not on GPU");
    TORCH_CHECK(c.is_contiguous(), "c is not contiguous");
    TORCH_CHECK(c.size(0) == size_m * top_k,
                "Shape mismatch: c.size(0) = ", c.size(0),
                ", size_m * topk = ", size_m * top_k);
    TORCH_CHECK(c.size(1) == size_n, "Shape mismatch: c.size(1) = ", c.size(1),
                ", size_n = ", size_n);
  } else {
    c = torch::empty({size_m * top_k, size_n}, a.options().dtype(c_dtype));
  }

  // Alloc C tmp buffer that is going to be used for the global reduce
  torch::Tensor c_tmp;
  auto options_fp32 = a.options().dtype(at::kFloat);
  if (use_fp32_reduce && !use_atomic_add) {
    // max num of threadblocks is sms * 4
    long max_c_tmp_size = min(
        (long)size_n * sorted_token_ids.size(0),
        (long)sms * 4 * moe_block_size * MARLIN_NAMESPACE_NAME::max_thread_n);
    if (moe_block_size == 8) max_c_tmp_size *= 2;
    c_tmp = torch::empty({max_c_tmp_size}, options_fp32);
  } else {
    c_tmp = torch::empty({0}, options_fp32);
  }

  // Detect groupsize and act_order
  int num_groups = -1;
  int group_size = -1;

  int rank = b_scales.dim();
  TORCH_CHECK(rank == 3, "b_scales rank = ", rank, " is not 3");
  TORCH_CHECK(b_scales.size(2) == size_n, "b_scales dim 2 = ", b_scales.size(2),
              " is not size_n = ", size_n);
  num_groups = b_scales.size(1);

  torch::Tensor g_idx, perm, a_tmp;
  auto opts_a = a.options().dtype(c_dtype);
  if (g_idx_or_none.has_value() && perm_or_none.has_value()) {
    g_idx = g_idx_or_none.value();
    perm = perm_or_none.value();

    TORCH_CHECK(g_idx.device().is_cuda(), "g_idx is not on GPU");
    TORCH_CHECK(g_idx.is_contiguous(), "g_idx is not contiguous");
    TORCH_CHECK(perm.device().is_cuda(), "perm is not on GPU");
    TORCH_CHECK(perm.is_contiguous(), "perm is not contiguous");

    // Verify g_idx and perm
    TORCH_CHECK((g_idx.size(-1) == 0 && perm.size(-1) == 0) ||
                    (g_idx.size(-1) == size_k && perm.size(-1) == size_k),
                "Unexpected g_idx.size(-1) = ", g_idx.size(-1),
                " and perm.size(-1) = ", perm.size(-1),
                ", where size_k = ", size_k);
  } else {
    g_idx = torch::empty({0}, opts_a);
    perm = torch::empty({0}, opts_a);
    a_tmp = torch::empty({0}, opts_a);
  }
  bool has_act_order = g_idx.size(-1) > 0 && perm.size(-1) > 0;

  if (has_act_order) {
    a_tmp = torch::empty({size_m * top_k, size_k}, opts_a);
    if (is_k_full) {
      TORCH_CHECK(num_groups > 1, "For act_order, num_groups must be > 1");
      TORCH_CHECK(size_k % num_groups == 0, "size_k = ", size_k,
                  ", is not divisible by num_groups = ", num_groups);
      group_size = size_k / num_groups;
    } else {
      group_size = 0;
    }

  } else {
    a_tmp = torch::empty({0}, opts_a);
    if (num_groups > 1) {
      TORCH_CHECK(
          size_k % num_groups == 0, "size_k = ", size_k,
          ", is not divisible by b_scales.size(1) = ", b_scales.size(1));
      group_size = size_k / num_groups;
    } else {
      group_size = -1;
    }
  }

  torch::Tensor global_scale;
  if (global_scale_or_none.has_value()) {
    global_scale = global_scale_or_none.value();
    TORCH_CHECK(b_type == vllm::kFE2M1f && s_type == vllm::kFE4M3fn,
                "global_scale can only be used for nvfp4 format.");
  } else {
    global_scale = torch::empty({0}, options_fp32);
    TORCH_CHECK(!(b_type == vllm::kFE2M1f && s_type == vllm::kFE4M3fn),
                "the global_scale parameter must be passed for nvfp4 format.");
  }

  bool has_bias = b_bias_or_none.has_value();
  torch::Tensor b_bias;
  if (has_bias) {
    b_bias = b_bias_or_none.value();
    TORCH_CHECK(b_bias.device().is_cuda(), "b_bias is not on GPU");
    TORCH_CHECK(b_bias.is_contiguous(), "b_bias is not contiguous");
    TORCH_CHECK(b_bias.size(1) == size_n, "b_bias.size(1) != size_n");
    TORCH_CHECK(b_bias.stride(1) == 1, "b_bias.stride(1) != 1");
  } else {
    b_bias = torch::empty({0}, opts_a);
  }

  torch::Tensor b_zeros;
  if (b_zeros_or_none.has_value()) {
    b_zeros = b_zeros_or_none.value();
    TORCH_CHECK(b_zeros.device().is_cuda(), "b_zeros is not on GPU");
    TORCH_CHECK(b_zeros.is_contiguous(), "b_zeros is not contiguous");
  } else {
    b_zeros = torch::empty({0}, opts_a);
  }
  bool has_zp = b_zeros.size(-1) > 0;
  if (has_zp) {
    TORCH_CHECK(b_type == vllm::kU4 || b_type == vllm::kU8,
                "b_type must be u4 or u8 when has_zp = True. Got = ",
                b_type.str());
  } else {
    TORCH_CHECK(b_type == vllm::kU4B8 || b_type == vllm::kU8B128 ||
                    b_type == vllm::kS4 || b_type == vllm::kS8 ||
                    b_type == vllm::kFE4M3fn || b_type == vllm::kFE2M1f,
                "b_type must be uint4b8, uint8b128, int4, int8, "
                "float8_e4m3fn or float4_e2m1f when has_zp = False. Got = ",
                b_type.str());
  }

  if (has_zp && is_zp_float) {
    TORCH_CHECK(a.scalar_type() == at::ScalarType::Half,
                "Computation type must be float16 (half) when using float zero "
                "points.");
  }

  // Verify b_zeros
  if (has_zp) {
    int rank2 = b_zeros.dim();
    TORCH_CHECK(rank2 == 3, "b_zeros rank = ", rank2, " is not 3");
    if (is_zp_float) {
      TORCH_CHECK(b_zeros.size(2) == size_n,
                  "b_zeros dim 2 = ", b_zeros.size(2),
                  " is not size_n = ", size_n);
      TORCH_CHECK(num_groups == b_zeros.size(1),
                  "b_zeros dim 1 = ", b_zeros.size(1),
                  " is not num_groups = ", num_groups);
      TORCH_CHECK(num_groups != -1, "num_groups must be != -1");
    } else {
      TORCH_CHECK(b_zeros.size(1) == num_groups,
                  "b_zeros dim 1 = ", b_zeros.size(1),
                  " is not num_groups = ", num_groups);
      TORCH_CHECK(b_zeros.size(2) == size_n / pack_factor,
                  "b_zeros dim 2 = ", b_zeros.size(2),
                  " is not size_n / pack_factor = ", size_n / pack_factor);
    }
  }

  // Verify workspace size
  TORCH_CHECK(size_n % MARLIN_NAMESPACE_NAME::min_thread_n == 0,
              "size_n = ", size_n, ", is not divisible by min_thread_n = ",
              MARLIN_NAMESPACE_NAME::min_thread_n);

  int max_n_tiles = size_n / MARLIN_NAMESPACE_NAME::min_thread_n;
  int min_workspace_size = min(
      max_n_tiles * (int)(sorted_token_ids.size(0) / moe_block_size), sms * 4);
  TORCH_CHECK(workspace.numel() >= min_workspace_size,
              "workspace.numel = ", workspace.numel(),
              " is below min_workspace_size = ", min_workspace_size);

  int dev = a.get_device();

  TORCH_CHECK(a_scales.scalar_type() == at::ScalarType::Float,
              "scalar type of a_scales must be float");
  TORCH_CHECK(global_scale.scalar_type() == at::ScalarType::Float,
              "scalar type of global_scale must be float");
  if (a_type.size_bits() == 16) {
    TORCH_CHECK(a.scalar_type() == c.scalar_type(),
                "scalar type of a must be the same with c for 16 bit "
                "activation");
  }

  MARLIN_NAMESPACE_NAME::marlin_mm(
      a.const_data_ptr(), b_q_weight.const_data_ptr(), c.mutable_data_ptr(),
      c_tmp.mutable_data_ptr(), b_bias.mutable_data_ptr(),
      a_scales.mutable_data_ptr(), b_scales.mutable_data_ptr(),
      global_scale.mutable_data_ptr(), b_zeros.mutable_data_ptr(),
      g_idx.mutable_data_ptr(), perm.mutable_data_ptr(),
      a_tmp.mutable_data_ptr(), sorted_token_ids.mutable_data_ptr(),
      expert_ids.mutable_data_ptr(), num_tokens_past_padded.mutable_data_ptr(),
      topk_weights.mutable_data_ptr(), moe_block_size, num_experts, top_k,
      mul_topk_weights, size_m, size_n, size_k, workspace.mutable_data_ptr(),
      a_type, b_type, c_type, s_type, has_bias, has_act_order, is_k_full,
      has_zp, num_groups, group_size, dev,
      at::cuda::getCurrentCUDAStream(dev).stream(), thread_k, thread_n, sms,
      blocks_per_sm, use_atomic_add, use_fp32_reduce, is_zp_float);

  return c;
}

// Defined in repack_standalone.cu (ported gptq_marlin_repack, torch::Tensor ABI).
torch::Tensor gptq_marlin_repack(torch::Tensor& b_q_weight, torch::Tensor& perm,
                                 int64_t size_k, int64_t size_n,
                                 int64_t num_bits, bool is_a_8bit);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  namespace py = pybind11;
  m.def("gptq_marlin_repack", &gptq_marlin_repack,
        "NVFP4 weight -> Marlin tile layout repack (standalone)",
        py::arg("b_q_weight"), py::arg("perm"), py::arg("size_k"),
        py::arg("size_n"), py::arg("num_bits"), py::arg("is_a_8bit") = false);
  m.def("moe_wna16_marlin_gemm", &moe_wna16_marlin_gemm,
        "NVFP4 W4A16 Marlin MoE GEMM (standalone)", py::arg("a"),
        py::arg("c_or_none"), py::arg("b_q_weight"), py::arg("b_bias_or_none"),
        py::arg("b_scales"), py::arg("a_scales_or_none"),
        py::arg("global_scale_or_none"), py::arg("b_zeros_or_none"),
        py::arg("g_idx_or_none"), py::arg("perm_or_none"), py::arg("workspace"),
        py::arg("sorted_token_ids"), py::arg("expert_ids"),
        py::arg("num_tokens_past_padded"), py::arg("topk_weights"),
        py::arg("moe_block_size"), py::arg("top_k"), py::arg("mul_topk_weights"),
        py::arg("b_type_id"), py::arg("size_m"), py::arg("size_n"),
        py::arg("size_k"), py::arg("is_k_full"), py::arg("use_atomic_add"),
        py::arg("use_fp32_reduce"), py::arg("is_zp_float"),
        py::arg("thread_k") = -1, py::arg("thread_n") = -1,
        py::arg("blocks_per_sm") = -1);
}
