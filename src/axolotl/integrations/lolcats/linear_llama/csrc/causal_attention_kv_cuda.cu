//
// Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
// Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
// Apoorv Vyas <avyas@idiap.ch>
//

//
// For modifications made inside namespace nvidia (authored by jdemouth):
//
// Copyright (c) 2021 NVIDIA CORPORATION. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <torch/extension.h>
#include <assert.h>
#include <stdio.h>

#define ENABLE_NVIDIA_OPTIMIZATIONS

#ifdef ENABLE_NVIDIA_OPTIMIZATIONS
namespace nvidia {

////////////////////////////////////////////////////////////////////////////////////////////////////

constexpr int THREADS_PER_WARP = 32;

////////////////////////////////////////////////////////////////////////////////////////////////////

constexpr int LOW_OCCUPANCY_THRESHOLD = 40; // TODO: Make it HW specific (like 1/2 SMs).

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ __host__ int div_up(int m, int n) {
  return (m + n-1) / n;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ __host__ int round_up(int m, int n) {
  return div_up(m, n) * n;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename T >
struct Lmha_params {

  // The output buffer. Dimensions [B, H, L, M].
  T *out;

  // The input Qs. Dimensions [B, H, L, E].
  const T *q;
  // The input Ks. Dimensions [B, H, L, E].
  const T *k;
  // The input Vs. Dimensions [B, H, L, M].
  const T *v;

  // The different dimensions.
  int B, L, H, E, M;

  // The strides for the different tensors.
  int q_stride_B, q_stride_H, q_stride_L;
  int k_stride_B, k_stride_H, k_stride_L;
  int v_stride_B, v_stride_H, v_stride_L;
  int o_stride_B, o_stride_H, o_stride_L;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int E, bool GO_BACKWARD, int WARPS, int COLS_PER_THREAD = 4 >
__global__ __launch_bounds__(WARPS * THREADS_PER_WARP)
void lmha_low_occupancy_kernel(Lmha_params<float> params) {

  // The number of threads per block.
  constexpr int THREADS_PER_BLOCK = WARPS * THREADS_PER_WARP;
  // The number of rows per thread.
  constexpr int ROWS_PER_THREAD = E / THREADS_PER_WARP;
  // The number of steps per iteration.
  constexpr int COLS_PER_ITER = WARPS * COLS_PER_THREAD;

  // Make sure E is a multiple of the warp size.
  static_assert(E % THREADS_PER_WARP == 0, "");

  // Shared memory to store V/O.
  __shared__ float smem_v[COLS_PER_ITER], smem_o[COLS_PER_ITER];
  // Shared memory buffer to performance the reductions.
  __shared__ float smem_reds[E * WARPS];

  // The sequence processed by that block.
  const int bi = blockIdx.z;
  // The head processed by that block.
  const int hi = blockIdx.y;
  // The hidden cell in the V/output buffers.
  const int vi = blockIdx.x;

  // The linear index of the thread.
  const int tidx = threadIdx.x;

  // Decompose the block in warp/lane.
  const int warp = tidx / THREADS_PER_WARP;
  const int lane = tidx % THREADS_PER_WARP;

  // The base offset loaded by the thread in Q and K.
  int offset_q = bi*params.q_stride_B + hi*params.q_stride_H + lane;
  int offset_k = bi*params.k_stride_B + hi*params.k_stride_H + lane;

  // If we walk backward, account for the extra offset.
  if( GO_BACKWARD ) {
    offset_q += (params.L-1)*params.q_stride_L;
    offset_k += (params.L-1)*params.k_stride_L;
  }

  // Position the warp at the beginning of the proper timestep.
  if( GO_BACKWARD ) {
    offset_q -= warp*COLS_PER_THREAD*params.q_stride_L;
    offset_k -= warp*COLS_PER_THREAD*params.k_stride_L;
  } else {
    offset_q += warp*COLS_PER_THREAD*params.q_stride_L;
    offset_k += warp*COLS_PER_THREAD*params.k_stride_L;
  }

  // Determine the base pointers for Q and K.
  const float *ptr_q = &params.q[offset_q];
  const float *ptr_k = &params.k[offset_k];

  // Is a given row valid?
  int valid_qk[ROWS_PER_THREAD];
  #pragma unroll
  for( int ii = 0; ii < ROWS_PER_THREAD; ++ii ) {
    valid_qk[ii] = lane + ii*THREADS_PER_WARP < params.E;
  }

  // The offset to the position loaded by the thread in V.
  int offset_v = bi*params.v_stride_B + hi*params.v_stride_H + vi;
  int offset_o = bi*params.o_stride_B + hi*params.o_stride_H + vi;

  // If we walk backward, account for the extra offset.
  if( GO_BACKWARD ) {
    offset_v += (params.L-1)*params.v_stride_L;
    offset_o += (params.L-1)*params.o_stride_L;
  }

  // We load/store a strided matrix of COLS_PER_ITER x OUTPUTS_PER_BLOCK.
  if( GO_BACKWARD ) {
    offset_v -= tidx*params.v_stride_L;
    offset_o -= tidx*params.o_stride_L;
  } else {
    offset_v += tidx*params.v_stride_L;
    offset_o += tidx*params.o_stride_L;
  }

  // Determine the base pointer for V.
  const float *ptr_v = &params.v[offset_v];
  // The output pointer.
  float *ptr_o = &params.out[offset_o];

  // The running KVs.
  float running_kv[ROWS_PER_THREAD];
  #pragma unroll
  for( int ri = 0; ri < ROWS_PER_THREAD; ++ri ) {
    running_kv[ri] = 0.f;
  }

  // Iterate over the timesteps. TODO: Use params.loop_count!!!
  for( int iter = 0; iter < params.L; iter += COLS_PER_ITER ) {

    // Each thread loads a matrix of elements.
    float q[ROWS_PER_THREAD][COLS_PER_THREAD], k[ROWS_PER_THREAD][COLS_PER_THREAD];

    // Trigger the memory loads for Q and K.
    #pragma unroll
    for( int ci = 0; ci < COLS_PER_THREAD; ++ci ) {
      #pragma unroll
      for( int ri = 0; ri < ROWS_PER_THREAD; ++ri ) {

        // For Q/K, each warp loads from various timesteps.
        int ti = iter + warp*COLS_PER_THREAD;
        if( GO_BACKWARD ) {
          ti = params.L - 1 - ti;
        }

        // Is it a valid access?
        int valid;
        if( GO_BACKWARD ) {
          valid = valid_qk[ri] && ti - ci >= 0;
        } else {
          valid = valid_qk[ri] && ti + ci < params.L;
        }

        // The extra offset to add.
        if( GO_BACKWARD ) {
          offset_q = ri*THREADS_PER_WARP - ci*params.q_stride_L;
          offset_k = ri*THREADS_PER_WARP - ci*params.k_stride_L;
        } else {
          offset_q = ri*THREADS_PER_WARP + ci*params.q_stride_L;
          offset_k = ri*THREADS_PER_WARP + ci*params.k_stride_L;
        }

        // Load Q/K if they are valid.
        q[ri][ci] = valid ? ptr_q[offset_q] : 0.f;
        k[ri][ci] = valid ? ptr_k[offset_k] : 0.f;
      }
    }

    // For the V tensor, we assign contiguous thread to different loads. So, ti is different.
    int ti = iter + tidx;
    if( GO_BACKWARD ) {
      ti = params.L - 1 - ti;
    }

    // Is it a valid access?
    int valid_vo = tidx < COLS_PER_ITER;
    if( GO_BACKWARD ) {
      valid_vo &= ti >= 0;
    } else {
      valid_vo &= ti < params.L;
    }

    // Trigger the loads for V.
    float ldg_v = valid_vo ? *ptr_v : 0.f;

    // Move the load pointers.
    if( GO_BACKWARD ) {
      ptr_q -= COLS_PER_ITER*params.q_stride_L;
      ptr_k -= COLS_PER_ITER*params.k_stride_L;
      ptr_v -= COLS_PER_ITER*params.v_stride_L;
    } else {
      ptr_q += COLS_PER_ITER*params.q_stride_L;
      ptr_k += COLS_PER_ITER*params.k_stride_L;
      ptr_v += COLS_PER_ITER*params.v_stride_L;
    }

    // Store to shared memory.
    if( tidx < COLS_PER_ITER ) {
      smem_v[tidx] = ldg_v;
    }

    // Make sure V is in shared memory.
    __syncthreads();

    // Read V from shared memory.
    float v[COLS_PER_THREAD];
    #pragma unroll
    for( int ci = 0; ci < COLS_PER_THREAD; ++ci ) {
      v[ci] = smem_v[warp*COLS_PER_THREAD + ci];
    }

    // Each thread computes local K*V products.
    float kv[ROWS_PER_THREAD][COLS_PER_THREAD];
    #pragma unroll
    for( int ri = 0; ri < ROWS_PER_THREAD; ++ri ) {
      #pragma unroll
      for( int ci = 0; ci < COLS_PER_THREAD; ++ci ) {
        kv[ri][ci] = 0.f;
      }
    }

    // Update the K*V^T product.
    #pragma unroll
    for( int ci = 0; ci < COLS_PER_THREAD; ++ci ) {
      #pragma unroll
      for( int ri = 0; ri < ROWS_PER_THREAD; ++ri ) {
        kv[ri][ci] += k[ri][ci] * v[ci];
      }
    }

    // We must perform the prefix sums within the thread-block. Start with the thread.
    #pragma unroll
    for( int ri = 0; ri < ROWS_PER_THREAD; ++ri ) {
      #pragma unroll
      for( int ci = 1; ci < COLS_PER_THREAD; ++ci ) {
        kv[ri][ci] += kv[ri][ci-1];
      }
    }

    // Store the partial sums to shared memory. Unless we have no inter-warp reduction to perform.
    #pragma unroll
    for( int ri = 0; ri < ROWS_PER_THREAD; ++ri ) {
      smem_reds[warp*E + ri*THREADS_PER_WARP + lane] = kv[ri][COLS_PER_THREAD-1];
    }

    // Make sure the data is in shared memory.
    __syncthreads();

    // Each thread deals with one or more column(s) of the matrix.
    constexpr int SUMS_PER_THREAD = (E + THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;
    #pragma unroll
    for( int ii = 0, idx = tidx; ii < SUMS_PER_THREAD; ++ii, idx += THREADS_PER_BLOCK ) {
      if( idx < E ) {
        float sum = smem_reds[idx];
        #pragma unroll
        for( int jj = 1; jj < WARPS; ++jj ) {
          smem_reds[idx + jj*E] = sum += smem_reds[idx + jj*E];
        }
      }
    }

    // Make sure the reductions are stored in shared memory.
    __syncthreads();

    // Each thread updates his partial products.
    #pragma unroll
    for( int ri = 0; ri < ROWS_PER_THREAD; ++ri ) {
      float sum = running_kv[ri];
      if( warp > 0 ) {
        sum += smem_reds[(warp-1)*E + lane + ri*THREADS_PER_WARP];
      }
      #pragma unroll
      for( int ci = 0; ci < COLS_PER_THREAD; ++ci ) {
        kv[ri][ci] += sum;
      }
    }

    // Compute the partial output values for that thread.
    float sum[COLS_PER_THREAD];
    #pragma unroll
    for( int ci = 0; ci < COLS_PER_THREAD; ++ci ) {
      sum[ci] = q[0][ci] * kv[0][ci];
      #pragma unroll
      for( int ri = 1; ri < ROWS_PER_THREAD; ++ri ) {
        sum[ci] += q[ri][ci] * kv[ri][ci];
      }
    }

    // Run the parallel reductions inside the warp.
    #pragma unroll
    for( int mask = THREADS_PER_WARP / 2; mask >= 1; mask /= 2 ) {
      #pragma unroll
      for( int ci = 0; ci < COLS_PER_THREAD; ++ci ) {
        sum[ci] += __shfl_xor_sync(uint32_t(-1), sum[ci], mask);
      }
    }

    // Store the final output to shared memory.
    if( lane == 0 ) {
      #pragma unroll
      for( int ci = 0; ci < COLS_PER_THREAD; ++ci ) {
        smem_o[warp*COLS_PER_THREAD + ci] = sum[ci];
      }
    }

    // Make sure the data is in shared memory.
    __syncthreads();

    // Store the output.
    if( valid_vo ) {
      *ptr_o = smem_o[tidx];
    }

    // Each thread updates his running kv.
    #pragma unroll
    for( int ri = 0; ri < ROWS_PER_THREAD; ++ri ) {
      running_kv[ri] += smem_reds[(WARPS-1)*E + lane + ri*THREADS_PER_WARP];
    }

    // Move to next location.
    if( GO_BACKWARD ) {
      ptr_o -= COLS_PER_ITER*params.o_stride_L;
    } else {
      ptr_o += COLS_PER_ITER*params.o_stride_L;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int E, bool GO_BACKWARD, int WARPS >
int lmha_low_occupancy_(const Lmha_params<float> &params) {

  // Make sure we are not going to launch an invalid grid.
  if( params.H > 65535 || params.B > 65535 ) {
    return 1;
  }

  // Prepare the grid and trigger the CUDA kernel.
  dim3 grid;
  grid.x = params.M;
  grid.y = params.H;
  grid.z = params.B;
  lmha_low_occupancy_kernel<E, GO_BACKWARD, WARPS><<<grid, WARPS*THREADS_PER_WARP>>>(params);
  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int E, bool GO_BACKWARD >
int lmha_low_occupancy_(const Lmha_params<float> &params, int blocks) {
         if( params.M * blocks >= 8*LOW_OCCUPANCY_THRESHOLD ) {
    return lmha_low_occupancy_<E, GO_BACKWARD,  4>(params);
  } else if( params.M * blocks >= 4*LOW_OCCUPANCY_THRESHOLD ) {
    return lmha_low_occupancy_<E, GO_BACKWARD,  8>(params);
  } else {
    return lmha_low_occupancy_<E, GO_BACKWARD, 16>(params);
  }
  return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int E, typename Params >
static inline __device__ __host__ int smem_buffer_elts_(const Params &params) {
  int M = round_up(params.M, 4);
  return 2*E + 2*M;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int E, int THREADS_PER_HEAD, bool GO_BACKWARD >
__global__
void lmha_kernel(Lmha_params<float> params) {

  // Make sure E is a multiple of 4.
  static_assert(E % 4 == 0, "");

  // The amount of shared memory per buffer (2 buffers for double-buffering).
  const int smem_buffer_elts = smem_buffer_elts_<E>(params);
  // The M dimension for shared memory.
  const int M = round_up(params.M, 4);

  // Shared memory to store Q, K and V. Size is 2*smem_buffer_elts.
  extern __shared__ float smem_[];

  // The various shared memory buffers.
  float *smem_q = &smem_[0*E];
  float *smem_k = &smem_[1*E];
  float *smem_v = &smem_[2*E];
  float *smem_o = &smem_[2*E + M];

  // The index of the shared memory buffer (for double-buffering).
  int smem_curr = 0;

  // The sequence processed by that block.
  const int bi = blockIdx.y;
  // The head processed by that block.
  const int hi = blockIdx.x;

  // The linear index of the thread.
  const int tidx = threadIdx.x;

  // The offset to the position loaded by the thread in Q.
  int offset_q = bi*params.q_stride_B + hi*params.q_stride_H + tidx;
  // The offset to the position loaded by the thread in K.
  int offset_k = bi*params.k_stride_B + hi*params.k_stride_H + tidx;

  // If we walk backward, account for the extra offset.
  if( GO_BACKWARD ) {
    offset_q += (params.L-1)*params.q_stride_L;
    offset_k += (params.L-1)*params.k_stride_L;
  }

  // Determine the base pointers for Q and K.
  const float *ptr_q = &params.q[offset_q];
  const float *ptr_k = &params.k[offset_k];

  // The offset to the position loaded by the thread in V and O.
  int offset_v = bi*params.v_stride_B + hi*params.v_stride_H + tidx;
  int offset_o = bi*params.o_stride_B + hi*params.o_stride_H + tidx;

  // If we walk backward, account for the extra offset.
  if( GO_BACKWARD ) {
    offset_v += (params.L-1)*params.v_stride_L;
    offset_o += (params.L-1)*params.o_stride_L;
  }

  // Determine the base pointers for V.
  const float *ptr_v = &params.v[offset_v];

  // Is it an active Q/K thread?
  const int active_qk = tidx < params.E;

  // Trigger the memory loads for Q and K.
  float ldg_q = 0.f, ldg_k = 0.f;
  if( active_qk ) {
    ldg_q = *ptr_q;
    ldg_k = *ptr_k;
  }

  // Is it an active V thread?
  const int active_v = tidx < params.M;

  // Trigger the memory loads for V.
  float ldg_v = 0.f;
  if( active_v ) {
    ldg_v = *ptr_v;
  }

  // Move the load pointers.
  if( GO_BACKWARD ) {
    ptr_q -= params.q_stride_L;
    ptr_k -= params.k_stride_L;
    ptr_v -= params.v_stride_L;
  } else {
    ptr_q += params.q_stride_L;
    ptr_k += params.k_stride_L;
    ptr_v += params.v_stride_L;
  }

  // The number of FLOAT4s per head.
  constexpr int FLOAT4s_PER_HEAD = E / 4;
  // The number of FLOAT4s per thread.
  constexpr int FLOAT4s_PER_THREAD = FLOAT4s_PER_HEAD / THREADS_PER_HEAD;

  // The storage for the K*V^T values.
  float4 kv[FLOAT4s_PER_THREAD];
  #pragma unroll
  for( int ii = 0; ii < FLOAT4s_PER_THREAD; ++ii ) {
    kv[ii] = make_float4(0.f, 0.f, 0.f, 0.f);
  }

  // The output pointer.
  float *out_ptr = &params.out[offset_o];

  // Store to shared memory Q and K.
  if( tidx < E ) {
    smem_q[smem_curr*smem_buffer_elts + tidx] = ldg_q;
    smem_k[smem_curr*smem_buffer_elts + tidx] = ldg_k;
  }

  // Store to shared memory V. All threads store valid values.
  if( tidx < M ) {
    smem_v[smem_curr*smem_buffer_elts + tidx] = ldg_v;
  }

  // The position of the thread in the V dimension.
  int vo = tidx / THREADS_PER_HEAD;
  int vi = tidx % THREADS_PER_HEAD;

  // Iterate over the timesteps.
  for( int ti = 0; ti < params.L; ++ti ) {

    // Is it the last iteration?
    int is_last = ti == params.L - 1;

    // Trigger the next loads for Q and K.
    if( !is_last && active_qk ) {
      ldg_q = *ptr_q;
      ldg_k = *ptr_k;
    }

    // Trigger the next loads for V.
    if( !is_last && active_v ) {
      ldg_v = *ptr_v;
    }

    // Move the load pointers.
    if( GO_BACKWARD ) {
      ptr_q -= params.q_stride_L;
      ptr_k -= params.k_stride_L;
      ptr_v -= params.v_stride_L;
    } else {
      ptr_q += params.q_stride_L;
      ptr_k += params.k_stride_L;
      ptr_v += params.v_stride_L;
    }

    // Make sure the data is in shared memory.
    __syncthreads();

    // Each thread loads 4 values from K.
    float4 k[FLOAT4s_PER_THREAD];
    #pragma unroll
    for( int ii = 0; ii < FLOAT4s_PER_THREAD; ++ii ) {
      int ki = tidx % THREADS_PER_HEAD * 4 + ii * THREADS_PER_HEAD * 4;
      k[ii] = *reinterpret_cast<const float4*>(&smem_k[smem_curr*smem_buffer_elts + ki]);
    }

    // Each thread loads a single V value.
    float v = 0.f;
    if( vo < params.M ) {
      v = *reinterpret_cast<const float *>(&smem_v[smem_curr*smem_buffer_elts + vo]);
    }

    // Update the K*V^T product.
    #pragma unroll
    for( int ii = 0; ii < FLOAT4s_PER_THREAD; ++ii ) {
      kv[ii].x += k[ii].x * v;
      kv[ii].y += k[ii].y * v;
      kv[ii].z += k[ii].z * v;
      kv[ii].w += k[ii].w * v;
    }

    // Load the Q values from shared memory.
    float4 q[FLOAT4s_PER_THREAD];
    #pragma unroll
    for( int ii = 0; ii < FLOAT4s_PER_THREAD; ++ii ) {
      int qi = tidx % THREADS_PER_HEAD * 4 + ii * THREADS_PER_HEAD * 4;
      q[ii] = *reinterpret_cast<const float4*>(&smem_q[smem_curr*smem_buffer_elts + qi]);
    }

    // Compute the partial output value for that thread.
    float sum = 0.f;
    #pragma unroll
    for( int ii = 0; ii < FLOAT4s_PER_THREAD; ++ii ) {
      sum += q[ii].x * kv[ii].x;
      sum += q[ii].y * kv[ii].y;
      sum += q[ii].z * kv[ii].z;
      sum += q[ii].w * kv[ii].w;
    }

    // Finalize the computation of the sum (if we have more than 1 thread per head).
    if( THREADS_PER_HEAD > 1 ) {

      // Finalize the sum for each head.
      #pragma unroll
      for( int mask = THREADS_PER_HEAD / 2; mask >= 1; mask /= 2 ) {
        sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
      }

      // Store to shared memory.
      if( vo < M && vi == 0 ) {
        smem_o[smem_curr*smem_buffer_elts + vo] = sum;
      }

      // Make sure the data is in shared memory.
      __syncthreads();

      // Active threads read the data to store.
      if( active_v ) {
        sum = smem_o[smem_curr*smem_buffer_elts + tidx];
      }

    } // THREADS_PER_HEAD > 1.

    // Store the output. All the threads are active.
    if( active_v ) {
      *out_ptr = sum;
    }

    // Move to next location.
    if( GO_BACKWARD ) {
      out_ptr -= params.o_stride_L;
    } else {
      out_ptr += params.o_stride_L;
    }

    // Move the shared memory buffer.
    smem_curr = (smem_curr + 1) % 2;

    // Store to shared memory for Q and K.
    if( !is_last && tidx < E ) {
      smem_q[smem_curr*smem_buffer_elts + tidx] = ldg_q;
      smem_k[smem_curr*smem_buffer_elts + tidx] = ldg_k;
    }

    // Store to shared memory for V.
    if( !is_last && tidx < M ) {
      smem_v[smem_curr*smem_buffer_elts + tidx] = ldg_v;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int E, int THREADS_PER_HEAD, bool GO_BACKWARD >
int lmha_(const Lmha_params<float> &params) {
  // The M dimension rounded up to 4.
  int M = round_up(params.M, 4);

  // The number of threads in the block.
  int block = round_up(max(E, M*THREADS_PER_HEAD), 32);
  if( block > 512 || params.B > 65535 ) {
    return 1;
  }

  // Prepare the kernel.
  dim3 grid(params.H, params.B);
  size_t smem = smem_buffer_elts_<E>(params)*2*sizeof(float);
  lmha_kernel<E, THREADS_PER_HEAD, GO_BACKWARD><<<grid, block, smem>>>(params);
  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< bool GO_BACKWARD >
int lmha(const Lmha_params<float> &params) {
  int blocks = params.B * params.H;
  int res = 1;
  if( blocks < LOW_OCCUPANCY_THRESHOLD ) {
           if( params.E <=  32 ) {
      res = lmha_low_occupancy_< 32, GO_BACKWARD>(params, blocks);
    } else if( params.E <=  64 ) {
      res = lmha_low_occupancy_< 64, GO_BACKWARD>(params, blocks);
    } else if( params.E <= 128 ) {
      res = lmha_low_occupancy_<128, GO_BACKWARD>(params, blocks);
    } else if( params.E <= 256 ) {
      res = lmha_low_occupancy_<256, GO_BACKWARD>(params, blocks);
    }
  } else {
           if( params.E <=  32 ) {
      res = lmha_< 32, 1, GO_BACKWARD>(params);
    } else if( params.E <=  48 ) {
      res = lmha_< 48, 1, GO_BACKWARD>(params);
    } else if( params.E <=  64 ) {
      res = lmha_< 64, 1, GO_BACKWARD>(params);
    } else if( params.E <= 128 ) {
      res = lmha_<128, 2, GO_BACKWARD>(params);
    } else if( params.E <= 256 ) {
      res = lmha_<256, 4, GO_BACKWARD>(params);
    }
  }
  return res;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename T >
inline void set_params(Lmha_params<T> &params,
                       const torch::Tensor q,
                       const torch::Tensor k,
                       const torch::Tensor v,
                       torch::Tensor       o) {

  // Define the pointers.
  params.out = o.data_ptr<T>();
  params.q   = q.data_ptr<T>();
  params.k   = k.data_ptr<T>();
  params.v   = v.data_ptr<T>();

  // Define the strides.
  params.q_stride_B = (int) q.stride(0);
  params.q_stride_H = (int) q.stride(1);
  params.q_stride_L = (int) q.stride(2);
  params.k_stride_B = (int) k.stride(0);
  params.k_stride_H = (int) k.stride(1);
  params.k_stride_L = (int) k.stride(2);
  params.v_stride_B = (int) v.stride(0);
  params.v_stride_H = (int) v.stride(1);
  params.v_stride_L = (int) v.stride(2);
  params.o_stride_B = (int) o.stride(0);
  params.o_stride_H = (int) o.stride(1);
  params.o_stride_L = (int) o.stride(2);

  // Extract the dimensions.
  int N = q.size(0);
  int H = q.size(1);
  int L = q.size(2);
  int E = q.size(3);
  int M = v.size(3);

  params.B = N;
  params.L = L;
  params.H  = H;
  params.E = E;
  params.M = M;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int lmha_fwd(const torch::Tensor queries,
             const torch::Tensor keys,
             const torch::Tensor values,
             torch::Tensor product) {

  // Make sure that we are using the correct GPU device
  torch::DeviceGuard _guard(queries.device());

  // Make sure the inner-most dimension of the tensors is packed.
  assert(queries.stride(3) == 1);
  assert(keys   .stride(3) == 1);
  assert(values .stride(3) == 1);
  assert(product.stride(3) == 1);

  // Extract the dimensions.
  int N = queries.size(0);
  int H = queries.size(1);
  int L = queries.size(2);
  int E = queries.size(3);
  int M = values.size (3);

  // The structure of params.
  Lmha_params<float> params;
  set_params(params, queries, keys, values, product);

  // Launch the kernel.
  return lmha<false>(params);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename T >
struct Lmha_bwd_params {

  // The output buffer for K. Dimensions [B, H, L, D].
  T *out_k;
  // The output buffer for V. Dimensions [B, H, L, D].
  T *out_v;

  // The input Qs. Dimensions [B, H, L, D].
  const T *q;
  // The input Ks. Dimensions [B, H, L, D].
  const T *k;
  // The input Vs. Dimensions [B, H, L, D].
  const T *v;
  // The input Gs. Dimensions [B, H, L, D].
  const T *g;

  // The dimensions.
  int B, L, H, M, E;

  // The strides for the input tensors.
  int q_stride_B, q_stride_L, q_stride_H;
  int k_stride_B, k_stride_L, k_stride_H;
  int v_stride_B, v_stride_L, v_stride_H;
  int g_stride_B, g_stride_L, g_stride_H;

  // The strides for the outputs.
  int out_k_stride_B, out_k_stride_L, out_k_stride_H;
  int out_v_stride_B, out_v_stride_L, out_v_stride_H;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int D, int THREADS_PER_HEAD >
__global__ __launch_bounds__(D*THREADS_PER_HEAD*2)
void lmha_bwd_kernel(Lmha_bwd_params<float> params) {

  // Make sure D is a multiple of 4.
  static_assert(D % 4 == 0, "");

  // The shared memory buffers.
  __shared__ struct Smem { float qg[2*D], kv[2*D], out_kv[2*D]; } smem_[2];

  // The index of the shared memory buffer (for double-buffering).
  int smem_curr = 0;

  // The sequence processed by that block.
  const int bi = blockIdx.y;
  // The head processed by that block.
  const int hi = blockIdx.x;

  // The linear index of the thread.
  const int tidx = threadIdx.x;

  // Split the threads into two slices.
  int so = tidx / (D*THREADS_PER_HEAD);
  int si = tidx % (D*THREADS_PER_HEAD);

  // The strides for B/L/H for the Q/G tensors.
  int qg_stride_B, qg_stride_L, qg_stride_H;
  if( so == 0 ) {
    qg_stride_B = params.q_stride_B;
    qg_stride_L = params.q_stride_L;
    qg_stride_H = params.q_stride_H;
  } else {
    qg_stride_B = params.g_stride_B;
    qg_stride_L = params.g_stride_L;
    qg_stride_H = params.g_stride_H;
  }

  // The strides for B/L/H for the K/V tensors.
  int kv_stride_B, kv_stride_L, kv_stride_H;
  if( so == 0 ) {
    kv_stride_B = params.k_stride_B;
    kv_stride_L = params.k_stride_L;
    kv_stride_H = params.k_stride_H;
  } else {
    kv_stride_B = params.v_stride_B;
    kv_stride_L = params.v_stride_L;
    kv_stride_H = params.v_stride_H;
  }

  // The hidden size.
  int hidden_size_per_head = 0;
  if( so == 0 ) {
    hidden_size_per_head = params.E;
  } else {
    hidden_size_per_head = params.M;
  }

  // Where to start reading from.
  int offset_qg = bi*qg_stride_B + hi*qg_stride_H + si;
  int offset_kv = bi*kv_stride_B + hi*kv_stride_H + si;

  // We walk backward, account for the extra offset.
  offset_qg += (params.L-1)*qg_stride_L;
  offset_kv += (params.L-1)*kv_stride_L;

  // Determine the base pointers for Q, K, V and G.
  const float *ptr_qg = &(so == 0 ? params.q : params.g)[offset_qg];
  const float *ptr_kv = &(so == 0 ? params.k : params.v)[offset_kv];

  // Is it an active thread?
  const int active = si < hidden_size_per_head;

  // Trigger the memory loads for Q, K, V and G.
  float ldg_qg = 0.f, ldg_kv = 0.f;
  if( active ) {
    ldg_qg = *ptr_qg;
    ldg_kv = *ptr_kv;
  }

  // Move the load pointers (backward).
  ptr_qg -= qg_stride_L;
  ptr_kv -= kv_stride_L;

  // The number of FLOAT4s per head.
  constexpr int FLOAT4s_PER_HEAD = D / 4;
  // The number of FLOAT4s per thread.
  constexpr int FLOAT4s_PER_THREAD = FLOAT4s_PER_HEAD / THREADS_PER_HEAD;

  // The storage for the G*Q^T or Q^T*G values.
  float4 gq[FLOAT4s_PER_THREAD];
  #pragma unroll
  for( int ii = 0; ii < FLOAT4s_PER_THREAD; ++ii ) {
    gq[ii] = make_float4(0.f, 0.f, 0.f, 0.f);
  }

  // The strides for B/L/H for the K/V tensors.
  int out_kv_stride_B, out_kv_stride_L, out_kv_stride_H;
  if( so == 0 ) {
    out_kv_stride_B = params.out_k_stride_B;
    out_kv_stride_L = params.out_k_stride_L;
    out_kv_stride_H = params.out_k_stride_H;
  } else {
    out_kv_stride_B = params.out_v_stride_B;
    out_kv_stride_L = params.out_v_stride_L;
    out_kv_stride_H = params.out_v_stride_H;
  }

  // Where to start reading from.
  int offset_out_kv = bi*out_kv_stride_B + hi*out_kv_stride_H + si;

  // We walk backward, account for the extra offset.
  offset_out_kv += (params.L-1)*out_kv_stride_L;

  // The output pointer.
  float *ptr_out_kv = &(so == 0 ? params.out_k : params.out_v)[offset_out_kv];

  // Store to shared memory.
  if( si < D ) {
    smem_[smem_curr].qg[so*D + si] = ldg_qg;
    smem_[smem_curr].kv[so*D + si] = ldg_kv;
  }

  // The position of the thread in the output dimension.
  int oo = si / THREADS_PER_HEAD % D;
  int oi = si % THREADS_PER_HEAD * 4;

  // Iterate over the timesteps.
  for( int ti = 0; ti < params.L; ++ti ) {

    // Is it the last iteration?
    int is_last = ti == params.L - 1;

    // Trigger the next loads.
    if( !is_last && active ) {
      ldg_qg = *ptr_qg;
      ldg_kv = *ptr_kv;
    }

    // Move the load pointers.
    ptr_qg -= qg_stride_L;
    ptr_kv -= kv_stride_L;

    // Make sure the data is in shared memory.
    __syncthreads();

    // Each thread loads 4 values from G or Q.
    float4 g[FLOAT4s_PER_THREAD];
    #pragma unroll
    for( int ii = 0; ii < FLOAT4s_PER_THREAD; ++ii ) {
      float *smem_ptr = &smem_[smem_curr].qg[(so^1)*D + oi];
      g[ii] = *reinterpret_cast<const float4*>(&smem_ptr[ii*THREADS_PER_HEAD*4]);
    }

    // Each thread loads a single from Q or G value.
    float q = smem_[smem_curr].qg[so*D + oo];

    // Update the G*Q^T or Q*G^T product.
    #pragma unroll
    for( int ii = 0; ii < FLOAT4s_PER_THREAD; ++ii ) {
      gq[ii].x += g[ii].x * q;
      gq[ii].y += g[ii].y * q;
      gq[ii].z += g[ii].z * q;
      gq[ii].w += g[ii].w * q;
    }

    // Load the V or K values from shared memory.
    float4 v[FLOAT4s_PER_THREAD];
    #pragma unroll
    for( int ii = 0; ii < FLOAT4s_PER_THREAD; ++ii ) {
      float *smem_ptr = &smem_[smem_curr].kv[(so^1)*D + oi];
      v[ii] = *reinterpret_cast<const float4*>(&smem_ptr[ii*THREADS_PER_HEAD*4]);
    }

    // Compute the partial output value for that thread.
    float sum = 0.f;
    #pragma unroll
    for( int ii = 0; ii < FLOAT4s_PER_THREAD; ++ii ) {
      sum += v[ii].x * gq[ii].x;
      sum += v[ii].y * gq[ii].y;
      sum += v[ii].z * gq[ii].z;
      sum += v[ii].w * gq[ii].w;
    }

    // Finalize the computation of the sum (if we have more than 1 thread per head).
    if( THREADS_PER_HEAD > 1 ) {

      // Finalize the sum for each head.
      #pragma unroll
      for( int mask = THREADS_PER_HEAD / 2; mask >= 1; mask /= 2 ) {
        sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
      }

      // Store to shared memory.
      if( oi == 0 ) {
        smem_[smem_curr].out_kv[so*D + oo] = sum;
      }

      // Make sure the data is in shared memory.
      __syncthreads();

      // Active threads read the data to store.
      if( si < hidden_size_per_head ) {
        sum = smem_[smem_curr].out_kv[so*D + si];
      }

    } // THREADS_PER_HEAD > 1.

    // Store the output. All the threads are active.
    if( si < hidden_size_per_head ) {
      *ptr_out_kv = sum;
    }

    // Move to next location.
    ptr_out_kv -= out_kv_stride_L;

    // Move the shared memory buffer.
    smem_curr = (smem_curr + 1) % 2;

    // Store to shared memory for Q and K.
    if( !is_last && si < D ) {
      smem_[smem_curr].qg[so*D + si] = ldg_qg;
      smem_[smem_curr].kv[so*D + si] = ldg_kv;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int D, int THREADS_PER_HEAD >
int lmha_bwd_(const Lmha_bwd_params<float> &params) {
  int block = D*THREADS_PER_HEAD*2;
  if( block >= 1024 || params.B > 65535 ) {
    return 1;
  }
  dim3 grid(params.H, params.B);
  lmha_bwd_kernel<D, THREADS_PER_HEAD><<<grid, block>>>(params);
  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int lmha_bwd(const Lmha_bwd_params<float> &params) {
  int blocks = params.B * params.H;
  if( blocks < LOW_OCCUPANCY_THRESHOLD ) {
    return 1;
  }

  int hidden_size_per_head = max(params.E, params.M);
  int res = 1;
  if( hidden_size_per_head <= 32 ) {
    res = lmha_bwd_< 32, 1>(params);
  } else if( hidden_size_per_head <= 64 ) {
    res = lmha_bwd_< 64, 1>(params);
  } else if( hidden_size_per_head <= 128 ) {
    res = lmha_bwd_<128, 2>(params);
  } else if( hidden_size_per_head <= 256 ) {
    res = lmha_bwd_<256, 4>(params);
  }
  return res;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int lmha_bwd(const torch::Tensor queries,
             const torch::Tensor keys,
             const torch::Tensor values,
             const torch::Tensor grad_out,
             torch::Tensor grad_queries,
             torch::Tensor grad_keys,
             torch::Tensor grad_values) {

  // Make sure that we are using the correct GPU device
  torch::DeviceGuard _guard(queries.device());

  // Make sure the inner-most dimension of the tensors is packed.
  assert(queries     .stride(3) == 1);
  assert(keys        .stride(3) == 1);
  assert(values      .stride(3) == 1);
  assert(grad_out    .stride(3) == 1);
  assert(grad_queries.stride(3) == 1);
  assert(grad_keys   .stride(3) == 1);
  assert(grad_values .stride(3) == 1);

  // Extract the dimensions.
  int N = queries.size(0);
  int H = queries.size(1);
  int L = queries.size(2);
  int E = queries.size(3);
  int M = values.size (3);

  // Gradient on Q.

  // The structure of params.
  Lmha_params<float> params;
  set_params(params, grad_out, values, keys, grad_queries);

  // Launch the kernel.
  int res = lmha<false>(params);
  if( res ) {
    return res;
  }

  // Gradient on K and V together.

  Lmha_bwd_params<float> bwd_params;
  bwd_params.out_k = grad_keys.data_ptr<float>();
  bwd_params.out_v = grad_values.data_ptr<float>();
  bwd_params.q = queries.data_ptr<float>();
  bwd_params.k = keys.data_ptr<float>();
  bwd_params.v = values.data_ptr<float>();
  bwd_params.g = grad_out.data_ptr<float>();

  bwd_params.B = N;
  bwd_params.L = L;
  bwd_params.H = H;
  bwd_params.E = E;
  bwd_params.M = M;

  bwd_params.q_stride_B = queries.stride(0);
  bwd_params.q_stride_H = queries.stride(1);
  bwd_params.q_stride_L = queries.stride(2);
  bwd_params.k_stride_B = keys.stride(0);
  bwd_params.k_stride_H = keys.stride(1);
  bwd_params.k_stride_L = keys.stride(2);
  bwd_params.v_stride_B = values.stride(0);
  bwd_params.v_stride_H = values.stride(1);
  bwd_params.v_stride_L = values.stride(2);
  bwd_params.g_stride_B = grad_out.stride(0);
  bwd_params.g_stride_H = grad_out.stride(1);
  bwd_params.g_stride_L = grad_out.stride(2);

  bwd_params.out_k_stride_B = grad_keys.stride(0);
  bwd_params.out_k_stride_H = grad_keys.stride(1);
  bwd_params.out_k_stride_L = grad_keys.stride(2);
  bwd_params.out_v_stride_B = grad_values.stride(0);
  bwd_params.out_v_stride_H = grad_values.stride(1);
  bwd_params.out_v_stride_L = grad_values.stride(2);

  // Try to run the fused kernel.
  int fallback = lmha_bwd(bwd_params);

  // If it failed, fallback on separate kernels for K and V.
  if( fallback ) {

    // Gradient on K.

    // Launch the kernel.
    set_params(params, values, grad_out, queries, grad_keys);
    res = lmha<true>(params);
    if( res ) {
      return res;
    }

    // Gradient on V.

    // Launch the kernel.
    set_params(params, keys, queries, grad_out, grad_values);
    return lmha<true>(params);
  }

  // It worked...
  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace nvidia
#endif // #ifdef ENABLE_NVIDIA_OPTIMIZATIONS

////////////////////////////////////////////////////////////////////////////////////////////////////

typedef torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> float_accessor;

#define E_BLOCK_SIZE 8

__global__ void causal_dot_product_kernel(
    const float_accessor queries,
    const float_accessor keys,
    const float_accessor values,
    float_accessor result,
    const int N,
    const int H,
    const int L,
    const int E,
    const int M
) {
    int n = blockIdx.y;
    int h = blockIdx.z;

    int e_start = blockIdx.x * E_BLOCK_SIZE;
    int m = threadIdx.x % M;

    extern __shared__ float shared_mem[];
    float* shared_kv = shared_mem;

    for (int e_local = 0; e_local < E_BLOCK_SIZE && e_local + e_start < E; e_local++) {
      shared_kv[m + e_local * M] = 0;
    }

    for (int t=0; t<L; t++) {
      float res = 0;
      for (int e_local = 0; e_local < E_BLOCK_SIZE && e_local + e_start < E; e_local++) {
        shared_kv[e_local*M + m] += keys[n][h][t][e_local + e_start] * values[n][h][t][m];
        res += queries[n][h][t][e_local + e_start] * shared_kv[e_local*M + m];
      }
      atomicAdd(
          &result[n][h][t][m],
          res
      );
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void causal_dot_product_(const torch::Tensor queries,
                         const torch::Tensor keys,
                         const torch::Tensor values,
                         torch::Tensor product) {
    // Make sure that we are using the correct GPU device
    torch::DeviceGuard _guard(queries.device());

    int N = queries.size(0);
    int H = queries.size(1);
    int L = queries.size(2);
    int E = queries.size(3);
    int M = values.size(3);

    const int blocks_per_sequence = (E + E_BLOCK_SIZE - 1) / E_BLOCK_SIZE;

    dim3 blockDim(M, 1, 1);
    dim3 gridDim(blocks_per_sequence, N, H);
    const int shared_mem_forward = E_BLOCK_SIZE * M * sizeof(float);

    causal_dot_product_kernel<<<gridDim, blockDim, shared_mem_forward>>>(
      queries.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
      keys.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
      values.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
      product.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
      N, H, L, E, M
    );
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void causal_dot_product(const torch::Tensor queries,
                        const torch::Tensor keys,
                        const torch::Tensor values,
                        torch::Tensor product) {
#ifdef ENABLE_NVIDIA_OPTIMIZATIONS
  int fallback = nvidia::lmha_fwd(queries, keys, values, product);
#else
  int fallback = 1;
#endif
  if( fallback ) {
    causal_dot_product_(queries, keys, values, product);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#define M_BLOCK_SIZE 4

// we need shared memory to store
// kv
// Backward direction
// kv_backwards
// Shared memory usage
__global__ void causal_dot_backward_query_key_kernel(
    const float_accessor queries,
    const float_accessor keys,
    const float_accessor values,
    const float_accessor grad_out,
    float_accessor grad_queries,
    float_accessor grad_keys,
    int N,
    int H,
    int L,
    int E,
    int M
) {
    int n = blockIdx.y;
    int h = blockIdx.z;

    int m_start = blockIdx.x * M_BLOCK_SIZE;
    int e = threadIdx.x % E;

    extern __shared__ float shared_mem[];
    const int shared_kv_size = M_BLOCK_SIZE * E;
    float* shared_kv = shared_mem;
    float* shared_kv_bw = shared_mem + shared_kv_size;

    for (int m_local = 0; m_local < M_BLOCK_SIZE && m_local + m_start < M; m_local++) {
      shared_kv[m_local * E + e] = 0;
      shared_kv_bw[m_local * E + e] = 0;
    }

    for (int l=0; l<L; l++) {
      float res = 0, res_bw = 0;
      int l_b = L - l - 1;
      for (int m_local = 0; m_local < M_BLOCK_SIZE && m_local + m_start < M; m_local++) {
        shared_kv[m_local*E + e] += keys[n][h][l][e] * values[n][h][l][m_start + m_local];
        shared_kv_bw[m_local*E + e] += queries[n][h][l_b][e] * grad_out[n][h][l_b][m_start + m_local];
        res += grad_out[n][h][l][m_start + m_local] * shared_kv[m_local*E + e];
        res_bw += values[n][h][l_b][m_start + m_local] * shared_kv_bw[m_local*E + e];
      }
      atomicAdd(
        &grad_queries[n][h][l][e],
        res
      );
      atomicAdd(
        &grad_keys[n][h][l_b][e],
        res_bw
      );
    }
}


__global__ void causal_dot_backward_value_kernel(
    const float_accessor queries,
    const float_accessor keys,
    const float_accessor values,
    const float_accessor grad_out,
    float_accessor grad_keys,
    float_accessor grad_values,
    int N,
    int H,
    int L,
    int E,
    int M
) {
    int n = blockIdx.y;
    int h = blockIdx.z;

    int e_start = blockIdx.x * E_BLOCK_SIZE;
    int m = threadIdx.x % M;

    extern __shared__ float shared_mem[];
    float* shared_kv = shared_mem;
    for (int e_local = 0; e_local < E_BLOCK_SIZE && e_local + e_start < E; e_local++) {
      shared_kv[m + e_local * M] = 0;
    }

    for (int l = 0; l < L; l++) {
        int l_b = L - l -1;
        float res = 0;
        for (int e_local = 0; e_local < E_BLOCK_SIZE && e_local + e_start < E; e_local++) {
          shared_kv[e_local*M + m] += queries[n][h][l_b][e_start + e_local] * grad_out[n][h][l_b][m];
          res += keys[n][h][l_b][e_start + e_local] * shared_kv[e_local*M + m];
        }
        atomicAdd(
            &grad_values[n][h][l_b][m],
            res
        );
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void causal_dot_backward_(const torch::Tensor queries,
                          const torch::Tensor keys,
                          const torch::Tensor values,
                          const torch::Tensor grad_out,
                          torch::Tensor grad_queries,
                          torch::Tensor grad_keys,
                          torch::Tensor grad_values) {

    // Make sure that we are using the correct GPU device
    torch::DeviceGuard _guard(queries.device());

    int N = queries.size(0);
    int H = queries.size(1);
    int L = queries.size(2);
    int E = queries.size(3);
    int M = values.size(3);

    const int blocks_per_sequence = (M + M_BLOCK_SIZE - 1) / M_BLOCK_SIZE;

    dim3 blockDim(E, 1, 1);
    dim3 gridDim(blocks_per_sequence, N, H);
    const int shared_mem_qk_backward = 2 * M_BLOCK_SIZE * E * sizeof(float);

    causal_dot_backward_query_key_kernel<<<gridDim, blockDim, shared_mem_qk_backward>>>(
      queries.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
      keys.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
      values.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
      grad_out.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
      grad_queries.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
      grad_keys.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
      N, H, L, E, M
    );

    const int blocks_per_sequence_value = (E + E_BLOCK_SIZE - 1) / E_BLOCK_SIZE;

    dim3 blockDimv(M, 1, 1);
    dim3 gridDimv(blocks_per_sequence_value, N, H);
    const int shared_mem_v_backward = E_BLOCK_SIZE * M * sizeof(float);
    causal_dot_backward_value_kernel<<<gridDimv, blockDimv, shared_mem_v_backward>>>(
      queries.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
      keys.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
      values.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
      grad_out.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
      grad_keys.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
      grad_values.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
      N, H, L, E, M
    );
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void causal_dot_backward(const torch::Tensor queries,
                         const torch::Tensor keys,
                         const torch::Tensor values,
                         const torch::Tensor grad_out,
                         torch::Tensor grad_queries,
                         torch::Tensor grad_keys,
                         torch::Tensor grad_values) {
#ifdef ENABLE_NVIDIA_OPTIMIZATIONS
  int fallback = nvidia::lmha_bwd(queries,
                                  keys,
                                  values,
                                  grad_out,
                                  grad_queries,
                                  grad_keys,
                                  grad_values);
#else
  int fallback = 1;
#endif
  if( fallback ) {
    // Make sure that the gradient tensors are 0. This is needed because the
    // bwd pass might have partially executed and filled in some values in
    // grad_queries or grad_keys.
    //
    // This adds a small overhead every time we have to fall back to the old
    // kernel for the backward pass.
    grad_queries.zero_();
    grad_keys.zero_();
    causal_dot_backward_(queries, keys, values, grad_out, grad_queries, grad_keys, grad_values);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "causal_dot_product",
        &causal_dot_product,
        "Compute the weighted sum of values but attending only to previous "
        "values."
    );
    m.def(
        "causal_dot_backward",
        &causal_dot_backward,
        "Compute the gradients for the causal dot product."
    );
}
