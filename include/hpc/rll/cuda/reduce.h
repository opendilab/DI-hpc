#ifndef HPC_RLL_CUDA_REDUCE_H_
#define HPC_RLL_CUDA_REDUCE_H_

#include "hpc/rll/cuda/common.h"

namespace hpc {
namespace rll {
namespace cuda {

const unsigned int WARP_REDUCE_MASK = 0xffffffff;

// reduce to all the threads in the warp
template <typename T>
__forceinline__ __device__ T warpReduceSum(T val) {
  for (int mask = (WARP_SIZE >> 1); mask > 0; mask >>= 1)
    val += __shfl_xor_sync(WARP_REDUCE_MASK, val, mask, WARP_SIZE);
  return val;
}

// reduce to all the threads in the warp
template <typename T>
__forceinline__ __device__ T warpReduceMax(T val) {
  for (int mask = (WARP_SIZE >> 1); mask > 0; mask >>= 1)
    val = max(val, __shfl_xor_sync(WARP_REDUCE_MASK, val, mask, WARP_SIZE));
  return val;
}

// reduce to all the threads in the warp
template <typename T>
__forceinline__ __device__ T warpReduceMin(T val) {
  for (int mask = (WARP_SIZE >> 1); mask > 0; mask >>= 1)
    val = min(val, __shfl_xor_sync(WARP_REDUCE_MASK, val, mask, WARP_SIZE));
  return val;
}

// Calculate the sum of all elements in a block,
// reduce to thread 0, not all the threads in the block
template <typename T>
__forceinline__ __device__ T blockReduceSum(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum<T>(val);

  if (lane == 0) shared[wid] = val;
  __syncthreads();

  if (wid == 0) {
      val = (threadIdx.x < ((blockDim.x + 31) >> 5)) ? shared[lane] : (T)0.0f;
      val = warpReduceSum<T>(val);
      return val;
  }
  return (T)0.0f;
}

// Calculate the maximum of all elements in a block
// reduce to thread 0, not all the threads in the block
template <typename T>
__forceinline__ __device__ T blockReduceMax(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceMax<T>(val);

  if (lane == 0) shared[wid] = val;
  __syncthreads();

  if (wid == 0) {
      val = (threadIdx.x < ((blockDim.x + 31) >> 5)) ? shared[lane]
          : CUDA_FLOAT_INF_NEG;
      val = warpReduceMax<T>(val);
      return val;
  }
  return CUDA_FLOAT_INF_NEG;
}

// Calculate the minimum of all elements in a block
// reduce to thread 0, not all the threads in the block
template <typename T>
__forceinline__ __device__ T blockReduceMin(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceMin<T>(val);

  if (lane == 0) shared[wid] = val;
  __syncthreads();

  if (wid == 0) {
      val = (threadIdx.x < ((blockDim.x + 31) >> 5)) ? shared[lane]
          : CUDA_FLOAT_INF_POS;
      val = warpReduceMin<T>(val);
      return val;
  }
  return CUDA_FLOAT_INF_POS;
}

}  // namespace cuda
}  // namespace rll
}  // namespace hpc

#endif // HPC_RLL_CUDA_REDUCE_H_
