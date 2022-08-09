#ifndef HPC_RLL_CUDA_SOFTARGMAX_KERNEL_H_
#define HPC_RLL_CUDA_SOFTARGMAX_KERNEL_H_

#include "hpc/rll/cuda/common.h"
#include "hpc/rll/cuda/reduce.h"
#include "hpc/rll/cuda/basic_math.h"
#include <float.h>
#include <math.h>
#include <memory>

namespace hpc {
namespace rll {
namespace cuda {
#define FINAL_MASK 0xffffffff

__device__ __forceinline__ float _SafeExp(const float v) {
    return expf(min(v, 80.0f));
}

__device__ __forceinline__ float _LogAdd(const float x, const float y) {
    return x + max(log(_SafeExp(y - x) + (float)1.0f), y - x);
}

__device__ __forceinline__ float WARP_SHFL_XOR(float value, int laneMask,
                                            int width = 32, unsigned int mask = FINAL_MASK) {
#if __CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__ * 10 >= 9000
    return __shfl_xor_sync(mask, value, laneMask, width);
#else
    return __shfl_xor(value, laneMask, width);
#endif
}

__device__ __forceinline__ float WarpReduceLogAddSum(float val) {
    for (int mask = 16; mask > 0; mask >>= 1)
        val = _LogAdd(WARP_SHFL_XOR(val, mask, 32, FINAL_MASK), val);
    return val;
}

__global__ void SoftmaxScoreKernel(const float* in, float* out, const int T) {
    auto cur_in = in + blockIdx.x * T;
    auto cur_out = out + blockIdx.x * T;
    // reduce log sum
    float log_sum = -80.0f;
    for(auto tid = threadIdx.x; tid < T; tid += blockDim.x) {
        log_sum = _LogAdd(__ldg(cur_in + tid), log_sum);
    }
    log_sum = WarpReduceLogAddSum(log_sum);
    for(auto tid = threadIdx.x; tid < T; tid += blockDim.x) {
        cur_out[tid] = expf(__ldg(cur_in + tid) - log_sum);
    }
}


__global__ void MulSumKernel(unsigned int H, unsigned int W, float* x, float* res) {
    unsigned int block_start = blockIdx.x * H * W;
    unsigned int start = block_start + threadIdx.x;
	unsigned int end = block_start + H * W;
    float sum_h = 0;
    float sum_w = 0;
    for(int i = start; i < end; i += blockDim.x) {
        sum_h += x[i] * ((i - block_start) / W);
        sum_w += x[i] * ((i - block_start) % W);
    }
    float reduce_sum_h = blockReduceSum<float>(sum_h);
    float reduce_sum_w = blockReduceSum<float>(sum_w);
    static __shared__ float s_sum_h;
    static __shared__ float s_sum_w;
    if (threadIdx.x == 0) {
        s_sum_h = reduce_sum_h;
        s_sum_w = reduce_sum_w;
    }
    __syncthreads();
    
    res[blockIdx.x * 2] = s_sum_h;
    res[blockIdx.x * 2 + 1] = s_sum_w;
}

__global__ void BackwardKernel(unsigned int H, unsigned int W, float* x, float* grad_res, float* grad_x) {
    unsigned int block_start = blockIdx.x * H * W;
    unsigned int start = block_start + threadIdx.x;
	unsigned int end = block_start + H * W;
    float sum_hw = 0;
    static __shared__ float s_sum_hw;
    for(int i = start; i < end; i += blockDim.x) {
        sum_hw += x[i] * (((i - block_start) / W) + ((i - block_start) % W));
    }
    float reduce_sum_hw = blockReduceSum<float>(sum_hw);
    if(threadIdx.x == 0) {
        s_sum_hw = reduce_sum_hw;
    }
    __syncthreads();
    for(int i = start; i < end; i += blockDim.x) {
        grad_x[i] = x[i] * (((i - block_start) / W) + ((i - block_start) % W) - s_sum_hw); //* (grad_res[blockIdx.x * 2] + grad_res[blockIdx.x * 2 + 1]);
    }
}



   




}  // namespace cuda
}  // namespace rll
}  // namespace hpc
#endif // HPC_RLL_CUDA_COMA_KERNEL_H_
