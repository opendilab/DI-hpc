#ifndef HPC_RLL_CUDA_RETRACE_KERNEL_H_
#define HPC_RLL_CUDA_RETRACE_KERNEL_H_

#include "hpc/rll/cuda/common.h"
#include "hpc/rll/cuda/basic_math.h"

namespace hpc {
namespace rll {
namespace cuda {


__global__ void gatherKernel(unsigned int T, unsigned int B, unsigned int N, float* q_gather, const float* q_values,
                            const int64_t* actions, float* ratio_gather, const float* ratio) {
    int index_offset = blockIdx.x * B;
    for(int tid = threadIdx.x; tid < B; tid += blockDim.x) {
        int index = tid + index_offset;
        int index_val = actions[index];
        int out_index = blockIdx.x * B * N + tid * N + index_val;
        q_gather[index] = q_values[out_index];
        ratio_gather[index] = ratio[out_index];
    }
}

__global__ void retraceKernel(unsigned int T, unsigned int B, const float* v_pred,
        const float* rewards, const int64_t* actions, const float* weights,
        float* q_retraces, float* q_gather, float* ratio_gather, float* tmp_retraces, float gamma) {
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= B) return;

    // step 1: fulfill q_retraces[T] & tmp_retraces
	int offset = T * B;
    float val = v_pred[tid + offset];
    tmp_retraces[tid] = val;
    q_retraces[tid + offset] = val;

	// step 2 q_retraces & tmp_retraces
    // for idx in reversed(T):
    //     q_retraces[idx, ...] = rewards[idx, ...] + gamma * weights[idx, ...] * tmp_retraces
    //     tmp_retraces = ratio_gather[idx, ...].clamp(max=1.0) * (q_retraces[idx, ...] - q_gather[idx, ...]) + v_pred[idx, ...]
    for(int idx = T - 1; idx >= 0; idx--) {
        int index = idx * B + tid;
        q_retraces[index] = rewards[index] + gamma * weights[index] * tmp_retraces[tid];
        tmp_retraces[tid] = min(ratio_gather[index] , 1.0) * (q_retraces[index] - q_gather[index]) + v_pred[index];
    }
	

}



}  // namespace cuda
}  // namespace rll
}  // namespace hpc
#endif // HPC_RLL_CUDA_RETRACE_KERNEL_H_
