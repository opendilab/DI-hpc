#ifndef HPC_RLL_CUDA_GAE_KERNEL_H_
#define HPC_RLL_CUDA_GAE_KERNEL_H_

#include "hpc/rll/cuda/common.h"

namespace hpc {
namespace rll {
namespace cuda {

void __global__ gaeForwardKernel(unsigned int time_step, unsigned int batch_size, float gamma, float lambda,
        const float* value, const float* reward, float* adv) {
    unsigned int gid = threadIdx.x + blockIdx.x*blockDim.x;
    if (gid < batch_size) {
        float gae_item = 0;
        float denom = 0;
        float factor = gamma * lambda;
        for (int t = time_step - 1; t >= 0; --t) {
            unsigned int index = t * batch_size + gid;

            denom = 1 + lambda * denom;
            float reward_data = reward[index];
            float value_data = value[index];
            float next_value_data = value[index + batch_size];
            float delta = reward_data + gamma * next_value_data - value_data;
            gae_item = denom * delta + factor * gae_item;
            adv[index] = gae_item / denom;
        }
    }
}

}  // namespace cuda
}  // namespace rll
}  // namespace hpc

#endif // HPC_RLL_CUDA_GAE_KERNEL_H_
