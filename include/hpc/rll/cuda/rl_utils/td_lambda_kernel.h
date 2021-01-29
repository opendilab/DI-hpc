#ifndef HPC_RLL_CUDA_TD_LAMBDA_KERNEL_H_
#define HPC_RLL_CUDA_TD_LAMBDA_KERNEL_H_

#include "hpc/rll/cuda/common.h"
#include "hpc/rll/cuda/reduce.h"

namespace hpc {
namespace rll {
namespace cuda {

void __global__ tdLambdaForwardKernel(unsigned int time_step, unsigned int batch_size, float gamma, float lambda,
        const float* value, const float* reward, const float* weight, float* loss, float* grad_buf) {
    unsigned int gid = threadIdx.x + blockIdx.x*blockDim.x;
    
    float sum_square = 0;
    if (gid < batch_size) {
        float rt = 0.f;
        for (int t = time_step - 1; t >= 0; --t) {
            unsigned int index = t * batch_size + gid;

            float value_data = value[index];
            float next_value_data = value[index + batch_size];
            float reward_data = reward[index];
            float weight_data = weight[index];

            float tmp = (t == time_step - 1) ? next_value_data : (lambda * rt + (1.f - lambda) * next_value_data);
            rt = reward_data + gamma * tmp;

            float loss = (rt - value_data);
            grad_buf[index] = weight_data * (2.f * loss * (-1.f));
            sum_square += loss * loss * weight_data;
        }
    }

    float reduced_sum_square = blockReduceSum<float>(sum_square);
    if (threadIdx.x == 0) {
        float mean_loss = 0.5 * reduced_sum_square / (time_step * batch_size);
        atomicAdd(loss, mean_loss);
    }
}

void __global__ tdLambdaBackwardKernel(unsigned int time_step, unsigned int batch_size,
        const float* grad_loss, const float* grad_buf, float* grad_value) {
    unsigned int gid = threadIdx.x + blockIdx.x*blockDim.x;

    if (gid < (time_step + 1) * batch_size) {
        float grad = *grad_loss;
        float grad_mean = 1.f / (time_step * batch_size);
        grad_value[gid] = (gid < time_step * batch_size) ? (grad * 0.5 * grad_mean * grad_buf[gid]) : 0.f;
    }
}

}  // namespace cuda
}  // namespace rll
}  // namespace hpc
#endif // HPC_RLL_CUDA_TD_LAMBDA_KERNEL_H_
