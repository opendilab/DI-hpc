#ifndef HPC_RLL_CUDA_UPGO_KERNEL_H_
#define HPC_RLL_CUDA_UPGO_KERNEL_H_

#include "hpc/rll/cuda/common.h"
#include "hpc/rll/cuda/reduce.h"

namespace hpc {
namespace rll {
namespace cuda {

void __global__ upgoAdvantageKernel(unsigned int time_step, unsigned int batch_size,
        const float* rho, const float* reward, const float* value, float* advantage) {
    unsigned int gid = threadIdx.x + blockIdx.x*blockDim.x;

    if (gid < batch_size) {
        float item = 0;
        for (int t = time_step - 1; t >= 0; --t) {
            unsigned int index = t * batch_size + gid;

            float rho_data = rho[index];

            float reward_data0 = reward[index];
            // Note: when t == time_step - 1, reward_data1 is not used. Just avoid accessing out of memory bound.
            float reward_data1 = (t == time_step - 1) ? 0.f : reward[index + batch_size];

            float value0 = value[index];
            float value1 = value[index + batch_size];
            // Note: when t == time_step - 1, value2 is not used. Just avoid accessing out of memory bound.
            float value2 = (t == time_step - 1) ? 0.f : value[index + batch_size * 2];

            float value_data = ((t < time_step - 1) && (reward_data1 + value2 >= value1)) ? item : value1;

            float rt = reward_data0 + value_data;
            advantage[index] = (rt - value0) * rho_data;
            item = rt;
        }
    }
}

void __global__ crossEntropyKernel(unsigned int num,
        const float* input, const int64_t* target, float* output, float* grad) {
    unsigned int block_start = blockIdx.x * num;
    unsigned int start = block_start + threadIdx.x;
    unsigned int end = block_start + num;

    // step 1 get max_x
    float max_x = CUDA_FLOAT_INF_NEG;
    for (int i = start; i < end; i += blockDim.x) {
        max_x = max(max_x, input[i]);
    }
    static __shared__ float s_max_x;
    float reduce_max_x =  blockReduceMax<float>(max_x);
    if (threadIdx.x == 0) {
        s_max_x = reduce_max_x;
    }
    __syncthreads();

    // step 2 compute sum(exp(x - max_x))
    static __shared__ float s_sum_exp_x;
    float sum_exp_x = 0.0;
    for (int i = start; i < end; i += blockDim.x) {
        sum_exp_x += std::exp(input[i] - s_max_x);
    }
    float reduce_sum_exp_x = blockReduceSum<float>(sum_exp_x);
    if (threadIdx.x == 0) {
        s_sum_exp_x = reduce_sum_exp_x;
    }
    __syncthreads();

    // step 2 compute cross entropy and grad
    for (int i = start; i < end; i += blockDim.x) {
        bool flag = (i - block_start == target[blockIdx.x]);

        float softmax_data = std::exp(input[i] - s_max_x) / s_sum_exp_x;

        if (flag)
            output[blockIdx.x] = std::log(softmax_data);

        grad[i] = flag ? (1 - softmax_data) : (-softmax_data);
    }
}

void __global__ upgoLossKernel(unsigned int time_step, unsigned int batch_size,
        const float* advantage, const float* metric, float* loss) {
    unsigned int gid = threadIdx.x + blockIdx.x*blockDim.x;

    float sum_loss = (gid < time_step * batch_size) ? (advantage[gid] * metric[gid]) : 0.f;
    float reduced_sum_loss = blockReduceSum<float>(sum_loss);

    if (threadIdx.x == 0) {
        float mean_loss = reduced_sum_loss / (time_step * batch_size);
        atomicAdd(loss, -mean_loss);
    }
}

void __global__ upgoBackwardKernel(unsigned int time_step, unsigned int batch_size, unsigned int num_output,
        const float* grad_loss, const float* grad_buf,
        const float* advantages, float* grad_target_output) {
    unsigned int gid = threadIdx.x + blockIdx.x*blockDim.x;

    if (gid < time_step * batch_size * num_output) {
        unsigned int tb_id = gid / num_output;

        float grad = (*grad_loss);
        float grad_mean = 1.f / (time_step * batch_size);
        grad_target_output[gid] = grad * (-1.f) * grad_mean * advantages[tb_id] * grad_buf[gid];
    }
}

}  // namespace cuda
}  // namespace rll
}  // namespace hpc
#endif // HPC_RLL_CUDA_UPGO_KERNEL_H_
