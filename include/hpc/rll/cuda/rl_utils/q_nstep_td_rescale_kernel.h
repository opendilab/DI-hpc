#ifndef HPC_RLL_CUDA_Q_NSTEP_TD_RESCALE_KERNEL_H_
#define HPC_RLL_CUDA_Q_NSTEP_TD_RESCALE_KERNEL_H_

#include "hpc/rll/cuda/common.h"
#include "hpc/rll/cuda/reduce.h"

namespace hpc {
namespace rll {
namespace cuda {

void __global__ qNStepTdRescaleForwardKernel(unsigned int time_step, unsigned int batch_size, unsigned int num_output, float gamma,
        const float* q, const float* next_n_q, const int64_t* action, const int64_t* next_n_action,
        const float* reward, const float* done, const float* weight,
        float* td_err, float* loss, float* grad_buf) {
    unsigned int gid = threadIdx.x + blockIdx.x*blockDim.x;

    float sum_square = 0;
    if (gid < batch_size) {
        unsigned int batch_id = gid;
        unsigned int num_out_id = action[batch_id];

        float qsa = q[batch_id * num_output + num_out_id];
        unsigned int next_n_num_out_id = next_n_action[batch_id];
        float target_qsa = next_n_q[batch_id * num_output + next_n_num_out_id];

        // value_inv_transform
        float eps = 1e-2;
        float sign = (target_qsa > 0.f ? 1.f : (target_qsa < 0.f ? -1.f : 0.f));
        float tmp = (sqrt(1.f + 4.f * eps * (abs(target_qsa) + 1.f + eps)) - 1) / (2 * eps);
        target_qsa = sign * (tmp * tmp - 1);

        // nstep_return
        float sum_reward = 0;
        float factor = 1;
        for (int t = 0; t < time_step; ++t) {
            float rw = reward[t * batch_size + batch_id];
            sum_reward += (factor * rw);
            factor *= gamma;
        }
        float done_ = done[batch_id];
        target_qsa = sum_reward + factor * target_qsa * (1.f - done_);

        // value_transform
        sign = (target_qsa > 0.f ? 1.f : (target_qsa < 0.f ? -1.f : 0.f));
        target_qsa = sign * (sqrt(abs(target_qsa) + 1) - 1) + eps * target_qsa;

        float diff = qsa - target_qsa;
        sum_square = diff * diff;
        td_err[batch_id] = sum_square;

        float w = weight[batch_id];
        sum_square *= w;
        grad_buf[batch_id] = 1.f / batch_size * (2.f * diff) * w;
    }

    float reduced_sum_square = blockReduceSum<float>(sum_square);
    if (threadIdx.x == 0) {
        float mean_loss = reduced_sum_square / batch_size;
        atomicAdd(loss, mean_loss);
    }
}

void __global__ qNStepTdRescaleBackwardKernel(unsigned int batch_size, unsigned int num_output,
        const float* grad_loss, const float* grad_buf, const int64_t* action, float* grad_q) {
    unsigned int gid = threadIdx.x + blockIdx.x*blockDim.x; // num_output

    if (gid < num_output) {
        unsigned int batch_id = blockIdx.y;
        float grad = (gid == action[batch_id]) ? grad_buf[batch_id] : 0;
        grad_q[batch_id * num_output + gid] = (*grad_loss) * grad;
    }
}

}  // namespace cuda
}  // namespace rll
}  // namespace hpc
#endif // HPC_RLL_CUDA_Q_NSTEP_TD_RESCALE_KERNEL_H_
