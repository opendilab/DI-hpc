#ifndef HPC_RLL_CUDA_QRDQN_NSTEP_TD_ERROR_KERNEL_H_
#define HPC_RLL_CUDA_QRDQN_NSTEP_TD_ERROR_KERNEL_H_

#include "hpc/rll/cuda/common.h"
#include "hpc/rll/cuda/reduce.h"

namespace hpc {
namespace rll {
namespace cuda {

void __global__ bellmanErrorKernel(unsigned int tau, unsigned int time_step, unsigned int batch_size, unsigned int action_dim, float gamma,
        const float* q, const float* next_n_q, const int64_t* action, const int64_t* next_n_action,
        const float* reward, const float* done, const float* value_gamma, float* bellman_err_buf) {
    unsigned int gidx = threadIdx.x + blockIdx.x*blockDim.x; // tau
    unsigned int gidy = blockIdx.y; // batch_size
    if (gidx >= tau) return;

    int64_t action_data = action[gidy];
    int64_t next_n_action_data = next_n_action[gidy];
    float value_gamma_data = value_gamma[gidy];
    float done_data = done[gidy];
    float not_done_data = 1.f - done_data;

    // for i in range(1, nstep):
    // reward_factor[i] = gamma * reward_factor[i - 1]
    // reward = torch.matmul(reward_factor, reward)
    float reward_factor = 1;
    float reward_data = 0;
    for (int t = 0; t < time_step; t++) {
        reward_data += reward_factor * reward[t * batch_size + gidy];
        reward_factor *= gamma;
    }

    float target_qsa = next_n_q[gidy * action_dim * tau + next_n_action_data * tau + gidx];
    float target_qsa_transform = reward_data + value_gamma_data * target_qsa * not_done_data;
    for (int t = 0; t < tau; t++) {
        float qsa = q[gidy * action_dim * tau + action_data * tau + t];
        bellman_err_buf[gidy * tau * tau + t * tau + gidx] = target_qsa_transform - qsa;
    }
}

void __global__ smoothL1LossKernel(unsigned int tau, unsigned int batch_size,
        const float* bellman_err_buf, float* quantile_huber_loss_buf, float* grad_buf) {
    unsigned int gidx = threadIdx.x + blockIdx.x*blockDim.x; // tau
    unsigned int gidy = threadIdx.y + blockIdx.y*blockDim.y; // tau
    unsigned int gidz = blockIdx.z; // batch_size
    if (gidx >= tau || gidy >= tau) return;

    float bellman_error_data = bellman_err_buf[gidz * tau * tau + gidy * tau + gidx];
    float huber_loss_data = 0.f;
    float grad_buf_data = 0.f;
    if (abs(bellman_error_data) < 1) {
        huber_loss_data = 0.5f * bellman_error_data * bellman_error_data;
        grad_buf_data = bellman_error_data;
    } else {
        huber_loss_data = (abs(bellman_error_data) - 0.5f);
        grad_buf_data = (bellman_error_data >= 0) ? 1.f : (-1.f);
    }

    float tmp1 = tau - ((bellman_error_data <= 0) ? 1.f : 0.f);
    float tmp2 = huber_loss_data * tmp1;
    float quantile_huber_loss_data = abs(tmp2);
    grad_buf_data *= (tmp2 >= 0 ? tmp1 : -tmp1);

    quantile_huber_loss_buf[gidz * tau * tau + gidy * tau + gidx] = quantile_huber_loss_data;
    grad_buf[gidz * tau * tau + gidy * tau + gidx] = grad_buf_data;

}

void __global__ lossKernel(unsigned int tau, unsigned int batch_size,
        const float* quantile_huber_loss_buf, const float* weight,
        float* td_err, float* loss) {
    unsigned int block_start = blockIdx.x * tau * tau;
    unsigned int start = block_start + threadIdx.x;
    unsigned int end = block_start + tau * tau;

    float partial_sum_huber = 0.f;
    for (int i = start; i < end; i += blockDim.x) {
        partial_sum_huber += quantile_huber_loss_buf[i];
    }
    float sum_huber = blockReduceSum<float>(partial_sum_huber);
    float mean_huber = sum_huber / tau;

    if (threadIdx.x == 0) {
        td_err[blockIdx.x] = mean_huber;
        atomicAdd(loss, mean_huber * weight[blockIdx.x] / batch_size);
    }
}

void __global__ backwardKernel(unsigned int tau, unsigned int batch_size, unsigned int action_dim,
        const float* grad_loss, const float* grad_buf,
        const float* weight, const int64_t* action, float* grad_q) {
    unsigned int gidx = threadIdx.x + blockIdx.x*blockDim.x; // tau
    unsigned int gidy = blockIdx.y; // batch_size
    if (gidx >= tau) return;

    float grad_data = 0.f;
    for (int t = 0; t < tau; t++) {
        grad_data += grad_buf[gidy * tau * tau + gidx * tau + t];
    }
    grad_data *= (grad_loss[0] / batch_size * weight[gidy] / tau); // mean, weight, mean
    grad_data *= -1.f; // target_qsa - qsa

    int output_index = gidy * action_dim * tau + action[gidy] * tau + gidx;
    grad_q[output_index] = grad_data;
}

}  // namespace cuda
}  // namespace rll
}  // namespace hpc
#endif // HPC_RLL_CUDA_QRDQN_NSTEP_TD_ERROR_KERNEL_H_
