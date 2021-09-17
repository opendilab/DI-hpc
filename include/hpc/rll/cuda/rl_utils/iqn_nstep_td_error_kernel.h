#ifndef HPC_RLL_CUDA_IQN_NSTEP_TD_ERROR_KERNEL_H_
#define HPC_RLL_CUDA_IQN_NSTEP_TD_ERROR_KERNEL_H_

#include "hpc/rll/cuda/common.h"
#include "hpc/rll/cuda/reduce.h"

namespace hpc {
namespace rll {
namespace cuda {

void __global__ bellmanErrorKernel(unsigned int tau, unsigned int tau_prime,
        unsigned int time_step, unsigned int batch_size, unsigned int action_dim, float gamma, float kappa,
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

    float qsa = q[gidx * batch_size * action_dim + gidy * action_dim + action_data];
    for (int t = 0; t < tau_prime; t++) {
        float target_qsa = next_n_q[t * batch_size * action_dim + gidy * action_dim + next_n_action_data];
        float target_qsa_transform = reward_data + value_gamma_data * target_qsa * not_done_data;

        bellman_err_buf[gidy * tau_prime * tau + t * tau + gidx] = target_qsa_transform - qsa;
    }
}

void __global__ quantileHuberErrorKernel(unsigned int tau, unsigned int tau_prime, unsigned int batch_size, float kappa,
        const float* bellman_err_buf, const float* replay_quantiles,
        float* quantile_huber_loss_buf, float* grad_buf) {
    unsigned int gidx = threadIdx.x + blockIdx.x*blockDim.x; // tau
    unsigned int gidy = threadIdx.y + blockIdx.y*blockDim.y; // tau_prime
    unsigned int gidz = blockIdx.z; // batch_size
    if (gidx >= tau || gidy >= tau_prime) return;

    float bellman_error_data = bellman_err_buf[gidz * tau_prime * tau + gidy * tau + gidx];
    float huber_loss_data = 0.f;
    float grad_buf_data = 0.f;
    if (abs(bellman_error_data) <= kappa) {
        huber_loss_data = 0.5 * bellman_error_data * bellman_error_data;
        grad_buf_data = bellman_error_data;
    } else {
        huber_loss_data = kappa * (abs(bellman_error_data) - 0.5 * kappa);
        grad_buf_data = (bellman_error_data >= 0) ? kappa : (-kappa);
    }

    float r_q_data = replay_quantiles[gidx * batch_size + gidz];
    float tmp = abs(r_q_data - ((bellman_error_data < 0) ? 1.f : 0.f)) / kappa;
    float quantile_huber_loss_data = tmp * huber_loss_data;
    grad_buf_data *= tmp;

    quantile_huber_loss_buf[gidz * tau_prime * tau + gidy * tau + gidx] = quantile_huber_loss_data;
    grad_buf[gidz * tau_prime * tau + gidy * tau + gidx] = grad_buf_data;
}

void __global__ lossKernel(unsigned int tau, unsigned int tau_prime, unsigned int batch_size,
        const float* quantile_huber_loss_buf, const float* weight,
        float* td_err, float* loss) {
    unsigned int block_start = blockIdx.x * tau * tau_prime;
    unsigned int start = block_start + threadIdx.x;
    unsigned int end = block_start + tau * tau_prime;

    float partial_sum_huber = 0.f;
    for (int i = start; i < end; i += blockDim.x) {
        partial_sum_huber += quantile_huber_loss_buf[i];
    }
    float sum_huber = blockReduceSum<float>(partial_sum_huber);
    float mean_huber = sum_huber / tau_prime;

    if (threadIdx.x == 0) {
        td_err[blockIdx.x] = mean_huber;
        atomicAdd(loss, mean_huber * weight[blockIdx.x] / batch_size);
    }
}

void __global__ backwardKernel(unsigned int tau, unsigned int tau_prime, unsigned int batch_size, unsigned int action_dim,
        const float* grad_loss, const float* grad_buf,
        const float* weight, const int64_t* action, float* grad_q) {
    unsigned int gidx = threadIdx.x + blockIdx.x*blockDim.x; // tau
    unsigned int gidy = blockIdx.y; // batch_size
    if (gidx >= tau) return;

    float grad_data = 0.f;
    for (int t = 0; t < tau_prime; t++) {
        grad_data += grad_buf[gidy * tau_prime * tau + t * tau + gidx];;
    }
    grad_data *= (grad_loss[0] / batch_size * weight[gidy] / tau_prime); // mean, weight, mean
    grad_data *= -1.f; // target_qsa - qsa

    int output_index = gidx * batch_size * action_dim + gidy * action_dim + action[gidy];
    grad_q[output_index] = grad_data;
}

}  // namespace cuda
}  // namespace rll
}  // namespace hpc
#endif // HPC_RLL_CUDA_IQN_NSTEP_TD_ERROR_KERNEL_H_
