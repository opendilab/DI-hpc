#include "hpc/rll/cuda/rl_utils/entry.h"
#include "hpc/rll/cuda/rl_utils/iqn_nstep_td_error_kernel.h"

namespace hpc {
namespace rll {
namespace cuda {

void IQNNStepTDErrorForward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs,
    float gamma,
    float kappa) {

    unsigned int index = 0;
    const torch::Tensor& q = inputs[index++];
    const torch::Tensor& next_n_q = inputs[index++];
    const torch::Tensor& action = inputs[index++];
    const torch::Tensor& next_n_action = inputs[index++];
    const torch::Tensor& reward = inputs[index++];
    const torch::Tensor& done = inputs[index++];
    const torch::Tensor& replay_quantiles = inputs[index++];
    const torch::Tensor& weight = inputs[index++];
    const torch::Tensor& value_gamma = inputs[index++];
    index = 0;
    torch::Tensor& loss = outputs[index++];
    torch::Tensor& td_err = outputs[index++];
    torch::Tensor& bellman_err_buf = outputs[index++];
    torch::Tensor& quantile_huber_loss_buf = outputs[index++];
    torch::Tensor& grad_buf = outputs[index++];

    // set zero for atomic add
    checkCudaErr(cudaMemsetAsync((float*)(loss.data_ptr()), 0, sizeof(float)));

    const unsigned int tau = q.size(0);
    const unsigned int tau_prime = next_n_q.size(0);
    const unsigned int time_step = reward.size(0);
    const unsigned int batch_size = q.size(1);
    const unsigned int action_dim = q.size(2);
 
    {
    dim3 block_size = {DEFAULT_WARP_NUM * WARP_SIZE, 1, 1};
    dim3 grid_size = {(tau + block_size.x - 1) / block_size.x, batch_size, 1};

    bellmanErrorKernel<<<grid_size, block_size>>>(
            tau, tau_prime, time_step, batch_size, action_dim, gamma, kappa,
            (float*)(q.data_ptr()), (float*)(next_n_q.data_ptr()),
            (int64_t*)(action.data_ptr()), (int64_t*)(next_n_action.data_ptr()),
            (float*)(reward.data_ptr()), (float*)(done.data_ptr()),
            (float*)(value_gamma.data_ptr()), (float*)(bellman_err_buf.data_ptr()));
    }

    {
        dim3 block_size = {WARP_SIZE, DEFAULT_WARP_NUM, 1};
        dim3 grid_size = {(tau + block_size.x - 1) / block_size.x, (tau_prime + block_size.y - 1) / block_size.y, batch_size};

        quantileHuberErrorKernel<<<grid_size, block_size>>> (
                tau, tau_prime, batch_size, kappa,
                (float*)(bellman_err_buf.data_ptr()), (float*)(replay_quantiles.data_ptr()),
                (float*)(quantile_huber_loss_buf.data_ptr()), (float*)(grad_buf.data_ptr()));
    }

    {
        dim3 block_size = {DEFAULT_WARP_NUM * WARP_SIZE, 1, 1};
        dim3 grid_size = {batch_size, 1, 1};

        lossKernel<<<grid_size, block_size>>> (
                tau, tau_prime, batch_size,
                (float*)(quantile_huber_loss_buf.data_ptr()), (float*)(weight.data_ptr()),
                (float*)(td_err.data_ptr()), (float*)(loss.data_ptr()));
    }
}

void IQNNStepTDErrorBackward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs) {

    unsigned int index = 0;
    const torch::Tensor& grad_loss = inputs[index++];
    const torch::Tensor& grad_buf = inputs[index++];
    const torch::Tensor& weight = inputs[index++];
    const torch::Tensor& action = inputs[index++];
    index = 0;
    torch::Tensor& grad_q = outputs[index++];

    const unsigned int batch_size = grad_buf.size(0);
    const unsigned int tau_prime = grad_buf.size(1);
    const unsigned int tau = grad_buf.size(2);
    const unsigned int action_dim = grad_q.size(2);

    // set zero
    checkCudaErr(cudaMemsetAsync((float*)(grad_q.data_ptr()), 0, tau * batch_size * action_dim * sizeof(float)));

    dim3 block_size = {DEFAULT_WARP_NUM * WARP_SIZE, 1, 1};
    dim3 grid_size = {(tau + block_size.x - 1) / block_size.x, batch_size, 1};
    backwardKernel<<<grid_size, block_size>>>(
            tau, tau_prime, batch_size, action_dim,
            (float*)(grad_loss.data_ptr()), (float*)(grad_buf.data_ptr()),
            (float*)(weight.data_ptr()), (int64_t*)(action.data_ptr()),
            (float*)(grad_q.data_ptr()));
}

}  // namespace cuda
}  // namespace rll
}  // namespace hpc
