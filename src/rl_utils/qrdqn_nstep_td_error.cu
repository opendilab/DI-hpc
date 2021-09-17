#include "hpc/rll/cuda/rl_utils/entry.h"
#include "hpc/rll/cuda/rl_utils/qrdqn_nstep_td_error_kernel.h"

namespace hpc {
namespace rll {
namespace cuda {

void QRDQNNStepTDErrorForward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs,
    float gamma) {

    unsigned int index = 0;
    const torch::Tensor& q = inputs[index++];
    const torch::Tensor& next_n_q = inputs[index++];
    const torch::Tensor& action = inputs[index++];
    const torch::Tensor& next_n_action = inputs[index++];
    const torch::Tensor& reward = inputs[index++];
    const torch::Tensor& done = inputs[index++];
    const torch::Tensor& weight = inputs[index++];
    const torch::Tensor& value_gamma = inputs[index++];
    index = 0;
    torch::Tensor& loss = outputs[index++];
    torch::Tensor& td_err = outputs[index++];
    torch::Tensor& bellman_err_buf = outputs[index++];
    torch::Tensor& quantile_huber_loss_buf = outputs[index++];
    torch::Tensor& grad_buf = outputs[index++];

    // set zero for atomic add
    checkCudaErr(cudaMemsetAsync(loss.data_ptr<float>(), 0, sizeof(float)));

    const unsigned int batch_size = q.size(0);
    const unsigned int action_dim = q.size(1);
    const unsigned int tau = q.size(2);
    const unsigned int time_step = reward.size(0);
 
    {
    dim3 block_size = {DEFAULT_WARP_NUM * WARP_SIZE, 1, 1};
    dim3 grid_size = {(tau + block_size.x - 1) / block_size.x, batch_size, 1};

    bellmanErrorKernel<<<grid_size, block_size>>>(
            tau, time_step, batch_size, action_dim, gamma,
            q.data_ptr<float>(), next_n_q.data_ptr<float>(),
            action.data_ptr<int64_t>(), next_n_action.data_ptr<int64_t>(),
            reward.data_ptr<float>(), done.data_ptr<float>(),
            value_gamma.data_ptr<float>(), bellman_err_buf.data_ptr<float>());
    }

    {
        dim3 block_size = {WARP_SIZE, DEFAULT_WARP_NUM, 1};
        dim3 grid_size = {(tau + block_size.x - 1) / block_size.x, (tau + block_size.y - 1) / block_size.y, batch_size};

        smoothL1LossKernel<<<grid_size, block_size>>> (
                tau, batch_size,
                bellman_err_buf.data_ptr<float>(), quantile_huber_loss_buf.data_ptr<float>(), grad_buf.data_ptr<float>());
    }

    {
        dim3 block_size = {DEFAULT_WARP_NUM * WARP_SIZE, 1, 1};
        dim3 grid_size = {batch_size, 1, 1};

        lossKernel<<<grid_size, block_size>>> (
                tau, batch_size,
                quantile_huber_loss_buf.data_ptr<float>(), weight.data_ptr<float>(),
                td_err.data_ptr<float>(), loss.data_ptr<float>());
    }
}

void QRDQNNStepTDErrorBackward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs) {

    unsigned int index = 0;
    const torch::Tensor& grad_loss = inputs[index++];
    const torch::Tensor& grad_buf = inputs[index++];
    const torch::Tensor& weight = inputs[index++];
    const torch::Tensor& action = inputs[index++];
    index = 0;
    torch::Tensor& grad_q = outputs[index++];

    const unsigned int batch_size = grad_q.size(0);
    const unsigned int action_dim = grad_q.size(1);
    const unsigned int tau = grad_q.size(2);

    // set zero
    checkCudaErr(cudaMemsetAsync(grad_q.data_ptr<float>(), 0, tau * batch_size * action_dim * sizeof(float)));

    dim3 block_size = {DEFAULT_WARP_NUM * WARP_SIZE, 1, 1};
    dim3 grid_size = {(tau + block_size.x - 1) / block_size.x, batch_size, 1};
    backwardKernel<<<grid_size, block_size>>>(
            tau, batch_size, action_dim,
            grad_loss.data_ptr<float>(), grad_buf.data_ptr<float>(),
            weight.data_ptr<float>(), action.data_ptr<int64_t>(),
            grad_q.data_ptr<float>());
}

}  // namespace cuda
}  // namespace rll
}  // namespace hpc