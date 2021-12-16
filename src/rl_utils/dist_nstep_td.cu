#include "hpc/rll/cuda/rl_utils/entry.h"
#include "hpc/rll/cuda/rl_utils/dist_nstep_td_kernel.h"

namespace hpc {
namespace rll {
namespace cuda {

void DistNStepTdForward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs,
    float gamma,
    float v_min,
    float v_max) {

    unsigned int index = 0;
    const torch::Tensor& dist = inputs[index++];
    const torch::Tensor& next_n_dist = inputs[index++];
    const torch::Tensor& action = inputs[index++];
    const torch::Tensor& next_n_action = inputs[index++];
    const torch::Tensor& reward = inputs[index++];
    const torch::Tensor& done = inputs[index++];
    const torch::Tensor& weight = inputs[index++];
    index = 0;
    torch::Tensor& td_err = outputs[index++];
    torch::Tensor& loss = outputs[index++];
    torch::Tensor& buf = outputs[index++];

    // set zero for atomic add
    checkCudaErr(cudaMemsetAsync(loss.data_ptr(), 0, sizeof(float) * loss.numel()));
    checkCudaErr(cudaMemsetAsync(buf.data_ptr(), 0, sizeof(float) * buf.numel()));

    const unsigned int time_step = reward.size(0);
    const unsigned int batch_size = dist.size(0);
    const unsigned int action_dim = dist.size(1);
    const unsigned int n_atom = dist.size(2);

    // buf0: B for reward x fp reward_factor
    // buf1: (B * n_atom) for fp proj_dist and bp grad
    float* buf0 = (float*)(buf.data_ptr());
    float* buf1 = (float*)(buf.data_ptr()) + batch_size;

    {
        unsigned int block_size = DEFAULT_WARP_NUM * WARP_SIZE;
        unsigned int grid_size = (batch_size + block_size - 1) / block_size;
        distNStepTdRewardKernel<<<grid_size, block_size>>>(
                time_step, batch_size, gamma, (float*)(reward.data_ptr()), buf0);
    }

    {
        float gamma_nstep = 1.f;
        for (int t = 0; t < time_step; t++)
            gamma_nstep *= gamma;
        float delta = (v_max - v_min) / (n_atom - 1);

        dim3 block_size = {DEFAULT_WARP_NUM * WARP_SIZE, 1, 1};
        dim3 grid_size = {(n_atom + block_size.x - 1) / block_size.x, batch_size, 1};
        distNStepTdProjKernel<<<grid_size, block_size>>>(
                batch_size, action_dim, n_atom, gamma_nstep, v_min, v_max, delta,
                (float*)(next_n_dist.data_ptr()), (int64_t*)(next_n_action.data_ptr()),
                buf0, (float*)(done.data_ptr()), buf1);
    }

    {
        dim3 block_size = {DEFAULT_WARP_NUM * WARP_SIZE, 1, 1};
        dim3 grid_size = {(n_atom + block_size.x - 1) / block_size.x, batch_size, 1};
        distNStepTdLossKernel<<<grid_size, block_size>>>(
                batch_size, action_dim, n_atom, (float*)(dist.data_ptr()), (int64_t*)(action.data_ptr()),
                (const float*)buf1, (float*)(weight.data_ptr()),
                (float*)(td_err.data_ptr()), (float*)(loss.data_ptr()), buf1);
    }
}

void DistNStepTdBackward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs) {

    unsigned int index = 0;
    const torch::Tensor& grad_loss = inputs[index++];
    const torch::Tensor& buf = inputs[index++];
    const torch::Tensor& action = inputs[index++];
    index = 0;
    torch::Tensor& grad_dist = outputs[index++];

    const unsigned int batch_size = grad_dist.size(0);
    const unsigned int action_dim = grad_dist.size(1);
    const unsigned int n_atom = grad_dist.size(2);

    // buf0: B for reward x fp reward_factor
    // buf1: (B * n_atom) for fp proj_dist and bp grad, here used for bp grad
    float* grad_buf = (float*)(buf.data_ptr()) + batch_size;

    unsigned int block_size = DEFAULT_WARP_NUM * WARP_SIZE;
    unsigned int grid_size = (batch_size * action_dim * n_atom + block_size - 1) / block_size;
    distNStepTdBackwardKernel<<<grid_size, block_size>>>(
            batch_size, action_dim, n_atom, (float*)(grad_loss.data_ptr()), grad_buf,
            (int64_t*)(action.data_ptr()), (float*)(grad_dist.data_ptr()));
}

}  // namespace cuda
}  // namespace rll
}  // namespace hpc

