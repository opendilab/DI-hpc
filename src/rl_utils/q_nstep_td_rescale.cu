#include "hpc/rll/cuda/rl_utils/entry.h"
#include "hpc/rll/cuda/rl_utils/q_nstep_td_rescale_kernel.h"

namespace hpc {
namespace rll {
namespace cuda {

void QNStepTdRescaleForward(
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
    index = 0;
    torch::Tensor& td_err = outputs[index++];
    torch::Tensor& loss = outputs[index++];
    torch::Tensor& grad_buf = outputs[index++];

    // set zero for atomic add
    checkCudaErr(cudaMemsetAsync((float*)(loss.data_ptr()), 0, sizeof(float)));

	const unsigned int time_step = reward.size(0);
	const unsigned int batch_size = q.size(0);
	const unsigned int num_output = q.size(1);

    unsigned int block_size = DEFAULT_WARP_NUM * WARP_SIZE;
    unsigned int grid_size = (batch_size + block_size - 1) / block_size;
    qNStepTdRescaleForwardKernel<<<grid_size, block_size>>>(
            time_step, batch_size, num_output, gamma,
            (float*)(q.data_ptr()), (float*)(next_n_q.data_ptr()),
            (int64_t*)(action.data_ptr()), (int64_t*)(next_n_action.data_ptr()),
            (float*)(reward.data_ptr()), (float*)(done.data_ptr()), (float*)(weight.data_ptr()),
            (float*)(td_err.data_ptr()), (float*)(loss.data_ptr()), (float*)(grad_buf.data_ptr()));
}

void QNStepTdRescaleBackward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs) {

    unsigned int index = 0;
    const torch::Tensor& grad_loss = inputs[index++];
    const torch::Tensor& grad_buf = inputs[index++];
    const torch::Tensor& action = inputs[index++];
    index = 0;
    torch::Tensor& grad_q = outputs[index++];

	const unsigned int batch_size = grad_q.size(0);
	const unsigned int num_output = grad_q.size(1);

    dim3 block_size = {DEFAULT_WARP_NUM * WARP_SIZE, 1, 1};
    dim3 grid_size = {(num_output + block_size.x - 1) / block_size.x, batch_size, 1};
    qNStepTdRescaleBackwardKernel<<<grid_size, block_size>>>(
            batch_size, num_output,
            (float*)(grad_loss.data_ptr()), (float*)(grad_buf.data_ptr()),
            (int64_t*)(action.data_ptr()), (float*)(grad_q.data_ptr()));
}

}  // namespace cuda
}  // namespace rll
}  // namespace hpc
