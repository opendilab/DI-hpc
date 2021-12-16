#include "hpc/rll/cuda/rl_utils/entry.h"
#include "hpc/rll/cuda/rl_utils/td_lambda_kernel.h"

namespace hpc {
namespace rll {
namespace cuda {

void TdLambdaForward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs,
    float gamma,
    float lambda) {

    unsigned int index = 0;
    const torch::Tensor& value = inputs[index++];
    const torch::Tensor& reward = inputs[index++];
    const torch::Tensor& weight = inputs[index++];
    index = 0;
    torch::Tensor& loss = outputs[index++];
    torch::Tensor& grad_buf = outputs[index++];

    checkCudaErr(cudaMemsetAsync((float*)(loss.data_ptr()), 0, sizeof(float)));

	const unsigned int time_step = reward.size(0);
	const unsigned int batch_size = reward.size(1);

    unsigned int block_size = 1 * WARP_SIZE; // in order to use as many sm processors as possible
    unsigned int grid_size = (batch_size + block_size - 1) / block_size;
    tdLambdaForwardKernel<<<grid_size, block_size>>>(
            time_step, batch_size, gamma, lambda,
            (float*)(value.data_ptr()), (float*)(reward.data_ptr()), (float*)(weight.data_ptr()),
            (float*)(loss.data_ptr()), (float*)(grad_buf.data_ptr()));
}

void TdLambdaBackward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs) {

    unsigned int index = 0;
    const torch::Tensor& grad_loss = inputs[index++];
    const torch::Tensor& grad_buf = inputs[index++];
    index = 0;
    torch::Tensor& grad_value = outputs[index++];

    const unsigned int time_step = grad_value.size(0) - 1;
	const unsigned int batch_size = grad_value.size(1);

    unsigned int block_size = DEFAULT_WARP_NUM * WARP_SIZE;
    unsigned int grid_size = ((time_step + 1) * batch_size + block_size - 1) / block_size;
    tdLambdaBackwardKernel<<<grid_size, block_size>>>(
            time_step, batch_size, (float*)(grad_loss.data_ptr()), (float*)(grad_buf.data_ptr()), (float*)(grad_value.data_ptr()));
}

}  // namespace cuda
}  // namespace rll
}  // namespace hpc
