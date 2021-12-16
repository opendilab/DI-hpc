#include "hpc/rll/cuda/rl_utils/entry.h"
#include "hpc/rll/cuda/rl_utils/upgo_kernel.h"

namespace hpc {
namespace rll {
namespace cuda {

void UpgoForward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs) {

    unsigned int index = 0;
    const torch::Tensor& target_output = inputs[index++];
    const torch::Tensor& rho = inputs[index++];
    const torch::Tensor& action = inputs[index++];
    const torch::Tensor& reward = inputs[index++];
    const torch::Tensor& value = inputs[index++];
    index = 0;
    torch::Tensor& advantage = outputs[index++];
    torch::Tensor& metric = outputs[index++];
    torch::Tensor& loss = outputs[index++];
    torch::Tensor& grad_buf = outputs[index++];

    checkCudaErr(cudaMemsetAsync((float*)(loss.data_ptr()), 0, sizeof(float)));

    const unsigned int time_step = target_output.size(0);
    const unsigned int batch_size = target_output.size(1);
    const unsigned int num_output = target_output.size(2);
    {
        unsigned int block_size = DEFAULT_WARP_NUM * WARP_SIZE;
        unsigned int grid_size = (batch_size + block_size - 1) / block_size;
        upgoAdvantageKernel<<<grid_size, block_size>>>(time_step, batch_size,
                (float*)(rho.data_ptr()), (float*)(reward.data_ptr()), (float*)(value.data_ptr()), (float*)(advantage.data_ptr()));
    }
    {
        unsigned int block_size = DEFAULT_WARP_NUM * WARP_SIZE;
        unsigned int grid_size = time_step * batch_size;
        crossEntropyKernel<<<grid_size, block_size>>>(num_output,
                (float*)(target_output.data_ptr()), (int64_t*)(action.data_ptr()), (float*)(metric.data_ptr()), (float*)(grad_buf.data_ptr()));
    }
    {
        unsigned int block_size = DEFAULT_WARP_NUM * WARP_SIZE;
        unsigned int grid_size = (time_step * batch_size + block_size - 1) / block_size;
        upgoLossKernel<<<grid_size, block_size>>>(time_step, batch_size,
                (float*)(advantage.data_ptr()), (float*)(metric.data_ptr()), (float*)(loss.data_ptr()));
    }
}

void UpgoBackward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs) {

    unsigned int index = 0;
    const torch::Tensor& grad_loss = inputs[index++];
    const torch::Tensor& grad_buf = inputs[index++];
    const torch::Tensor& advantage = inputs[index++];
    index = 0;
    torch::Tensor& grad_target_output = outputs[index++];

	const unsigned int time_step = grad_target_output.size(0);
	const unsigned int batch_size = grad_target_output.size(1);
	const unsigned int num_output = grad_target_output.size(2);

    unsigned int block_size = DEFAULT_WARP_NUM * WARP_SIZE;
    unsigned int grid_size = (time_step * batch_size * num_output + block_size - 1) / block_size;
    upgoBackwardKernel<<<grid_size, block_size>>>(time_step, batch_size, num_output,
            (float*)(grad_loss.data_ptr()), (float*)(grad_buf.data_ptr()),
            (float*)(advantage.data_ptr()), (float*)(grad_target_output.data_ptr()));
}

}  // namespace cuda
}  // namespace rll
}  // namespace hpc
