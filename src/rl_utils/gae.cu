#include "hpc/rll/cuda/rl_utils/entry.h"
#include "hpc/rll/cuda/rl_utils/gae_kernel.h"

namespace hpc {
namespace rll {
namespace cuda {

void GaeForward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs,
    float gamma,
    float lambda) {

    unsigned int index = 0;
    const torch::Tensor& value = inputs[index++];
    const torch::Tensor& reward = inputs[index++];
    index = 0;
    torch::Tensor& adv = outputs[index++];

	const unsigned int time_step = reward.size(0);
	const unsigned int batch_size = reward.size(1);

    unsigned int block_size = 1 * WARP_SIZE; // single warp to utilize more blocks
    unsigned int grid_size = (batch_size + block_size - 1) / block_size;
    gaeForwardKernel<<<grid_size, block_size>>>(
            time_step, batch_size, gamma, lambda,
            (float*)(value.data_ptr()), (float*)(reward.data_ptr()), (float*)(adv.data_ptr()));
}

}  // namespace cuda
}  // namespace rll
}  // namespace hpc
