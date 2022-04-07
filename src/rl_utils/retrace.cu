#include "hpc/rll/cuda/rl_utils/entry.h"
#include "hpc/rll/cuda/rl_utils/retrace_kernel.h"

namespace hpc {
namespace rll {
namespace cuda {

void RetraceForward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs,
    float gamma) {

    unsigned int index = 0;
    const torch::Tensor& q_values = inputs[index++];
    const torch::Tensor& v_pred = inputs[index++];
    const torch::Tensor& rewards = inputs[index++];
    const torch::Tensor& actions = inputs[index++];
    const torch::Tensor& weights = inputs[index++];
    const torch::Tensor& ratio = inputs[index++];

    index = 0;
    torch::Tensor& q_retraces = outputs[index++];
    torch::Tensor& q_gather = outputs[index++];
    torch::Tensor& ratio_gather = outputs[index++];
    torch::Tensor& tmp_retraces = outputs[index++];

    const unsigned int T = q_values.size(0) - 1;
    const unsigned int B = q_values.size(1);
    const unsigned int N = q_values.size(2);

    unsigned int block_size = DEFAULT_WARP_NUM * WARP_SIZE;
    unsigned int grid_size = T;
    gatherKernel<<<grid_size, block_size>>>(
                T, B, N, (float*)(q_gather.data_ptr()), (float*)(q_values.data_ptr()),
                (int64_t*)(actions.data_ptr()), (float*)(ratio_gather.data_ptr()),
                (float*)(ratio.data_ptr()));
    {
        unsigned int block_size = DEFAULT_WARP_NUM * WARP_SIZE;
        unsigned int grid_size = (B + block_size - 1) / block_size;
        retraceKernel<<<grid_size, block_size>>>(
                T, B, (float*)(v_pred.data_ptr()),
                (float*)(rewards.data_ptr()), (int64_t*)(actions.data_ptr()),
                (float*)(weights.data_ptr()),
                (float*)(q_retraces.data_ptr()), (float*)(q_gather.data_ptr()),
                (float*)(ratio_gather.data_ptr()), (float*)(tmp_retraces.data_ptr()), gamma);
    }
    
}



}  // namespace cuda
}  // namespace rll
}  // namespace hpc

