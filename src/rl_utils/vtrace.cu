#include "hpc/rll/cuda/rl_utils/entry.h"
#include "hpc/rll/cuda/rl_utils/vtrace_kernel.h"

namespace hpc {
namespace rll {
namespace cuda {

void VTraceForward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs,
    float gamma,
    float lambda,
    float rho_clip_ratio,
    float c_clip_ratio,
    float rho_pg_clip_ratio) {

    unsigned int index = 0;
    const torch::Tensor& target_output = inputs[index++];
    const torch::Tensor& behaviour_output = inputs[index++];
    const torch::Tensor& action = inputs[index++];
    const torch::Tensor& value = inputs[index++];
    const torch::Tensor& reward = inputs[index++];
    const torch::Tensor& weight = inputs[index++];

    index = 0;
    torch::Tensor& target_output_prob = outputs[index++];
    torch::Tensor& target_output_entropy = outputs[index++];
    torch::Tensor& target_output_grad_logits = outputs[index++];
    torch::Tensor& target_output_grad_prob = outputs[index++];
    torch::Tensor& target_output_grad_entropy = outputs[index++];
    torch::Tensor& behaviour_output_prob = outputs[index++];
    torch::Tensor& is = outputs[index++];
    torch::Tensor& ret = outputs[index++];
    torch::Tensor& adv = outputs[index++];
    torch::Tensor& pg_loss = outputs[index++];
    torch::Tensor& value_loss = outputs[index++];
    torch::Tensor& entropy_loss = outputs[index++];

    checkCudaErr(cudaMemsetAsync((float*)(pg_loss.data_ptr()), 0, sizeof(float)));
    checkCudaErr(cudaMemsetAsync((float*)(value_loss.data_ptr()), 0, sizeof(float)));
    checkCudaErr(cudaMemsetAsync((float*)(entropy_loss.data_ptr()), 0, sizeof(float)));

    const unsigned int time_step = target_output.size(0);
    const unsigned int batch_size = target_output.size(1);
    const unsigned int num_output = target_output.size(2);
    {
        unsigned int block_size = DEFAULT_WARP_NUM * WARP_SIZE;
        unsigned int grid_size = time_step * batch_size;
        categoricalTarget<<<grid_size, block_size>>>(num_output,
                (float*)(target_output.data_ptr()), (int64_t*)(action.data_ptr()),
                (float*)(target_output_prob.data_ptr()), (float*)(target_output_entropy.data_ptr()),
                (float*)(target_output_grad_logits.data_ptr()), (float*)(target_output_grad_prob.data_ptr()),
                (float*)(target_output_grad_entropy.data_ptr()));
        categoricalBehaviour<<<grid_size, block_size>>>(num_output,
                (float*)(behaviour_output.data_ptr()), (int64_t*)(action.data_ptr()), (float*)(behaviour_output_prob.data_ptr()));
    }
    {
        unsigned int block_size = DEFAULT_WARP_NUM * WARP_SIZE;
        unsigned int grid_size = (time_step * batch_size + block_size - 1) / block_size;
        computeImportanceWeights<<<grid_size, block_size>>>(time_step, batch_size,
                (float*)(target_output_prob.data_ptr()), (float*)(behaviour_output_prob.data_ptr()), (float*)(is.data_ptr()));
    }
    {
        unsigned int block_size = DEFAULT_WARP_NUM * WARP_SIZE;
        unsigned int grid_size = (batch_size + block_size - 1) / block_size;
        vtraceNStepReturn<<<grid_size, block_size>>>(
                time_step, batch_size, gamma, lambda, rho_clip_ratio, c_clip_ratio, 
                (float*)(is.data_ptr()), (float*)(reward.data_ptr()), (float*)(value.data_ptr()), (float*)(ret.data_ptr()));
    }
    {
        unsigned int block_size = DEFAULT_WARP_NUM * WARP_SIZE;
        unsigned int grid_size = (time_step * batch_size + block_size - 1) / block_size;
        vtraceAdvantage<<<grid_size, block_size>>>(
                time_step, batch_size, gamma, rho_pg_clip_ratio,
                (float*)(is.data_ptr()), (float*)(reward.data_ptr()), (float*)(value.data_ptr()),
                (float*)(ret.data_ptr()), (float*)(adv.data_ptr()));
    }
    {
        unsigned int block_size = DEFAULT_WARP_NUM * WARP_SIZE;
        unsigned int grid_size = (time_step * batch_size + block_size - 1) / block_size;
        vtraceLoss<<<grid_size, block_size>>>(time_step, batch_size,
                (float*)(value.data_ptr()), (float*)(target_output_prob.data_ptr()), (float*)(target_output_entropy.data_ptr()),
                (float*)(ret.data_ptr()), (float*)(adv.data_ptr()), (float*)(weight.data_ptr()),
                (float*)(pg_loss.data_ptr()), (float*)(value_loss.data_ptr()), (float*)(entropy_loss.data_ptr()));
    }
}

void VTraceBackward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs) {

    unsigned int index = 0;
    const torch::Tensor& grad_pg_loss = inputs[index++];
    const torch::Tensor& grad_value_loss = inputs[index++];
    const torch::Tensor& grad_entropy_loss = inputs[index++];
    const torch::Tensor& value = inputs[index++];
    const torch::Tensor& action = inputs[index++];
    const torch::Tensor& weight = inputs[index++];
    const torch::Tensor& ret = inputs[index++];
    const torch::Tensor& adv = inputs[index++];
    const torch::Tensor& target_output_grad_logits = inputs[index++];
    const torch::Tensor& target_output_grad_prob = inputs[index++];
    const torch::Tensor& target_output_grad_entropy = inputs[index++];

    index = 0;
    torch::Tensor& grad_value = outputs[index++];
    torch::Tensor& grad_target_output = outputs[index++];

    const unsigned int time_step = grad_target_output.size(0);
    const unsigned int batch_size = grad_target_output.size(1);
    const unsigned int num_output = grad_target_output.size(2);
    {
        unsigned int block_size = DEFAULT_WARP_NUM * WARP_SIZE;
        unsigned int grid_size = ((time_step + 1) * batch_size + block_size - 1) / block_size;
        vtraceBackwardValue<<<grid_size, block_size>>>(time_step, batch_size,
                (float*)(grad_value_loss.data_ptr()), (float*)(value.data_ptr()), (float*)(ret.data_ptr()),
                (float*)(weight.data_ptr()), (float*)(grad_value.data_ptr()));
    }
    {
        unsigned int block_size = DEFAULT_WARP_NUM * WARP_SIZE;
        unsigned int grid_size = time_step * batch_size;
        vtraceBackwardTargetOutput<<<grid_size, block_size>>>(
                time_step, batch_size, num_output,
                (float*)(grad_entropy_loss.data_ptr()), (float*)(grad_pg_loss.data_ptr()),
                (float*)(target_output_grad_logits.data_ptr()),
                (float*)(target_output_grad_entropy.data_ptr()),
                (float*)(target_output_grad_prob.data_ptr()),
                (float*)(adv.data_ptr()), (float*)(weight.data_ptr()), (float*)(grad_target_output.data_ptr()));
    }
}

}  // namespace cuda
}  // namespace rll
}  // namespace hpc

