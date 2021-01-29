#include "hpc/rll/cuda/rl_utils/entry.h"
#include "hpc/rll/cuda/rl_utils/ppo_kernel.h"

namespace hpc {
namespace rll {
namespace cuda {

void PPOForward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs,
    bool use_value_clip,
    float clip_ratio,
    float dual_clip) {

    unsigned int index = 0;
    const torch::Tensor& logits_new = inputs[index++];
    const torch::Tensor& logits_old = inputs[index++];
    const torch::Tensor& action = inputs[index++];
    const torch::Tensor& value_new = inputs[index++];
    const torch::Tensor& value_old = inputs[index++];
    const torch::Tensor& adv = inputs[index++];
    const torch::Tensor& return_ = inputs[index++];
    const torch::Tensor& weight = inputs[index++];

    index = 0;
    torch::Tensor& logits_new_prob = outputs[index++];
    torch::Tensor& logits_new_entropy = outputs[index++];
    torch::Tensor& logits_new_grad_logits = outputs[index++];
    torch::Tensor& logits_new_grad_prob = outputs[index++];
    torch::Tensor& logits_new_grad_entropy = outputs[index++];
    torch::Tensor& logits_old_prob = outputs[index++];
    torch::Tensor& grad_policy_loss_buf = outputs[index++];
    torch::Tensor& grad_value_loss_buf = outputs[index++];
    torch::Tensor& grad_entropy_loss_buf = outputs[index++];
    torch::Tensor& policy_loss = outputs[index++];
    torch::Tensor& value_loss = outputs[index++];
    torch::Tensor& entropy_loss = outputs[index++];
    torch::Tensor& approx_kl = outputs[index++];
    torch::Tensor& clipfrac = outputs[index++];

    checkCudaErr(cudaMemsetAsync(policy_loss.data_ptr<float>(), 0, sizeof(float)));
    checkCudaErr(cudaMemsetAsync(value_loss.data_ptr<float>(), 0, sizeof(float)));
    checkCudaErr(cudaMemsetAsync(entropy_loss.data_ptr<float>(), 0, sizeof(float)));
    checkCudaErr(cudaMemsetAsync(approx_kl.data_ptr<float>(), 0, sizeof(float)));
    checkCudaErr(cudaMemsetAsync(clipfrac.data_ptr<float>(), 0, sizeof(float)));

    const unsigned int batch_size = logits_new.size(0);
    const unsigned int num_output = logits_new.size(1);
    {
        unsigned int block_size = DEFAULT_WARP_NUM * WARP_SIZE;
        unsigned int grid_size = batch_size;
        categoricalProbEntropy<<<grid_size, block_size>>>(
                num_output, logits_new.data_ptr<float>(), action.data_ptr<int64_t>(),
                logits_new_prob.data_ptr<float>(), logits_new_entropy.data_ptr<float>(),
                logits_new_grad_logits.data_ptr<float>(), logits_new_grad_prob.data_ptr<float>(),
                logits_new_grad_entropy.data_ptr<float>());
        categoricalProb<<<grid_size, block_size>>>(
                num_output, logits_old.data_ptr<float>(), action.data_ptr<int64_t>(), logits_old_prob.data_ptr<float>());
    }
    {
        unsigned int block_size = DEFAULT_WARP_NUM * WARP_SIZE;
        unsigned int grid_size = (batch_size + block_size - 1) / block_size;
        ppoLoss<<<grid_size, block_size>>>(
                batch_size, value_new.data_ptr<float>(), value_old.data_ptr<float>(),
                logits_new_prob.data_ptr<float>(), logits_old_prob.data_ptr<float>(), logits_new_entropy.data_ptr<float>(),
                adv.data_ptr<float>(), return_.data_ptr<float>(), weight.data_ptr<float>(),
                use_value_clip, clip_ratio, dual_clip,
                policy_loss.data_ptr<float>(), value_loss.data_ptr<float>(), entropy_loss.data_ptr<float>(),
                approx_kl.data_ptr<float>(), clipfrac.data_ptr<float>(),
                grad_policy_loss_buf.data_ptr<float>(), grad_value_loss_buf.data_ptr<float>(), grad_entropy_loss_buf.data_ptr<float>());
    }
}

void PPOBackward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs) {

    unsigned int index = 0;
    const torch::Tensor& grad_policy_loss = inputs[index++];
    const torch::Tensor& grad_value_loss = inputs[index++];
    const torch::Tensor& grad_entropy_loss = inputs[index++];
    const torch::Tensor& grad_policy_loss_buf = inputs[index++];
    const torch::Tensor& grad_value_loss_buf = inputs[index++];
    const torch::Tensor& grad_entropy_loss_buf = inputs[index++];
    const torch::Tensor& logits_new_grad_logits = inputs[index++];
    const torch::Tensor& logits_new_grad_prob = inputs[index++];
    const torch::Tensor& logits_new_grad_entropy = inputs[index++];

    index = 0;
    torch::Tensor& grad_value = outputs[index++];
    torch::Tensor& grad_logits_new = outputs[index++];

    const unsigned int batch_size = grad_logits_new.size(0);
    const unsigned int num_output = grad_logits_new.size(1);
    {
        unsigned int block_size = DEFAULT_WARP_NUM * WARP_SIZE;
        unsigned int grid_size = (batch_size + block_size - 1) / block_size;
        ppoBackwardValueNew<<<grid_size, block_size>>>(
                batch_size, grad_value_loss.data_ptr<float>(), grad_value_loss_buf.data_ptr<float>(), grad_value.data_ptr<float>());
    }
    {
        unsigned int block_size = DEFAULT_WARP_NUM * WARP_SIZE;
        unsigned int grid_size = batch_size;
        ppoBackwardLogitsNew<<<grid_size, block_size>>>(
                batch_size, num_output, grad_policy_loss.data_ptr<float>(), grad_entropy_loss.data_ptr<float>(),
                grad_policy_loss_buf.data_ptr<float>(), grad_entropy_loss_buf.data_ptr<float>(),
                logits_new_grad_logits.data_ptr<float>(), logits_new_grad_prob.data_ptr<float>(), logits_new_grad_entropy.data_ptr<float>(),
                grad_logits_new.data_ptr<float>());
    }
}

}  // namespace cuda
}  // namespace rll
}  // namespace hpc

