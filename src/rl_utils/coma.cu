#include "hpc/rll/cuda/rl_utils/entry.h"
#include "hpc/rll/cuda/rl_utils/coma_kernel.h"

namespace hpc {
namespace rll {
namespace cuda {



inline int GetBlockSize(const int n, const int max_size = 1024) {
    int ret = 32;
    while(ret < n && ret < max_size) {
        ret <<= 1;
    }
    return ret;
}

void COMAForward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs,
    float gamma,
    float lambda_) {

    unsigned int index = 0;
    const torch::Tensor& logit = inputs[index++];
    const torch::Tensor& action = inputs[index++];
    const torch::Tensor& q_value = inputs[index++];
    const torch::Tensor& target_q_value = inputs[index++];
    const torch::Tensor& reward = inputs[index++];
    const torch::Tensor& weight = inputs[index++];


    index = 0;
    torch::Tensor& reward_new = outputs[index++];
    torch::Tensor& q_taken = outputs[index++];
    torch::Tensor& target_q_taken = outputs[index++];
    torch::Tensor& prob = outputs[index++];
    torch::Tensor& adv = outputs[index++];
    torch::Tensor& entropy = outputs[index++];
    torch::Tensor& return_ = outputs[index++];
    torch::Tensor& logits_grad_logits = outputs[index++];
    torch::Tensor& logits_grad_prob = outputs[index++];
    torch::Tensor& logits_grad_adv = outputs[index++];
    torch::Tensor& logits_grad_entropy = outputs[index++];
    torch::Tensor& qvalue_grad_adv = outputs[index++];
    torch::Tensor& grad_policy_loss_buf = outputs[index++];
    torch::Tensor& grad_value_loss_buf = outputs[index++];
    torch::Tensor& grad_entropy_loss_buf = outputs[index++];
    torch::Tensor& policy_loss = outputs[index++];
    torch::Tensor& value_loss = outputs[index++];
    torch::Tensor& entropy_loss = outputs[index++];


    checkCudaErr(cudaMemsetAsync((float*)(policy_loss.data_ptr()), 0, sizeof(float)));
    checkCudaErr(cudaMemsetAsync((float*)(value_loss.data_ptr()), 0, sizeof(float)));
    checkCudaErr(cudaMemsetAsync((float*)(entropy_loss.data_ptr()), 0, sizeof(float)));

    const unsigned int T = logit.size(0);
    const unsigned int B = logit.size(1);
    const unsigned int A = logit.size(2);
    const unsigned int N = logit.size(3);
    {
        unsigned int block_size = DEFAULT_WARP_NUM * WARP_SIZE;
        unsigned int grid_size = T;
        COMAGather<<<grid_size, block_size>>>(
                N, B, A, (float*)(q_value.data_ptr()), (float*)(target_q_value.data_ptr()), (int64_t*)(action.data_ptr()),
                (float*)(q_taken.data_ptr()), (float*)(target_q_taken.data_ptr()), (float*)(reward.data_ptr()), (float*)(reward_new.data_ptr()));
    }

    {
        unsigned int block_size = DEFAULT_WARP_NUM * WARP_SIZE;
        unsigned int grid_size = T * B * A;
        ProbEntropyAdv<<<grid_size, block_size>>>(
                N, (float*)(logit.data_ptr()), (int64_t*)(action.data_ptr()), (float*)(q_value.data_ptr()), (float*)(q_taken.data_ptr()),
                (float*)(prob.data_ptr()), (float*)(adv.data_ptr()), (float*)(entropy.data_ptr()),
                (float*)(logits_grad_logits.data_ptr()), (float*)(logits_grad_prob.data_ptr()), (float*)(logits_grad_adv.data_ptr()),
                (float*)(logits_grad_entropy.data_ptr()), (float*)(qvalue_grad_adv.data_ptr()));

    }

    {
        unsigned int block_size = DEFAULT_WARP_NUM * WARP_SIZE;
        unsigned int grid_size = (B*A + block_size - 1) / block_size;;
        GeneralizedLambdaReturns<<<grid_size, block_size>>>(
                T, B*A, (float*)(target_q_taken.data_ptr()), (float*)(reward_new.data_ptr()), (float*)(return_.data_ptr()),
                gamma, lambda_);
    }

    {
        unsigned int block_size = DEFAULT_WARP_NUM * WARP_SIZE;
        unsigned int grid_size = (T*B*A + block_size - 1) / block_size;
        COMALoss<<<grid_size, block_size>>>(
                T, B*A, (float*)(q_taken.data_ptr()), (float*)(prob.data_ptr()), (float*)(entropy.data_ptr()),
                (float*)(adv.data_ptr()), (float*)(return_.data_ptr()), (float*)(weight.data_ptr()),
                (float*)(policy_loss.data_ptr()), (float*)(value_loss.data_ptr()), (float*)(entropy_loss.data_ptr()),
                (float*)(grad_policy_loss_buf.data_ptr()), (float*)(grad_value_loss_buf.data_ptr()),
                (float*)(grad_entropy_loss_buf.data_ptr()));
    }
}

void COMABackward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs) {

    unsigned int index = 0;
    const torch::Tensor& grad_policy_loss = inputs[index++];
    const torch::Tensor& grad_value_loss = inputs[index++];
    const torch::Tensor& grad_entropy_loss = inputs[index++];
    const torch::Tensor& grad_policy_loss_buf = inputs[index++];
    const torch::Tensor& grad_value_loss_buf = inputs[index++];
    const torch::Tensor& grad_entropy_loss_buf = inputs[index++];
    const torch::Tensor& prob = inputs[index++];
    const torch::Tensor& adv = inputs[index++];
    const torch::Tensor& action = inputs[index++];
    const torch::Tensor& logits_grad_logits = inputs[index++];
    const torch::Tensor& logits_grad_prob = inputs[index++];
    const torch::Tensor& logits_grad_adv = inputs[index++];
    const torch::Tensor& logits_grad_entropy = inputs[index++];
    const torch::Tensor& qvalue_grad_adv = inputs[index++];

    index = 0;
    torch::Tensor& grad_q_value = outputs[index++];
    torch::Tensor& grad_logits = outputs[index++];

    const unsigned int T = grad_logits.size(0);
    const unsigned int B = grad_logits.size(1);
    const unsigned int A = grad_logits.size(2);
    const unsigned int N = grad_logits.size(3);

    
    unsigned int block_size = GetBlockSize(N);
    unsigned int grid_size = T * B * A;
    COMABackward<<<grid_size, block_size>>>(
            N, (float*)(grad_policy_loss.data_ptr()), (float*)(grad_value_loss.data_ptr()), (float*)(grad_entropy_loss.data_ptr()),
            (float*)(grad_policy_loss_buf.data_ptr()), (float*)(grad_value_loss_buf.data_ptr()), (float*)(grad_entropy_loss_buf.data_ptr()),
            (float*)(prob.data_ptr()), (float*)(adv.data_ptr()), (int64_t*)(action.data_ptr()), (float*)(logits_grad_logits.data_ptr()), (float*)(logits_grad_prob.data_ptr()),
            (float*)(logits_grad_adv.data_ptr()), (float*)(logits_grad_entropy.data_ptr()), (float*)(qvalue_grad_adv.data_ptr()),
            (float*)(grad_q_value.data_ptr()), (float*)(grad_logits.data_ptr()));
    
}

}  // namespace cuda
}  // namespace rll
}  // namespace hpc