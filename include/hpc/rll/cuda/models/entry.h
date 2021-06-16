#ifndef HPC_RLL_CUDA_NETWORK_H_
#define HPC_RLL_CUDA_NETWORK_H_

#include "hpc/rll/cuda/common.h"

namespace hpc {
namespace rll {
namespace cuda {

void actor_critic_update_ae(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs);

void actor_critic_lstm_activation(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs);

void actor_critic_pre_sample(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs);

}  // namespace cuda
}  // namespace rll
}  // namespace hpc

#endif // HPC_RLL_CUDA_NETWORK_H_
