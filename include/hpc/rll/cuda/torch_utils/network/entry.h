#ifndef HPC_RLL_CUDA_NETWORK_H_
#define HPC_RLL_CUDA_NETWORK_H_

#include "hpc/rll/cuda/common.h"

namespace hpc {
namespace rll {
namespace cuda {

// lstm
void LstmForward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs,
    float dropout_threshold);

void LstmBackward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs,
    float dropout_threshold);

// scatter_connection
void ScatterConnectionForward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs,
    const char* scatter_type);

void ScatterConnectionBackward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs);

}  // namespace cuda
}  // namespace rll
}  // namespace hpc

#endif // HPC_RLL_CUDA_NETWORK_H_
