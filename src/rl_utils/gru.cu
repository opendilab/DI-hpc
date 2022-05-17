#include "hpc/rll/cuda/rl_utils/entry.h"
#include "hpc/rll/cuda/rl_utils/gru_kernel.h"

namespace hpc {
namespace rll {
namespace cuda {
void GRUForward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs,
    int TB,
    int input_dim) {

    unsigned int index = 0;
    const torch::Tensor& x = inputs[index++];
    const torch::Tensor& Wzy = inputs[index++];
    const torch::Tensor& Uzx = inputs[index++];
    const torch::Tensor& Wgy = inputs[index++];
    const torch::Tensor& Ugrx = inputs[index++];
    const torch::Tensor& bg = inputs[index++];

    index = 0;
    torch::Tensor& g = outputs[index++];
    torch::Tensor& h = outputs[index++];
    torch::Tensor& z = outputs[index++];


    unsigned int block_size = 256;
    unsigned int grid_size = TB;
    GRUPartialKernel<<<grid_size, block_size>>>(
            input_dim, (float*)(x.data_ptr()),
            (float*)(Wzy.data_ptr()), (float*)(Uzx.data_ptr()), (float*)(Wgy.data_ptr()), 
            (float*)(Ugrx.data_ptr()), (float*)(bg.data_ptr()), (float*)(g.data_ptr()), (float*)(h.data_ptr()), (float*)(z.data_ptr()));

}

void GRUBackward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs,
    int TB,
    int input_dim) {

    unsigned int index = 0;
    const torch::Tensor& grad_g = inputs[index++];
    const torch::Tensor& h = inputs[index++];
    const torch::Tensor& z = inputs[index++];
    const torch::Tensor& x = inputs[index++];

    index = 0;
    torch::Tensor& grad_Wzy = outputs[index++];
    torch::Tensor& grad_Uzx = outputs[index++];
    torch::Tensor& grad_Wgy = outputs[index++];
    torch::Tensor& grad_Ugrx = outputs[index++];
    torch::Tensor& grad_bg = outputs[index++];


    unsigned int block_size = 256;
    unsigned int grid_size = input_dim;
    GRUBackwardKernel<<<grid_size, block_size>>>(
            TB, input_dim, (float*)(grad_g.data_ptr()), (float*)(h.data_ptr()), (float*)(z.data_ptr()), (float*)(x.data_ptr()),
            (float*)(grad_Wzy.data_ptr()), (float*)(grad_Uzx.data_ptr()), (float*)(grad_Wgy.data_ptr()), (float*)(grad_Ugrx.data_ptr()), (float*)(grad_bg.data_ptr()));

}

}  // namespace cuda
}  // namespace rll
}  // namespace hpc
