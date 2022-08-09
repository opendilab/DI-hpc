#include "hpc/rll/cuda/rl_utils/entry.h"
#include "hpc/rll/cuda/rl_utils/soft_argmax_kernel.h"
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

void SOFTARGMAXForward(
    std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs) {

    unsigned int index = 0;
    torch::Tensor& x = inputs[index++];


    index = 0;
    torch::Tensor& softmax_x = outputs[index++];
    torch::Tensor& res = outputs[index++];


    const unsigned int B = x.size(0);
    const unsigned int H = x.size(2);
    const unsigned int W = x.size(3);

    {
        unsigned int block_size = GetBlockSize(H * W);
        unsigned int grid_size = B;
        SoftmaxScoreKernel<<<grid_size, block_size>>>((float*)(x.data_ptr()), (float*)(softmax_x.data_ptr()),  H * W);
        MulSumKernel<<<grid_size, block_size>>>(
                H, W, (float*)(softmax_x.data_ptr()), (float*)(res.data_ptr()));

    }

}

void SOFTARGMAXBackward(
    std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs) {

    unsigned int index = 0;
    torch::Tensor& grad_res = inputs[index++];
    torch::Tensor& softmax_x = inputs[index++];


    index = 0;
    torch::Tensor& grad_x = outputs[index++];


    const unsigned int B = softmax_x.size(0);
    const unsigned int H = softmax_x.size(2);
    const unsigned int W = softmax_x.size(3);

    {
        unsigned int block_size = GetBlockSize(H * W);
        unsigned int grid_size = B;
        BackwardKernel<<<grid_size, block_size>>>(
                H, W, (float*)(softmax_x.data_ptr()), (float*)(grad_res.data_ptr()), (float*)(grad_x.data_ptr()));

    }

}



}  // namespace cuda
}  // namespace rll
}  // namespace hpc