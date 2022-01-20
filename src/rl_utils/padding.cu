#include "hpc/rll/cuda/rl_utils/entry.h"
#include "hpc/rll/cuda/rl_utils/padding_kernel.h"

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

template<typename T>
void Pad1D(const std::vector<torch::Tensor>& inputs, const int* shape, torch::Tensor new_x, torch::Tensor mask) {
    const int n = inputs.size();
    int max_shape = 0;
    for(int i = 0; i < n; i++) {
        if(max_shape < shape[i])
            max_shape = shape[i];
    }
    dim3 grid(n, 1, 1);
    dim3 block(GetBlockSize(max_shape));
    Pad1D_kernel<T><<<grid, block>>>(inputs, shape, (T*)(new_x.data_ptr()), (T*)(mask.data_ptr()), max_shape, 0);
}

template<typename T>
void Unpad1D(torch::Tensor inputs, const int* ori_shape, std::vector<torch::Tensor>& outputs) {
    const int n = outputs.size();
    int max_shape = 0;
    for(int i = 0; i < n; i++) {
        if(max_shape < ori_shape[i])
            max_shape = ori_shape[i];
    }
    dim3 grid(n, 1, 1);
    dim3 block(GetBlockSize(max_shape));
    Unpad1D_kernel<T><<<grid, block>>>((T*)(inputs.data_ptr()), ori_shape, outputs, max_shape);
}

template<typename T>
void Pad2D(const std::vector<torch::Tensor>& inputs, const int* shape, torch::Tensor new_x,
            torch::Tensor mask, int value = 0) {
    int max_shape0 = 0;
    int max_shape1 = 0;
    const int n = inputs.size();
    for(int i = 0; i < n; i++) {
        if(max_shape0 < shape[i * 2]) {
            max_shape0 = shape[i * 2];
        }
        if(max_shape1 < shape[i * 2 + 1]) {
            max_shape1 = shape[i * 2 + 1];
        }
    }
    dim3 grid(n, 1, 1);
    dim3 block(GetBlockSize(max_shape0), GetBlockSize(max_shape1), 1);

    Pad2D_kernel<T><<<grid, block>>>(inputs, shape, (T*)(new_x.data_ptr()), (T*)(mask.data_ptr()), max_shape0, max_shape1, value);
}

template<typename T>
void Unpad2D(torch::Tensor inputs, const int* ori_shape, std::vector<torch::Tensor>& outputs) {
    int max_shape0 = 0;
    int max_shape1 = 0;
    const int n = inputs.size();
    for(int i = 0; i < n; i++) {
        if(max_shape0 < ori_shape[i * 2]) {
            max_shape0 = ori_shape[i * 2];
        }
        if(max_shape1 < ori_shape[i * 2 + 1]) {
            max_shape1 = ori_shape[i * 2 + 1];
        }
    }
    dim3 grid(n, 1, 1);
    dim3 block(GetBlockSize(max_shape0), GetBlockSize(max_shape1), 1);
    Unpad2D_kernel<T><<<grid, block>>>((T*)(inputs.data_ptr()), ori_shape, outputs, max_shape0, max_shape1);
}




}  // namespace cuda
}  // namespace rll
}  // namespace hpc
