#include "hpc/rll/cuda/rl_utils/entry.h"
#include "hpc/rll/cuda/rl_utils/padding_kernel.h"

namespace hpc {
namespace rll {
namespace cuda {

template<typename T>
void Pad1DForward(const std::vector<torch::Tensor>& inputs, torch::Tensor shape,
                 torch::Tensor new_x, torch::Tensor mask, int max_shape) {
    const int n = inputs.size();
    int* shape_ptr = (int*)shape.data_ptr();
    dim3 grid(n, 1, 1);
    dim3 block(GetBlockSize(max_shape));
    Pad1D_kernel<T><<<grid, block>>>(inputs, shape_ptr, (T*)(new_x.data_ptr()), (T*)(mask.data_ptr()), max_shape, 0);
}

template<typename T>
void Unpad1DForward(torch::Tensor inputs, torch::Tensor shape, std::vector<torch::Tensor>& outputs) {
    const int n = outputs.size();
    int max_shape = 0;
    const int* shape_ptr = (int*)shape.data_ptr();
    for(int i = 0; i < n; i++) {
        if(max_shape < shape_ptr[i])
            max_shape = shape_ptr[i];
    }
    dim3 grid(n, 1, 1);
    dim3 block(GetBlockSize(max_shape));
    Unpad1D_kernel<T><<<grid, block>>>((T*)(inputs.data_ptr()), shape_ptr, outputs, max_shape);
}

template<typename T>
void Pad2DForward(const std::vector<torch::Tensor>& inputs, torch::Tensor shape, torch::Tensor new_x,
            torch::Tensor mask, int max_shape0, int max_shape1) {
    const int* shape_ptr = (int*)shape.data_ptr();
    const int n = inputs.size();
    dim3 grid(n, 1, 1);
    dim3 block(GetBlockSize(max_shape0), GetBlockSize(max_shape1), 1);
    Pad2D_kernel<T><<<grid, block>>>(inputs, shape_ptr, (T*)(new_x.data_ptr()), (T*)(mask.data_ptr()), max_shape0, max_shape1, 0);
}

template<typename T>
void Unpad2DForward(torch::Tensor inputs, torch::Tensor shape, std::vector<torch::Tensor>& outputs) {
    int max_shape0 = 0;
    int max_shape1 = 0;
    const int* shape_ptr = (int*)shape.data_ptr();
    const int n = inputs.size();
    auto
    for(int i = 0; i < n; i++) {
        if(max_shape0 < shape_ptr[i * 2]) {
            max_shape0 = shape_ptr[i * 2];
        }
        if(max_shape1 < shape_ptr[i * 2 + 1]) {
            max_shape1 = shape_ptr[i * 2 + 1];
        }
    }
    dim3 grid(n, 1, 1);
    dim3 block(GetBlockSize(max_shape0), GetBlockSize(max_shape1), 1);
    Unpad2D_kernel<T><<<grid, block>>>((T*)(inputs.data_ptr()), shape_ptr, outputs, max_shape0, max_shape1);
}




}  // namespace cuda
}  // namespace rll
}  // namespace hpc
