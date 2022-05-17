#ifndef HPC_RLL_CUDA_GRU_KERNEL_H_
#define HPC_RLL_CUDA_GRU_KERNEL_H_

#include "hpc/rll/cuda/common.h"
#include "hpc/rll/cuda/reduce.h"
#include "hpc/rll/cuda/basic_math.h"

namespace hpc {
namespace rll {
namespace cuda {

__global__ void GRUPartialKernel(unsigned int input_dim, const float* x, const float* Wzy, 
        const float* Uzx, const float* Wgy, const float* Ugrx, const float* bg, float* g, float* h, float* z) {
	unsigned int block_start = blockIdx.x * input_dim;
    unsigned int start = block_start + threadIdx.x;
	unsigned int end = block_start + input_dim;
    for(int i = start; i < end; i += blockDim.x) {
        double z_item = sigmoid(Wzy[i] + Uzx[i] - bg[i - start]);
        double h_item = tanh(Wgy[i] + Ugrx[i]);
        z[i] = z_item;
        h[i] = h_item;
        g[i] = (1 - z_item) * x[i] + z_item * h_item;
    }
}


__global__ void GRUBackwardKernel(int TB, int input_dim, const float* grad_g, const float* h, const float* z, const float* x,
            float* grad_Wzy, float* grad_Uzx, float* grad_Wgy, float* grad_Ugrx, float* grad_bg) {
    unsigned int block_start = blockIdx.x;
    unsigned int start = block_start + threadIdx.x * input_dim;
	unsigned int end = block_start + TB * input_dim;
    float grad_bg_item = 0;
    for(int i = start; i < end; i += blockDim.x * input_dim) {
        grad_Wzy[i] = grad_g[i] * (h[i] - x[i]) * z[i] * (1 - z[i]);
        grad_Uzx[i] = grad_g[i] * (h[i] - x[i]) * z[i] * (1 - z[i]);
        grad_Wgy[i] = grad_g[i] * z[i] * (1 - h[i] * h[i]);
        grad_Ugrx[i] = grad_g[i] * z[i] * (1 - h[i] * h[i]);
        grad_bg_item += grad_g[i] * (x[i] - h[i]) * z[i] * (1 - z[i]);
    }
    grad_bg[blockIdx.x] = blockReduceSum<float>(grad_bg_item);

}




}  // namespace cuda
}  // namespace rll
}  // namespace hpc
#endif // HPC_RLL_CUDA_GRU_KERNEL_H_
