#ifndef HPC_RLL_CUDA_GAE_KERNEL_H_
#define HPC_RLL_CUDA_GAE_KERNEL_H_

#include "hpc/rll/cuda/common.h"
#include<vector>

namespace hpc {
namespace rll {
namespace cuda {
    /*
template<typename T>
void Pad1D :
std::vector<torch::Tensor>& inputs,    用vector保存输入的list of tensor
const int* shape,                      用指针保存一维输入的shape
T* new_x,                              输出new_x，按照输入X的顺序排列，n*max_shape
T* mask,                               输出mask，n*max_shape
const int max_shape，                  输入shape中的最大值
const int value,                       padding value

template<typename T>
void UnPad1D :
const T* inputs,                        用一维保存输入的padded tensor，n*max_shape
const int* ori_shape,                   用指针保存一维的原始shape
std::vector<torch::Tensor>& outputs,    输出x
const int max_shape，                   输入shape中的最大值


template<typename T>
void Pad2D :
std::vector<torch::Tensor>& inputs,    用vector保存输入的list of tensor
const int* shape,           用指针保存二维输入的shape
T* new_x,                   输出new_x，按照输入X的顺序排列，n*max_shape0*max_shape1
T* mask,                    输出mask，n*max_shape0*max_shape1
const int max_shape0，      输入shape中0维度的最大值
const int max_shape1，      输入shape中1维度的最大值
const int value,            padding value


template<typename T>
void UnPad2D :
const T* inputs,                            用一维保存输入的padded tensor，n*max_shape0*max_shape1
const int* ori_shape,                       用指针保存二维的原始shape
std::vector<torch::Tensor> outputs,         输出
const int max_shape0，                      输入shape中0维度的最大值
const int max_shape1，                      输入shape中1维度的最大值
*/

inline int GetBlockSize(const int n, const int max_size = 1024) {
    int ret = 32;
    while(ret < n && ret < max_size) {
        ret <<= 1;
    }
    return ret;
}

template<typename T>
__global__ void Pad1D_kernel(const std::vector<torch::Tensor>& inputs, const int* shape, T* new_x, T* mask, 
                                const int max_shape, const int value = 0) {
    auto cur_in = (T*)inputs[blockIdx.x].data_ptr();
    const int cur_shape = shape[blockIdx.x];
    const int base = blockIdx.x * max_shape;
    for(auto tid = threadIdx.x; tid < max_shape; tid += blockDim.x) {
        new_x[base + tid] = (tid < cur_shape) ? __ldg(cur_in + tid) : value;
        mask[base + tid] = (tid < cur_shape) ? 1 : value;
    }
}

template<typename T>
__global__ void Unpad1D_kernel(const T* inputs, const int* ori_shape, std::vector<torch::Tensor>& outputs,
                                 const int max_shape) {
    auto cur_in = inputs + blockIdx.x * max_shape;
    auto cur_out = (T*)outputs[blockIdx.x].data_ptr();
    auto cur_shape = ori_shape[blockIdx.x];
    for(auto tid = threadIdx.x; tid < cur_shape; tid += blockDim.x) {
        cur_out[tid] = __ldg(cur_in + tid);
    }
}

template<typename T>
__global__ void Pad2D_kernel(const std::vector<torch::Tensor>& inputs, const int* shape, T* new_x, T* mask,
                            const int max_shape0, const int max_shape1, int value = 0) {
    auto cur_in = (T*)inputs[blockIdx.x].data_ptr();
    const int cur_shape0 = shape[blockIdx.x * 2];
    const int cur_shape1 = shape[blockIdx.x * 2 + 1];
    const int base = blockIdx.x * max_shape0 * max_shape1;
    for(auto tid_y = threadIdx.y; tid_y < max_shape1; tid_y += blockDim.y)
        for(auto tid_x = threadIdx.x; tid_x < max_shape0; tid_x += blockDim.x) {
            new_x[base + tid_y * max_shape0 + tid_x] = (tid_x < cur_shape0 && tid_y < cur_shape1) ? __ldg(cur_in + tid_y * max_shape0 + tid_x) : value;
            mask[base + tid_y * max_shape0 + tid_x] = (tid_x < cur_shape0 && tid_y < cur_shape1) ? 1 : value;
        }
}

template<typename T>
__global__ void Unpad2D_kernel(const T* inputs, const int* ori_shape, std::vector<torch::Tensor>& outputs,
                                 const int max_shape0, const int max_shape1) {
    auto cur_in = inputs + blockIdx.x * max_shape0 * max_shape1;
    auto cur_out = (T*)outputs[blockIdx.x].data_ptr();
    auto cur_shape0 = ori_shape[blockIdx.x * 2];
    auto cur_shape1 = ori_shape[blockIdx.x * 2 + 1];
    for(auto tid_y = threadIdx.y; tid_y < cur_shape1; tid += blockDim.y)
        for(auto tid_x = threadIdx.x; tid_x < cur_shape0; tid += blockDim.x) {
            cur_out[tid_y * cur_shape0 + tid_x] = __ldg(cur_in + tid_y * cur_shape0 + tid_x);
        }
}



    


        }
    }
}

}  // namespace cuda
}  // namespace rll
}  // namespace hpc

#endif // HPC_RLL_CUDA_GAE_KERNEL_H_