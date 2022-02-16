#ifndef HPC_RLL_CUDA_PAD_KERNEL_H_
#define HPC_RLL_CUDA_PAD_KERNEL_H_

#include "hpc/rll/cuda/common.h"
#include<stdio.h>

namespace hpc {
namespace rll {
namespace cuda {
    /*
void Pad1D :
std::vector<torch::Tensor>& inputs,    用vector保存输入的list of tensor
const int* shape,                      用指针保存一维输入的shape
float* new_x,                              输出new_x，按照输入X的顺序排列，n*max_shape
float* mask,                               输出mask，n*max_shape
const int max_shape，                  输入shape中的最大值
const int value,                       padding value

void UnPad1D :
const float* inputs,                        用一维保存输入的padded tensor，n*max_shape
const int* ori_shape,                   用指针保存一维的原始shape
std::vector<torch::Tensor>& outputs,    输出x
const int max_shape，                   输入shape中的最大值


void Pad2D :
std::vector<torch::Tensor>& inputs,    用vector保存输入的list of tensor
const int* shape,           用指针保存二维输入的shape
float* new_x,                   输出new_x，按照输入X的顺序排列，n*max_shape0*max_shape1
float* mask,                    输出mask，n*max_shape0*max_shape1
const int max_shape0，      输入shape中0维度的最大值
const int max_shape1，      输入shape中1维度的最大值
const int value,            padding value


void UnPad2D :
const float* inputs,                            用一维保存输入的padded tensor，n*max_shape0*max_shape1
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

// __global__ void Pad1D_kernel(float* inputs, const int* shape, float* new_x, int* mask, 
//                                 const int max_shape, const int value, int v_id) {
//     const int cur_shape = shape[v_id];
//     const int base = v_id * max_shape;
//     for(auto tid = threadIdx.x; tid < max_shape; tid += blockDim.x) {
//         new_x[base + tid] = (tid < cur_shape) ? __ldg(inputs + tid) : value;
//         mask[base + tid] = (tid < cur_shape) ? 1 : value;
//     }
// }

__global__ void Pad1D_kernel(float** inputs, const int* shape, float* new_x, int* mask, 
                                const int max_shape, const int value = 0) {
    auto cur_in = inputs[blockIdx.x];
    const int cur_shape = shape[blockIdx.x];
    const int base = blockIdx.x * max_shape;
    for(auto tid = threadIdx.x; tid < max_shape; tid += blockDim.x) {
        new_x[base + tid] = (tid < cur_shape) ? __ldg(cur_in + tid) : value;
        mask[base + tid] = (tid < cur_shape) ? 1 : value;
    }
}

__global__ void GroupPad1D_kernel(float** inputs, const int* shape, float** new_x, int** mask, 
                                const int* max_shape, const int* group_id, const int* group_idx, int value = 0) {
    // group_id: batch_id -> group_id
    // group_idx: 每个group的起始batch_id
    const float* cur_in = inputs[blockIdx.x];
    const int cur_shape = shape[blockIdx.x];
    const int gid = group_id[blockIdx.x];
    const int group_max_shape = max_shape[gid];
    float* cur_new_x = new_x[gid];

    
    int* cur_mask = mask[gid];
    int group_base = (blockIdx.x - group_idx[gid]) * group_max_shape;
    for(auto tid = threadIdx.x; tid < group_max_shape; tid += blockDim.x) {
        cur_new_x[group_base + tid] = (tid < cur_shape) ? __ldg(cur_in + tid) : value;
        cur_mask[group_base + tid] = (tid < cur_shape) ? 1 : value;
    }


}

__global__ void Unpad1D_kernel(float* inputs, const int* ori_shape, float** outputs,
                                 const int max_shape) {
    auto cur_in = inputs + blockIdx.x * max_shape;
    auto cur_out = outputs[blockIdx.x];
    auto cur_shape = ori_shape[blockIdx.x];
    for(auto tid = threadIdx.x; tid < cur_shape; tid += blockDim.x) {
        cur_out[tid] = __ldg(cur_in + tid);
    }
}

__global__ void Pad2D_kernel(float** inputs, const int* shape, float* new_x, int* mask,
                            const int max_shape0, const int max_shape1, const int value = 0) {
    auto cur_in = inputs[blockIdx.x];
    const int cur_shape0 = shape[blockIdx.x * 2];
    const int cur_shape1 = shape[blockIdx.x * 2 + 1];
    const int base = blockIdx.x * max_shape0 * max_shape1;
    for(auto tid_y = threadIdx.y; tid_y < max_shape0; tid_y += blockDim.y) {
        for(auto tid_x = threadIdx.x; tid_x < max_shape1; tid_x += blockDim.x) {
            int offset_out = tid_y * max_shape1 + tid_x;
            int offset_in = tid_y * cur_shape1 + tid_x;
            new_x[base + offset_out] = (tid_x < cur_shape1 && tid_y < cur_shape0) ? __ldg(cur_in + offset_in) : value;
            mask[base + offset_out] = (tid_x < cur_shape1 && tid_y < cur_shape0) ? 1 : value;
        }
    }
}

__global__ void GroupPad2D_kernel(float** inputs, const int* shape, float** new_x, int** mask, 
                                const int* max_shape, const int* group_id, const int* group_idx, const int value = 0) {
    const float* cur_in = inputs[blockIdx.x];
    const int cur_shape0 = shape[blockIdx.x * 2];
    const int cur_shape1 = shape[blockIdx.x * 2 + 1];
    const int gid = group_id[blockIdx.x];
    const int group_max_shape0 = max_shape[gid * 2];
    const int group_max_shape1 = max_shape[gid * 2 + 1];
    float* cur_new_x = new_x[gid];
    int* cur_mask = mask[gid];
    const int group_base = (blockIdx.x - group_idx[gid]) * group_max_shape0 * group_max_shape1;
    for(auto tid_y = threadIdx.y; tid_y < group_max_shape0; tid_y += blockDim.y)
        for(auto tid_x = threadIdx.x; tid_x < group_max_shape1; tid_x += blockDim.x) {
            int offset_out = tid_y * group_max_shape1 + tid_x;
            int offset_in = tid_y * cur_shape1 + tid_x;
            cur_new_x[group_base + offset_out] = (tid_x < cur_shape1 && tid_y < cur_shape0) ? __ldg(cur_in + offset_in) : value;
            cur_mask[group_base + offset_out] = (tid_x < cur_shape1 && tid_y < cur_shape0) ? 1 : value;
        }
}

__global__ void Unpad2D_kernel(float* inputs, const int* ori_shape, float** outputs,
                                 const int max_shape0, const int max_shape1) {
    auto cur_shape0 = ori_shape[blockIdx.x * 2];
    auto cur_shape1 = ori_shape[blockIdx.x * 2 + 1];
    auto cur_in = inputs + blockIdx.x * max_shape0 * max_shape1;
    auto cur_out = outputs[blockIdx.x];
    for(auto tid_y = threadIdx.y; tid_y < cur_shape0; tid_y += blockDim.y)
        for(auto tid_x = threadIdx.x; tid_x < cur_shape1; tid_x += blockDim.x) {
            int offset_out = tid_y * cur_shape1 + tid_x;
            int offset_in = tid_y * max_shape1 + tid_x;
            cur_out[offset_out] = __ldg(cur_in + offset_in);
        }
}

__global__ void Pad3D_kernel(float** inputs, const int* shape, float* new_x, int* mask,
                             const int max_shape0, const int max_shape1, const int max_shape2, 
                             const int value = 0) {
    auto cur_in = inputs[blockIdx.x];
    const int cur_shape0 = shape[blockIdx.x * 3];
    const int cur_shape1 = shape[blockIdx.x * 3 + 1];
    const int cur_shape2 = shape[blockIdx.x * 3 + 2];
    const int base = blockIdx.x * max_shape0 * max_shape1 * max_shape2;
    for(auto tid_z = threadIdx.z; tid_z < max_shape0; tid_z += blockDim.z)
        for(auto tid_y = threadIdx.y; tid_y < max_shape1; tid_y += blockDim.y)
            for(auto tid_x = threadIdx.x; tid_x < max_shape2; tid_x += blockDim.x) {
                int offset_out = tid_z * max_shape1 * max_shape2 + tid_y * max_shape2 + tid_x;
                int offset_in = tid_z * cur_shape1 * cur_shape2 + tid_y * cur_shape2 + tid_x;
                bool pred = (tid_x < cur_shape2 && tid_y < cur_shape1 && tid_z < cur_shape0);
                new_x[base + offset_out] = pred ? __ldg(cur_in + offset_in) : value;
                mask[base + offset_out] = pred ? 1 : value;
            }
}

__global__ void GroupPad3D_kernel(float** inputs, const int* shape, float** new_x, int** mask, 
                                const int* max_shape, const int* group_id, const int* group_idx, int value = 0) {
    const float* cur_in = inputs[blockIdx.x];
    const int cur_shape0 = shape[blockIdx.x * 3];
    const int cur_shape1 = shape[blockIdx.x * 3 + 1];
    const int cur_shape2 = shape[blockIdx.x * 3 + 2];
    const int gid = group_id[blockIdx.x];
    const int group_max_shape0 = max_shape[gid * 3];
    const int group_max_shape1 = max_shape[gid * 3 + 1];
    const int group_max_shape2 = max_shape[gid * 3 + 2];
    float* cur_new_x = new_x[gid];
    int* cur_mask = mask[gid];
    const int group_base = (blockIdx.x - group_idx[gid]) * group_max_shape0 * group_max_shape1 * group_max_shape2;
    for(auto tid_z = threadIdx.z; tid_z < group_max_shape0; tid_z += blockDim.z)
        for(auto tid_y = threadIdx.y; tid_y < group_max_shape1; tid_y += blockDim.y)
            for(auto tid_x = threadIdx.x; tid_x < group_max_shape2; tid_x += blockDim.x) {
                int offset_out = tid_z * group_max_shape1 * group_max_shape2 + tid_y * group_max_shape2 + tid_x;
                int offset_in = tid_z * cur_shape1 * cur_shape2 + tid_y * cur_shape2 + tid_x;
                bool pred = (tid_x < cur_shape2 && tid_y < cur_shape1 && tid_z < cur_shape0);
                cur_new_x[group_base + offset_out] = pred ? __ldg(cur_in + offset_in) : value;
                cur_mask[group_base + offset_out] = pred ? 1 : value;
            }
}

__global__ void Unpad3D_kernel(float* inputs, const int* ori_shape, float** outputs,
                                 const int max_shape0, const int max_shape1, const int max_shape2) {
    auto cur_shape0 = ori_shape[blockIdx.x * 3];
    auto cur_shape1 = ori_shape[blockIdx.x * 3 + 1];
    auto cur_shape2 = ori_shape[blockIdx.x * 3 + 2];
    auto cur_in = inputs + blockIdx.x * max_shape0 * max_shape1 * max_shape2;
    auto cur_out = outputs[blockIdx.x];
    for(auto tid_z = threadIdx.z; tid_z < cur_shape0; tid_z += blockDim.z)
        for(auto tid_y = threadIdx.y; tid_y < cur_shape1; tid_y += blockDim.y)
            for(auto tid_x = threadIdx.x; tid_x < cur_shape2; tid_x += blockDim.x) {
                int offset_out = tid_z * cur_shape1 * cur_shape2 + tid_y * cur_shape2 + tid_x;
                int offset_in = tid_z * max_shape1 * max_shape2 + tid_y * max_shape2 + tid_x;
                cur_out[offset_out] = __ldg(cur_in + offset_in);
            }
}


}  // namespace cuda
}  // namespace rll
}  // namespace hpc

#endif // HPC_RLL_CUDA_PAD_KERNEL_H_
