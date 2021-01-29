#ifndef HPC_RLL_CUDA_SCATTER_CONNECTION_KERNEL_H_
#define HPC_RLL_CUDA_SCATTER_CONNECTION_KERNEL_H_

#include "hpc/rll/cuda/common.h"
#include "hpc/rll/cuda/reduce.h"

namespace hpc {
namespace rll {
namespace cuda {

// deprecated
// single block deal with one out element
// single warp in block
// consider input data overlap, and keep sequence
void __global__ scatterConnectionCoverKeepSeqForwardKernel(unsigned int B, unsigned int M, unsigned int N, unsigned int H, unsigned int W,
        const float* in, const int64_t* location, float* out) {
    unsigned int lid = threadIdx.x; // each out element use 1 block
    unsigned int out_id = blockIdx.x; // B * N * H * W
    unsigned int bid = out_id / (N * H * W);
    unsigned int nid = (out_id % (N * H * W)) / (H * W);
    unsigned int hwid = out_id % (H * W);
    unsigned int bhw_id = bid * (H * W) + hwid;

    int in_id = -1;
    int max_id = -1;
    for (int b = 0; b < B; b++) {
        for (int m = lid; m < M; m += WARP_SIZE) {
            unsigned int y = location[b * M * 2 + m * 2 + 0];
            unsigned int x = location[b * M * 2 + m * 2 + 1];
            unsigned int target_id = b * H * W + y * W + x;

            if (bhw_id == target_id) {
                max_id = b * M + m;
                in_id = b * M * N +  m * N + nid;
            }
        }
    }

    // reduce max idx
    int reduced_max_id = blockReduceMax<int>(max_id);
    static __shared__ int s_max_id;
    if (lid == 0) {
        s_max_id = reduced_max_id;
    }
    __syncthreads();

    if (s_max_id == -1) {
        if (lid == 0)
            out[out_id] = 0;
    } else {
        if (((s_max_id % M) % WARP_SIZE) == lid) {
            out[out_id] = in[in_id];
        }
    }
}

// assuming no overlap of input data
void __global__ scatterConnectionCoverForwardKernel(unsigned int B, unsigned int M, unsigned int N, unsigned int H, unsigned int W,
        const float* in, const int64_t* location, float* out) {
    unsigned int nid = threadIdx.x + blockIdx.x*blockDim.x; // N
    unsigned int mid = threadIdx.y + blockIdx.y*blockDim.y; // M
    unsigned int bid = blockIdx.z; // B
    if (nid >= N || mid >= M || bid >= B) return;

    unsigned int in_id = bid * M * N + mid * N + nid;

    unsigned int yid = location[bid * M * 2 + mid * 2 + 0];
    unsigned int xid = location[bid * M * 2 + mid * 2 + 1];

    unsigned int out_id = bid * N * H * W + nid * H * W + yid * W + xid;
    out[out_id] = in[in_id];
}

void __global__ scatterConnectionAddForwardKernel(unsigned int B, unsigned int M, unsigned int N, unsigned int H, unsigned int W,
        const float* in, const int64_t* location, float* out) {
    unsigned int nid = threadIdx.x + blockIdx.x*blockDim.x; // N
    unsigned int mid = threadIdx.y + blockIdx.y*blockDim.y; // M
    unsigned int bid = blockIdx.z; // B
    if (nid >= N || mid >= M || bid >= B) return;

    unsigned int in_id = bid * M * N + mid * N + nid;

    unsigned int yid = location[bid * M * 2 + mid * 2 + 0];
    unsigned int xid = location[bid * M * 2 + mid * 2 + 1];

    unsigned int out_id = bid * N * H * W + nid * H * W + yid * W + xid;
    atomicAdd(&out[out_id], in[in_id]);
}

void __global__ scatterConnectionBackwardKernel(unsigned int B, unsigned int M, unsigned int N, unsigned int H, unsigned int W,
        const float* grad_out, const int64_t* location, float* grad_in) {
    unsigned int nid = threadIdx.x + blockIdx.x*blockDim.x; // N
    unsigned int mid = threadIdx.y + blockIdx.y*blockDim.y; // M
    unsigned int bid = blockIdx.z; // B
    if (nid >= N || mid >= M || bid >= B) return;

    unsigned int in_id = bid * M * N + mid * N + nid;

    unsigned int yid = location[bid * M * 2 + mid * 2 + 0];
    unsigned int xid = location[bid * M * 2 + mid * 2 + 1];

    unsigned int out_id = bid * N * H * W + nid * H * W + yid * W + xid;

    grad_in[in_id] = grad_out[out_id];
}

}  // namespace cuda
}  // namespace rll
}  // namespace hpc
#endif // HPC_RLL_CUDA_SCATTER_CONNECTION_KERNEL_H_
