#ifndef HPC_RLL_CUDA_COMMON_H_
#define HPC_RLL_CUDA_COMMON_H_

#include <float.h>
#include <math.h>
#include <vector>
#include <fstream>

#include <torch/extension.h>

#include "hpc/rll/cuda/status.h"

namespace hpc {
namespace rll {
namespace cuda {

#define TRACE { \
    int err = cudaDeviceSynchronize(); \
    fprintf(stderr, "TRACE: %s %d, err = %d\n", __FILE__, __LINE__, err); \
}

static void print_tensor(const char* tensor_name, float* ptr, int len) {
    float hostbuf[len];
    checkCudaErr(cudaMemcpy(hostbuf, ptr, len * sizeof(float), cudaMemcpyDeviceToHost));

    fprintf(stderr, "%s\n", tensor_name);
    for (int i = 0; i < len; i++)
        fprintf(stderr, "%lf\n", hostbuf[i]);
}

static void save_tensor(const char* tensor_name, float* ptr, int len) {
    float hostbuf[len];
    checkCudaErr(cudaMemcpy(hostbuf, ptr, len * sizeof(float), cudaMemcpyDeviceToHost));

    char filename[256];
    sprintf(filename, "%s.dat", tensor_name);
    std::ofstream outfile;
    outfile.open(filename);
    for (int i = 0; i < len; i++)
        outfile << hostbuf[i] << std::endl;
    outfile.close();
}

const unsigned int DEFAULT_WARP_NUM = 8;
const unsigned int WARP_SIZE = 32;
const float CUDA_FLOAT_INF_POS = FLT_MAX;
const float CUDA_FLOAT_INF_NEG = -FLT_MAX;

// torch.finfo(torch.float32).eps say epsilon = 1.19209e-07, but pytorch layernorm userguide say epsilon = 1e-5
const float EPSILON = 1e-5;

}  // namespace cuda
}  // namespace rll
}  // namespace hpc

#endif // HPC_RLL_CUDA_COMMON_H_
