#ifndef HPC_RLL_CUDA_STATUS_H_
#define HPC_RLL_CUDA_STATUS_H_

#include <cuda.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <torch/extension.h>

namespace hpc {
namespace rll {
namespace cuda {

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

static int checkCudaError(cudaError_t code, const char* expr, const char* file, int line, bool abort = true) {
    if (code) {
        fprintf(stderr, "CUDA error at %s:%d, code=%d (%s) in '%s'", file, line, (int) code, cudaGetErrorString(code), expr);
        if (abort)
            throw std::logic_error("CUDA Error.");
    }
    return 0;
}

#define checkCudaErr(...) do { int err = checkCudaError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__); } while (0)

static const char* cublasGetErrorString(cublasStatus_t status)
{
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"; 
    }
    return "unknown error";
}

static int checkCublasError(cublasStatus_t code, const char* expr, const char* file, int line, bool abort = true) {
    if (code) {
        fprintf(stderr, "CUBLAS error at %s:%d, code=%d (%s) in '%s'", file, line, (int) code, cublasGetErrorString(code), expr);
        if (abort)
            throw std::logic_error("CUBLAS Error.");
    }
    return 0;
}

#define checkCublasErr(...) do { int err = checkCublasError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__); } while (0)

static const char* curandGetErrorString(curandStatus_t status)
{
    switch (status)
    {
        case CURAND_STATUS_SUCCESS: return "CURAND_STATUS_SUCCESS";
        case CURAND_STATUS_VERSION_MISMATCH: return "CURAND_STATUS_VERSION_MISMATCH";
        case CURAND_STATUS_NOT_INITIALIZED: return "CURAND_STATUS_NOT_INITIALIZED";
        case CURAND_STATUS_ALLOCATION_FAILED: return "CURAND_STATUS_ALLOCATION_FAILED";
        case CURAND_STATUS_TYPE_ERROR: return "CURAND_STATUS_TYPE_ERROR";
        case CURAND_STATUS_OUT_OF_RANGE: return "CURAND_STATUS_OUT_OF_RANGE";
        case CURAND_STATUS_LENGTH_NOT_MULTIPLE: return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
        case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED: return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
        case CURAND_STATUS_LAUNCH_FAILURE: return "CURAND_STATUS_LAUNCH_FAILURE";
        case CURAND_STATUS_PREEXISTING_FAILURE: return "CURAND_STATUS_PREEXISTING_FAILURE";
        case CURAND_STATUS_INITIALIZATION_FAILED: return "CURAND_STATUS_INITIALIZATION_FAILED";
        case CURAND_STATUS_ARCH_MISMATCH: return "CURAND_STATUS_ARCH_MISMATCH";
        case CURAND_STATUS_INTERNAL_ERROR: return "CURAND_STATUS_INTERNAL_ERROR";
    }
    return "unknown error";
}

static int checkCurandError(curandStatus_t code, const char* expr, const char* file, int line, bool abort = true) {
    if (code) {
        fprintf(stderr, "CURAND error at %s:%d, code=%d (%s) in '%s'", file, line, (int) code, curandGetErrorString(code), expr);
        if (abort)
            throw std::logic_error("CURAND Error.");
    }
    return 0;
}

#define checkCurandErr(...) do { int err = checkCurandError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__); } while (0)

}  // namespace cuda
}  // namespace rll
}  // namespace hpc

#endif // HPC_RLL_CUDA_STATUS_H_
