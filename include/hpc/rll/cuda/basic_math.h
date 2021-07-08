#ifndef HPC_RLL_CUDA_BASIC_MATH_H_
#define HPC_RLL_CUDA_BASIC_MATH_H_

#include <float.h>
#include <math.h>
#include <vector>
#include <fstream>

namespace hpc {
namespace rll {
namespace cuda {

template<typename T>
__forceinline__ __device__
T clamp(T in, T min, T max)  {
    if (in < min)
        return min;
    else if (in <= max)
        return in;
    else
        return max;
}

template<typename T>
__forceinline__ __device__
T sigmoid(T in)  {
    T one = static_cast<T>(1.0);
    return one / (one + ::exp(-in));
}

}  // namespace cuda
}  // namespace rll
}  // namespace hpc

#endif // HPC_RLL_CUDA_BASIC_MATH_H_
