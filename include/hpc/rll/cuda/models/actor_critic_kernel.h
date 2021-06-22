#ifndef HPC_RLL_CUDA_ACTOR_CRITIC_KERNEL_H_
#define HPC_RLL_CUDA_ACTOR_CRITIC_KERNEL_H_

#include "hpc/rll/cuda/common.h"
#include "hpc/rll/cuda/reduce.h"
#include "hpc/rll/cuda/basic_math.h"

namespace hpc {
namespace rll {
namespace cuda {

// key_embeddings: [batch_size, max_entity_num, input_dim]
// autoregressive_embedding: [batch_size, input_dim]
__global__ void autoregressive_embedding_fp(int64_t batch_size, int64_t max_entity_num, int64_t input_dim,
        int64_t* sample_entity, int64_t* entity_num,
        const float* key_embeddings, float* autoregressive_embedding) {
    unsigned int gidx = blockIdx.x * blockDim.x + threadIdx.x; // input_dim
    unsigned int gidy = blockIdx.y; // batch_size

    int64_t entity_index = sample_entity[gidy];
    bool end_flag = (entity_index == entity_num[gidy]);

    float ke = 0.f;
    if (!end_flag) {
        int64_t ke_index = gidy * max_entity_num * input_dim + entity_index * input_dim + gidx;
        ke = key_embeddings[ke_index];
    }

    int64_t ae_index = gidy * input_dim + gidx;
    autoregressive_embedding[ae_index] += ke;
}

__global__ void lstm_activation_fp(unsigned int batch_size, unsigned int hidden_size,
        const float* in_x, const float* in_h, const float* bias, float* h, float* c) {
    unsigned int gidx = blockIdx.x * blockDim.x + threadIdx.x; // hidden_size
    unsigned int gidy = blockIdx.y; // batch_size
    unsigned int start = gidy * hidden_size * 4;
    if (gidx < hidden_size) {
        float val[4];
        for (int i = 0; i < 4; i++) {
            val[i] = in_x[start + i * hidden_size + gidx] + in_h[start + i * hidden_size + gidx]
                + bias[i * hidden_size + gidx];
        }

        float i = sigmoid(val[0]);
        float f = sigmoid(val[1]);
        float g = tanh(val[2]);
        float o = sigmoid(val[3]);
        float pre_c = c[gidy * hidden_size + gidx];
        float new_c = f * pre_c + i * g;
        float new_h = o * tanh(new_c);

        h[gidy * hidden_size + gidx] = new_h;
        c[gidy * hidden_size + gidx] = new_c;
    }
}

__global__ void pre_sample_fp(unsigned int batch_size, unsigned int max_entity_num, unsigned int hidden_size,
        const float mask_value, const float div_factor,
        const float* mat, const float* vec, const bool* mask, float* output) {
    unsigned int tidx = threadIdx.x; // hidden_size
    unsigned int gidy = blockIdx.y; // max_entity_num
    unsigned int gidz = blockIdx.z; // batch_size

    if (mask[gidz * max_entity_num + gidy]) {
        float mul_val = 0.f;
        for (int i = tidx; i < hidden_size; i += blockDim.x) {
            float mat_val = mat[gidz * max_entity_num * hidden_size + gidy * hidden_size + i];
            float vec_val = vec[gidz * hidden_size + i];
            mul_val += mat_val * vec_val;
        }
        float reduce_sum = blockReduceSum<float>(mul_val);
        if (tidx == 0)
            output[gidz * max_entity_num + gidy] = reduce_sum / div_factor;
    } else {
        if (tidx == 0)
            output[gidz * max_entity_num + gidy] = mask_value / div_factor;
    }
}

}  // namespace cuda
}  // namespace rll
}  // namespace hpc
#endif // HPC_RLL_CUDA_ACTOR_CRITIC_KERNEL_H_
