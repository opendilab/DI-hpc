#ifndef HPC_RLL_CUDA_ACTOR_CRITIC_KERNEL_H_
#define HPC_RLL_CUDA_ACTOR_CRITIC_KERNEL_H_

#include "hpc/rll/cuda/common.h"
#include "hpc/rll/cuda/reduce.h"
#include "hpc/rll/cuda/basic_math.h"

namespace hpc {
namespace rll {
namespace cuda {

// key_embeddings: [batch_size, entity_num, input_dim]
// autoregressive_embedding: [batch_size, input_dim]
__global__ void autoregressive_embedding_fp(int64_t batch_size, int64_t entity_num, int64_t input_dim,
        int64_t* sample_result, int64_t* true_entity_num,
        const float* key_embeddings, float* autoregressive_embedding) {
    unsigned int gidx = blockIdx.x * blockDim.x + threadIdx.x; // input_dim
    unsigned int gidy = blockIdx.y; // batch_size

    int64_t entity_index = sample_result[0];
    bool end_flag = (entity_index == true_entity_num[0]);

    float ke = 0.f;
    if (!end_flag) {
        int64_t ke_index = gidy * entity_num * input_dim + entity_index * input_dim + gidx;
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

__global__ void pre_sample_fp(unsigned int batch_size, unsigned int entity_num, unsigned int hidden_size,
        const float mask_value, const float div_factor,
        const float* mat, const float* vec, const bool* mask, float* output) {
    unsigned int gidx = threadIdx.x; // hidden_size
    unsigned int gidy = blockIdx.y; // batch_size * entity_num

    if (mask[gidy]) {
        float mat_val = mat[gidy * hidden_size + gidx];
        float vec_val = vec[gidx];
        float mul_val = mat_val * vec_val;
        float reduce_sum = blockReduceSum<float>(mul_val);
        if (threadIdx.x == 0)
            output[gidy] = reduce_sum / div_factor;
    } else {
        if (threadIdx.x == 0)
            output[gidy] = mask_value / div_factor;
    }
}

}  // namespace cuda
}  // namespace rll
}  // namespace hpc
#endif // HPC_RLL_CUDA_ACTOR_CRITIC_KERNEL_H_
