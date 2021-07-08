#ifndef HPC_RLL_CUDA_VTRACE_KERNEL_H_
#define HPC_RLL_CUDA_VTRACE_KERNEL_H_

#include "hpc/rll/cuda/common.h"
#include "hpc/rll/cuda/reduce.h"

namespace hpc {
namespace rll {
namespace cuda {

__global__ void categoricalTarget(unsigned int num_output, const float* target, const int64_t* action,
        float* target_prob, float* target_entropy, float* target_grad_logits,
        float* target_grad_prob, float* target_grad_entropy) {
	unsigned int block_start = blockIdx.x * num_output;
    unsigned int start = block_start + threadIdx.x;
	unsigned int end = block_start + num_output;

    // step 1: logits = x - logsumexp(x)
	// step 1.1 get max_x
	float max_x = CUDA_FLOAT_INF_NEG;
	for (int i = start; i < end; i += blockDim.x) {
        max_x = max(max_x, target[i]);
	}
    static __shared__ float s_max_x;
    float reduced_max_x = blockReduceMax<float>(max_x);
	if (threadIdx.x == 0) {
        s_max_x = reduced_max_x;
    }
	__syncthreads();

	// step 1.2 compute log(sum(exp(x - max_x))) + max_x
    static __shared__ float s_sum_exp_x;
	float sum_exp_x = 0.0;
	for (int i = start; i < end; i += blockDim.x) {
        sum_exp_x += std::exp(target[i] - s_max_x);
	}
    float reduced_sum_exp_x = blockReduceSum<float>(sum_exp_x);
    if (threadIdx.x == 0) {
        s_sum_exp_x = reduced_sum_exp_x;
    }
	__syncthreads();


    // step 2: entropy = -sum(logits * softmax(logits))
    float log_sum_exp_x = std::log(s_sum_exp_x) + s_max_x;

	// step 2.1 get max_logits
	float max_logits = CUDA_FLOAT_INF_NEG;
    for (int i = start; i < end; i += blockDim.x) {
        float logits = target[i] - log_sum_exp_x;
        max_logits = max(max_logits, logits);
	}
    float reduced_max_logits = blockReduceMax<float>(max_logits);
    static __shared__ float s_max_logits;
	if (threadIdx.x == 0) {
        s_max_logits = reduced_max_logits;
    }
	__syncthreads();

	// step 2.2 compute sum(exp(logits - max_logits))
	float sum_exp_logits = 0.0;
	for (int i = start; i < end; i += blockDim.x) {
        float logits = target[i] - log_sum_exp_x;
        sum_exp_logits += std::exp(logits - s_max_logits);
	}
    float reduced_sum_exp_logits = blockReduceSum<float>(sum_exp_logits);
    static __shared__ float s_sum_exp_logits;
    if (threadIdx.x == 0) {
        s_sum_exp_logits = reduced_sum_exp_logits;
    }
    __syncthreads();

    // step 2.3 get sum(logits * softmax(logits))
    // softmax(logits) = std::exp(logits - s_max_logits) / s_sum_exp_logits
    static __shared__ float s_sum_entropy_val;
    float sum_entropy_val = 0.f;
	for (int i = start; i < end; i += blockDim.x) {
        float logits = target[i] - log_sum_exp_x;
        float softmax_logits = std::exp(logits - s_max_logits) / s_sum_exp_logits;
        sum_entropy_val += logits * softmax_logits;
	}
    float reduced_sum_entropy_val = blockReduceSum<float>(sum_entropy_val);
    if (threadIdx.x == 0) { 
        s_sum_entropy_val = reduced_sum_entropy_val;
    }
    __syncthreads();

    // step 3. output
    // output prob, entropy, grad_logits, grad_prob, grad_entropy
    // grad_entropy[i] = (-1) * softmax(logits[i]) * (1 + logits[i] - sum(logits * softmax(logits)))
	for (int i = start; i < end; i += blockDim.x) {
        bool flag = ((i - block_start) == action[blockIdx.x]);
        float val = target[i];
        float logits = val - log_sum_exp_x;
        float softmax_logits = std::exp(logits - s_max_logits) / s_sum_exp_logits;

        if (flag)
            target_prob[blockIdx.x] = val - log_sum_exp_x;

        target_entropy[blockIdx.x] = -s_sum_entropy_val;

        // grad of logsumexp(x)
        float grad = std::exp(val - s_max_x) / s_sum_exp_x;
        target_grad_logits[i] = grad;

        // grad of x - logsumexp(x)
        target_grad_prob[i] = (flag ? 1 : 0) - grad;

        // grad of -sum(logits * softmax(logits))
        target_grad_entropy[i] = (-1.f) * softmax_logits * (1 + logits - s_sum_entropy_val);
    }
}

__global__ void categoricalBehaviour(unsigned int num_output,
        const float* behaviour, const int64_t* action, float* behaviour_prob) {
	unsigned int block_start = blockIdx.x * num_output;
    unsigned int start = block_start + threadIdx.x;
	unsigned int end = block_start + num_output;

	// step 0. get max_x
	float max_x = CUDA_FLOAT_INF_NEG;
	for (int i = start; i < end; i += blockDim.x) {
        max_x = max(max_x, behaviour[i]);
	}

    static __shared__ float s_max_x;
    float reduced_max_x = blockReduceMax<float>(max_x);
	if (threadIdx.x == 0) {
        s_max_x = reduced_max_x;
    }
	__syncthreads();

	// step 1. compute log(sum(exp(x - max_x))) + max_x
    static __shared__ float s_sum_exp_x;
	float sum_exp_x = 0.0;
	for (int i = start; i < end; i += blockDim.x) {
        sum_exp_x += std::exp(behaviour[i] - s_max_x);
	}
    float reduced_sum_exp_x = blockReduceSum<float>(sum_exp_x);
    if (threadIdx.x == 0) {
        s_sum_exp_x = reduced_sum_exp_x;
    }
    __syncthreads();

	for (int i = start; i < end; i += blockDim.x) {
        if ((i - block_start) == action[blockIdx.x]) {
            float log_sum_exp_x = std::log(s_sum_exp_x) + s_max_x;
            behaviour_prob[blockIdx.x] = behaviour[i] - log_sum_exp_x;
        }
    }
}

__global__ void computeImportanceWeights(unsigned int time_step, unsigned int batch_size,
        const float* target_output_prob, const float* behaviour_output_prob, float* is) {
    unsigned int gid = threadIdx.x + blockIdx.x*blockDim.x;
    if (gid < time_step * batch_size) {
        is[gid] = std::exp(target_output_prob[gid] - behaviour_output_prob[gid]);
    }
}

void __global__ vtraceNStepReturn(unsigned int time_step, unsigned int batch_size,
        float gamma, float lambda, float rho_clip_ratio, float c_clip_ratio,
        const float* is, const float* reward, const float* value, float* ret) {
    unsigned int gid = threadIdx.x + blockIdx.x*blockDim.x;

    if (gid < batch_size) {
        float item = 0;
        for (int t = time_step - 1; t >= 0; --t) {
            float is_data = is[t * batch_size + gid];
            float clipped_rho = min(is_data, rho_clip_ratio);
            float clipped_c = min(is_data, c_clip_ratio);
            float reward_data = reward[t * batch_size + gid];
            float value_data = value[t * batch_size + gid];
            float value_plus1_data = value[(t + 1) * batch_size + gid];
            float delta = clipped_rho * (reward_data + gamma * value_plus1_data - value_data);
            item = delta + gamma * lambda * clipped_c * item;
            ret[t * batch_size + gid] = value_data + item;
        }
    }
}

void __global__ vtraceAdvantage(unsigned int time_step, unsigned int batch_size, float gamma, float rho_pg_clip_ratio,
        const float* is, const float* reward, const float* value, const float* ret, float* adv) {
    unsigned int gid = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int ts_id = gid / batch_size;
    unsigned int batch_id = gid % batch_size;
    if (gid < time_step * batch_size) {
        float is_data = is[gid];
        float clipped_pg_rho = min(is_data, rho_pg_clip_ratio);
        float reward_data = reward[gid];
        float value_data = value[gid];
        float ret_data = (ts_id == (time_step - 1)) ? value[time_step * batch_size + batch_id]: ret[(ts_id + 1) * batch_size + batch_id];
        adv[gid] = clipped_pg_rho * (reward_data + gamma * ret_data - value_data);
    }
}

void __global__ vtraceLoss(unsigned int time_step, unsigned int batch_size,
        const float* value, const float* target_output_prob, const float* entropy,
        const float* ret, const float* adv, const float* weight,
        float* pg_loss, float* value_loss, float* entropy_loss) {
    unsigned int gid = threadIdx.x + blockIdx.x*blockDim.x;

    float pg = 0.f;
    float val = 0.f;
    float entr = 0.f;
    if (gid < time_step * batch_size) {
        pg = -(target_output_prob[gid] * adv[gid] * weight[gid]);

        float diff = value[gid] - ret[gid];
        val = diff * diff * weight[gid];

        entr = entropy[gid] * weight[gid];
    }
    
    float sum_pg = blockReduceSum<float>(pg);
    float sum_val = blockReduceSum<float>(val);
    float sum_entropy = blockReduceSum<float>(entr);
    if (threadIdx.x == 0) {
        atomicAdd(pg_loss, sum_pg / (time_step * batch_size));
        atomicAdd(value_loss, sum_val / (time_step * batch_size));
        atomicAdd(entropy_loss, sum_entropy / (time_step * batch_size));
    }
}

void __global__ vtraceBackwardValue(unsigned int time_step, unsigned int batch_size,
        const float* grad_value_loss, const float* value,
        const float* ret, const float* weight, float* grad_value) {
    unsigned int gid = threadIdx.x + blockIdx.x*blockDim.x;
    if (gid < (time_step + 1) * batch_size) {
        grad_value[gid] = (gid < time_step * batch_size) ?
            ((*grad_value_loss) * ((1.f / (time_step * batch_size)) * 2 * (value[gid] - ret[gid]) * weight[gid])) : 0.f;
    }
}

void __global__ vtraceBackwardTargetOutput(unsigned int time_step, unsigned int batch_size, unsigned int num_output,
        const float* grad_entropy_loss, const float* grad_pg_loss,
        const float* grad_logits, const float* grad_entropy, const float* grad_prob,
        const float* adv, const float* weight, float* grad_target_output) {
	unsigned int block_start = blockIdx.x * num_output;
    unsigned int start = block_start + threadIdx.x;
	unsigned int end = block_start + num_output;

    float weight_data = weight[blockIdx.x];
    float adv_data = adv[blockIdx.x];

    // get pre entropy grad: mean->multiply_weight
    float pre_entropy_grad = *grad_entropy_loss;
    pre_entropy_grad = pre_entropy_grad * (1.f / (time_step * batch_size)) * weight_data;

	// get sum_grad_entropy
    // grad: mean->multiply_weight->(-sum(x_multiply_softmax_x))
	float grad_entropy_val = 0.f;
	for (int i = start; i < end; i += blockDim.x) {
        grad_entropy_val += pre_entropy_grad * grad_entropy[i];
	}
    static __shared__ float s_grad_entropy_val;
    float reduced_grad_entropy_val = blockReduceSum<float>(grad_entropy_val);
	if (threadIdx.x == 0) {
        s_grad_entropy_val = reduced_grad_entropy_val;
    }
	__syncthreads();

    // get pre pg grad: mean->multiply_weight->multiply_adv
    float pre_pg_grad = *grad_pg_loss;
    pre_pg_grad = pre_pg_grad * (-1.f / (time_step * batch_size)) * weight_data * adv_data;

	for (int i = start; i < end; i += blockDim.x) {
        float pg_bp = pre_pg_grad * grad_prob[i];
        // bp of: x - logsumexp(x), is: b_i - grad_logsumexp_i * sum_b
        float entropy_bp = pre_entropy_grad * grad_entropy[i] - grad_logits[i] * s_grad_entropy_val;
        grad_target_output[i] = entropy_bp + pg_bp;
    }
}

}  // namespace cuda
}  // namespace rll
}  // namespace hpc
#endif // HPC_RLL_CUDA_VTRACE_KERNEL_H_
