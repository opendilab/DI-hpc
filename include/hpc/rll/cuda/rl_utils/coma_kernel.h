#ifndef HPC_RLL_CUDA_COMA_KERNEL_H_
#define HPC_RLL_CUDA_COMA_KERNEL_H_

#include "hpc/rll/cuda/common.h"
#include "hpc/rll/cuda/reduce.h"
#include "hpc/rll/cuda/basic_math.h"

namespace hpc {
namespace rll {
namespace cuda {

__global__ void COMAGather(unsigned int N, unsigned int B, unsigned int A, const float* q_value, 
        const float* target_q_value, const int64_t* action, float* q_taken, float* target_q_taken, const float* reward, float* reward_new) {
    unsigned int block_start = blockIdx.x * B * A;
    unsigned int start = block_start + threadIdx.x;
	unsigned int end = block_start + B * A;
    for(int i = start; i < end; i += blockDim.x) {
        int index = action[i];
        q_taken[i] = q_value[i * N + index];
        target_q_taken[i] = target_q_value[i * N + index];
        reward_new[i] = reward[(i - block_start) / A + blockIdx.x * B];
    }
}

__global__ void ProbEntropyAdv(unsigned int N, const float* x, const int64_t* action, float* q_value, float* q_taken, 
        float* prob, float* adv, float* entropy, float* grad_logits, float* grad_prob, float* grad_entropy) {
    unsigned int block_start = blockIdx.x * N;
    unsigned int start = block_start + threadIdx.x;
	unsigned int end = block_start + N;
    // step 1: logits = x - logsumexp(x)
	// step 1.1 get max_x
	float max_x = CUDA_FLOAT_INF_NEG;
	for (int i = start; i < end; i += blockDim.x) {
        max_x = max(max_x, x[i]);
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
        sum_exp_x += std::exp(x[i] - s_max_x);
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
        float logits = x[i] - log_sum_exp_x;
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
        float logits = x[i] - log_sum_exp_x;
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
    static __shared__ float s_sum_grad_adv_val;
    float sum_grad_adv_val = 0.f;
	for (int i = start; i < end; i += blockDim.x) {
        float logits = x[i] - log_sum_exp_x;
        float softmax_logits = std::exp(logits - s_max_logits) / s_sum_exp_logits;
        sum_entropy_val += logits * softmax_logits;
        sum_grad_adv_val += q_value[i] * softmax_logits;
	}
    float reduced_sum_entropy_val = blockReduceSum<float>(sum_entropy_val);
    float reduced_sum_grad_adv_val = blockReduceSum<float>(sum_grad_adv_val);
    if (threadIdx.x == 0) { 
        s_sum_entropy_val = reduced_sum_entropy_val;
        s_sum_grad_adv_val = reduced_sum_grad_adv_val;
    }
    __syncthreads();

    // step 3. output
    // output prob, entropy, grad_logits, grad_prob, grad_entropy
    // grad_entropy[i] = (-1) * softmax(logits[i]) * (1 + logits[i] - sum(logits * softmax(logits)))
    static __shared__ float s_sum_baseline_val;
    float sum_baseline_val = 0.f;
	for (int i = start; i < end; i += blockDim.x) {
        bool flag = ((i - block_start) == action[blockIdx.x]);
        float val = x[i];
        float logits = val - log_sum_exp_x;
        float softmax_logits = std::exp(logits - s_max_logits) / s_sum_exp_logits;

        sum_baseline_val += softmax_logits * q_value[i];

        if (flag)
            prob[blockIdx.x] = val - log_sum_exp_x;

        entropy[blockIdx.x] = -s_sum_entropy_val;

        // grad of logsumexp(x)
        float grad = std::exp(val - s_max_x) / s_sum_exp_x;
        grad_logits[i] = grad;

        // grad of x - logsumexp(x)
        grad_prob[i] = (flag ? 1 : 0) - grad;

        // grad of -sum(logits * softmax(logits))
        grad_entropy[i] = (-1.f) * softmax_logits * (1 + logits - s_sum_entropy_val);

    }

    float reduced_sum_baseline_val = blockReduceSum<float>(sum_baseline_val);
    if (threadIdx.x == 0) { 
        s_sum_baseline_val = reduced_sum_baseline_val;
    }
    __syncthreads();
    adv[blockIdx.x] = q_taken[blockIdx.x] - s_sum_baseline_val;

}

__global__ void GeneralizedLambdaReturns(unsigned int T, unsigned int BA, const float* target_q_taken, const float* rewards, float* return_, float gamma, float lambda_) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < BA) {
        return_[(T - 2) * BA + tid] = rewards[(T - 2) * BA + tid] + gamma * target_q_taken[(T - 1) * BA + tid];
        for(int t = T - 3; t >= 0; t--) {
            return_[t * BA + tid] = rewards[t * BA + tid] + gamma * lambda_ * return_[(t + 1) * BA + tid] + (gamma - gamma * lambda_) * target_q_taken[(t + 1) * BA + tid];
        }
    }
}


__global__ void COMALoss(unsigned int T, unsigned int BA, const float* q_taken, const float* prob, const float* entropy, const float* adv, const float* return_,
                        const float* weight, float* policy_loss, float* value_loss, float* entropy_loss, float* grad_policy_loss_buf, float* grad_value_loss_buf,
                        float* grad_entropy_loss_buf) {
    int gid = threadIdx.x + blockIdx.x*blockDim.x;
    float scale = 1.f / (T * BA);
    float scale_v = 1.f / ((T - 1) * BA);
    
    float policy_loss_val = 0.f;
    float value_loss_val = 0.f;
    float entropy_loss_val = 0.f;

    if(gid < T * BA) {
        float w = weight[gid];
        if(abs(w) <= 1e-6) {
            w = 1;
        }
        entropy_loss_val = entropy[gid] * w;
        policy_loss_val = - adv[gid] * prob[gid] * w;

        grad_entropy_loss_buf[gid] = w * scale;
        grad_policy_loss_buf[gid] = w * scale;
        grad_value_loss_buf[gid] = 0;
        if(gid < (T - 1) * BA) {
            value_loss_val = pow((q_taken[gid] - return_[gid]), 2) * w;
            grad_value_loss_buf[gid] = 2 * (q_taken[gid] - return_[gid]) * w * scale_v;
        }
    }
    // mean
    float sum_policy_loss = blockReduceSum<float>(policy_loss_val);
    float sum_value_loss = blockReduceSum<float>(value_loss_val);
    float sum_entropy_loss = blockReduceSum<float>(entropy_loss_val);
    if (threadIdx.x == 0) {
        atomicAdd(policy_loss, sum_policy_loss * scale);
        atomicAdd(value_loss, sum_value_loss * scale_v);
        atomicAdd(entropy_loss, sum_entropy_loss * scale);
    }
}


__global__ void COMABackward(unsigned int N, const float* grad_policy_loss, const float* grad_value_loss, const float* grad_entropy_loss, 
                            const float* grad_policy_loss_buf, const float* grad_value_loss_buf, const float* grad_entropy_loss_buf, const float* adv,
                            const int64_t* action,  const float* logits_grad_logits, const float* logits_grad_prob,
                            const float* logits_grad_entropy, float* grad_q_value, float* grad_logits) {
    unsigned int block_start = blockIdx.x * N;
    unsigned int start = block_start + threadIdx.x;
	unsigned int end = block_start + N;

    float pre_entropy_grad = (*grad_entropy_loss) * grad_entropy_loss_buf[blockIdx.x];
    float pre_policy_grad = (*grad_policy_loss) * grad_policy_loss_buf[blockIdx.x];

    // get sum_grad_entropy
    // grad: mean->multiply_weight->(-sum(x_multiply_softmax_x))
	float grad_entropy_val = 0.f;
	for (int i = start; i < end; i += blockDim.x) {
        grad_entropy_val += pre_entropy_grad * logits_grad_entropy[i];
	}
    static __shared__ float s_grad_entropy_val;
    float reduced_grad_entropy_val = blockReduceSum<float>(grad_entropy_val);
	if (threadIdx.x == 0) {
        s_grad_entropy_val = reduced_grad_entropy_val;
    }
	__syncthreads();

	for (int i = start; i < end; i += blockDim.x) {
        bool flag = ((i - block_start) == action[blockIdx.x]);
        float policy_bp_logit = (-1) * pre_policy_grad * logits_grad_prob[i] * adv[blockIdx.x];
        // bp of: x - logsumexp(x), is: b_i - grad_logsumexp_i * sum_b
        float entropy_bp = pre_entropy_grad * logits_grad_entropy[i] - logits_grad_logits[i] * s_grad_entropy_val;
        grad_logits[i] = policy_bp_logit + entropy_bp;

        float q_value_bp = (flag ? 1 : 0) * (*grad_value_loss) * grad_value_loss_buf[blockIdx.x];

        grad_q_value[i] = q_value_bp; // + policy_bp_q_value;
    }         
}

   




}  // namespace cuda
}  // namespace rll
}  // namespace hpc
#endif // HPC_RLL_CUDA_COMA_KERNEL_H_
