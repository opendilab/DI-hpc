#ifndef HPC_RLL_CUDA_PPO_KERNEL_H_
#define HPC_RLL_CUDA_PPO_KERNEL_H_

#include "hpc/rll/cuda/common.h"
#include "hpc/rll/cuda/reduce.h"
#include "hpc/rll/cuda/basic_math.h"

namespace hpc {
namespace rll {
namespace cuda {

__global__ void categoricalProbEntropy(unsigned int num_output, const float* x, const int64_t* action,
        float* prob, float* entropy, float* grad_logits, float* grad_prob, float* grad_entropy) {
	unsigned int block_start = blockIdx.x * num_output;
    unsigned int start = block_start + threadIdx.x;
	unsigned int end = block_start + num_output;

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
	for (int i = start; i < end; i += blockDim.x) {
        float logits = x[i] - log_sum_exp_x;
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
        float val = x[i];
        float logits = val - log_sum_exp_x;
        float softmax_logits = std::exp(logits - s_max_logits) / s_sum_exp_logits;

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
}

__global__ void categoricalProb(unsigned int num_output, const float* x, const int64_t* action, float* prob) {
	unsigned int block_start = blockIdx.x * num_output;
    unsigned int start = block_start + threadIdx.x;
	unsigned int end = block_start + num_output;

	// step 0. get max_x
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

	// step 1. compute log(sum(exp(x - max_x))) + max_x
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

	for (int i = start; i < end; i += blockDim.x) {
        if ((i - block_start) == action[blockIdx.x]) {
            float log_sum_exp_x = std::log(s_sum_exp_x) + s_max_x;
            prob[blockIdx.x] = x[i] - log_sum_exp_x;
        }
    }
}

__global__ void ppoLoss(unsigned int batch_size, const float* value_new, const float* value_old,
        const float* logits_new_prob, const float* logits_old_prob, const float* logits_new_entropy,
        const float* advantage, const float* return_, const float* weight,
        bool use_value_clip, float clip_ratio, float dual_clip,
        float* policy_loss, float* value_loss, float* entropy_loss, float* approx_kl, float* clipfrac,
        float* grad_policy_loss_buf, float* grad_value_loss_buf, float* grad_entropy_loss_buf) {
    unsigned int gid = threadIdx.x + blockIdx.x*blockDim.x; // batch_size
    float scale = 1.f / batch_size;

    float policy_loss_val = 0.f;
    float value_loss_val = 0.f;
    float entropy_loss_val = 0.f;
    float approx_kl_val = 0.f;
    float clipfrac_val = 0.f;
    if (gid < batch_size) {
        float w = weight[gid];
        float adv = advantage[gid];

        // entropy loss
        entropy_loss_val = logits_new_entropy[gid] * w;
        grad_entropy_loss_buf[gid] = w * scale;

        // policy_loss
        float diff_prob = logits_new_prob[gid] - logits_old_prob[gid];
        float ratio = std::exp(diff_prob);
        bool ratio_clamp_flag = (ratio >= (1 - clip_ratio) && ratio <= (1 + clip_ratio));

        float surr1 = ratio * adv;
        float surr2 = clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv; // TODO basic_math.h, clamp

        float grad_ratio = ratio;
        float grad_surr1 = grad_ratio * adv;
        float grad_surr2 = grad_ratio * adv * ratio_clamp_flag;

        float min_surr = (surr1 <= surr2) ? surr1 : surr2;
        float grad_min_surr = (surr1 <= surr2) ? grad_surr1 : grad_surr2;
        if (dual_clip < 1.f) {
            policy_loss_val = -min_surr * w;
            grad_policy_loss_buf[gid] = (-grad_min_surr) * w * scale;
        } else {
            float dual_clip_adv = dual_clip * adv;
            float max_val = max(min_surr, dual_clip_adv);
            policy_loss_val = -max_val * w;
            if (min_surr >= dual_clip_adv) {
                grad_policy_loss_buf[gid] = (-grad_min_surr) * w * scale;
            } else {
                grad_policy_loss_buf[gid] = 0;
            }
        }

        // monitor info
        approx_kl_val = -diff_prob;
        if (!ratio_clamp_flag) clipfrac_val = 1.f;

        // value loss
        float diff_v_r = value_new[gid] - return_[gid];
        float v_r_squre = diff_v_r * diff_v_r;
        if (use_value_clip) {
            float value_diff = value_new[gid] - value_old[gid];
            float value_clip = value_old[gid] + clamp(value_diff, -clip_ratio, clip_ratio);
            bool value_diff_clamp_flag = (value_diff >= -clip_ratio && value_diff <= clip_ratio);

            float diff_vclip_r = value_clip - return_[gid];
            float vclip_r_squre = diff_vclip_r * diff_vclip_r;
            value_loss_val = 0.5 * (max(v_r_squre, vclip_r_squre) * w);

            if (v_r_squre >= vclip_r_squre) {
                grad_value_loss_buf[gid] = 0.5 * (2 * diff_v_r) * w * scale;
            } else {
                grad_value_loss_buf[gid] = 0.5 * (2 * diff_vclip_r * value_diff_clamp_flag) * w * scale;
            }
        } else {
            value_loss_val = 0.5 * v_r_squre * w;
            grad_value_loss_buf[gid] = 0.5 * (2 * diff_v_r) * w * scale;
        }
    }

    // mean
    float sum_policy_loss = blockReduceSum<float>(policy_loss_val);
    float sum_value_loss = blockReduceSum<float>(value_loss_val);
    float sum_entropy_loss = blockReduceSum<float>(entropy_loss_val);
    float sum_approx_kl = blockReduceSum<float>(approx_kl_val);
    float sum_clipfrac = blockReduceSum<float>(clipfrac_val);
    if (threadIdx.x == 0) {
        atomicAdd(policy_loss, sum_policy_loss * scale);
        atomicAdd(value_loss, sum_value_loss * scale);
        atomicAdd(entropy_loss, sum_entropy_loss * scale);
        atomicAdd(approx_kl, sum_approx_kl * scale);
        atomicAdd(clipfrac, sum_clipfrac * scale);
    }
}

void __global__ ppoBackwardValueNew(unsigned int batch_size,
        const float* grad_value_loss, const float* grad_value_loss_buf, float* grad_value) {
    unsigned int gid = threadIdx.x + blockIdx.x*blockDim.x;
    if (gid < batch_size) {
        grad_value[gid] = (*grad_value_loss) * grad_value_loss_buf[gid];
    }
}

void __global__ ppoBackwardLogitsNew(unsigned int batch_size, unsigned int num_output,
        const float* grad_policy_loss, const float* grad_entropy_loss,
        const float* grad_policy_loss_buf, const float* grad_entropy_loss_buf,
        const float* grad_logits, const float* grad_prob, const float* grad_entropy,
        float* grad_logits_new) {
	unsigned int block_start = blockIdx.x * num_output;
    unsigned int start = block_start + threadIdx.x;
	unsigned int end = block_start + num_output;

    float pre_entropy_grad = (*grad_entropy_loss) * grad_entropy_loss_buf[blockIdx.x];
    float pre_policy_grad = (*grad_policy_loss) * grad_policy_loss_buf[blockIdx.x];

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

	for (int i = start; i < end; i += blockDim.x) {
        float policy_bp = pre_policy_grad * grad_prob[i];
        // bp of: x - logsumexp(x), is: b_i - grad_logsumexp_i * sum_b
        float entropy_bp = pre_entropy_grad * grad_entropy[i] - grad_logits[i] * s_grad_entropy_val;
        grad_logits_new[i] = entropy_bp + policy_bp;
    }
}

}  // namespace cuda
}  // namespace rll
}  // namespace hpc
#endif // HPC_RLL_CUDA_PPO_KERNEL_H_
