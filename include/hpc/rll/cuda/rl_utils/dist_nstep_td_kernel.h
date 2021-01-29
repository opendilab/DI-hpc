#ifndef HPC_RLL_CUDA_DIST_NSTEP_TD_KERNEL_H_
#define HPC_RLL_CUDA_DIST_NSTEP_TD_KERNEL_H_

#include "hpc/rll/cuda/common.h"
#include "hpc/rll/cuda/reduce.h"

namespace hpc {
namespace rll {
namespace cuda {

void __global__ distNStepTdRewardKernel(unsigned int time_step, unsigned int batch_size, float gamma,
        const float* reward, float* reward_buf) {
    unsigned int gid = threadIdx.x + blockIdx.x*blockDim.x; // batch_size

    if (gid < batch_size) {
        unsigned int batch_id = gid;

        float sum_reward = 0;
        float factor = 1;
        for (int t = 0; t < time_step; ++t) {
            float rw = reward[t * batch_size + batch_id];
            sum_reward += (factor * rw);
            factor *= gamma;
        }

        reward_buf[batch_id] = sum_reward;
    }
}

void __global__ distNStepTdProjKernel(unsigned int batch_size, unsigned int action_dim, unsigned int n_atom,
        float gamma_nstep, float v_min, float v_max, float delta,
        const float* next_n_dist, const int64_t* next_n_action,
        const float* reward_buf, const float* done, float* proj_dist) {
    unsigned int gidx = threadIdx.x + blockIdx.x*blockDim.x; // n_atom
    unsigned int gidy = blockIdx.y; // batch_size

    if (gidx < n_atom) {
        unsigned int atom_id = gidx;
        unsigned int batch_id = gidy;

        float reward = reward_buf[batch_id];
        float support = v_min + atom_id * delta;
        float target = reward + (1 - done[batch_id]) * gamma_nstep * support;
        target = min(v_max, target);
        target = max(v_min, target);

        float local_box_id = (target - v_min) / delta;
        unsigned int local_box_id_l = floor(local_box_id);
        unsigned int local_box_id_u = ceil(local_box_id);
        unsigned int global_box_id_l = batch_id * n_atom + local_box_id_l;
        unsigned int global_box_id_u = batch_id * n_atom + local_box_id_u;

        unsigned int next_n_action_id = next_n_action[batch_id];
        float target_dist_sa = next_n_dist[batch_id * action_dim * n_atom + next_n_action_id * n_atom + atom_id];

        float proj_dist_l = target_dist_sa * ((float)local_box_id_u - local_box_id);
        float proj_dist_u = target_dist_sa * (local_box_id - (float)local_box_id_l);
        atomicAdd(&proj_dist[global_box_id_l], proj_dist_l);
        atomicAdd(&proj_dist[global_box_id_u], proj_dist_u);
    }
}

void __global__ distNStepTdLossKernel(unsigned int batch_size, unsigned int action_dim, unsigned int n_atom,
        const float* dist, const int64_t* action, const float* proj_dist, const float* weight,
        float* td_err, float* loss, float* grad_buf) {
    unsigned int gidx = threadIdx.x + blockIdx.x*blockDim.x; // n_atom
    unsigned int gidy = blockIdx.y; // batch_size

    float sum_val = 0;
    float sum_td_err = 0;
    if (gidx < n_atom) {
        unsigned int atom_id = gidx;
        unsigned int batch_id = gidy;

        unsigned int action_id = action[batch_id];
        float dist_sa = dist[batch_id * action_dim * n_atom + action_id * n_atom + atom_id];
        float log_p = log(dist_sa);

        float w = weight[batch_id];
        float proj = proj_dist[batch_id * n_atom + atom_id];
        sum_val = log_p * proj * w;
        sum_td_err = log_p * proj;

        grad_buf[batch_id * n_atom + atom_id] = (-1.f) / (float)batch_size * w * proj * (1.f / dist_sa);
    }

    float reduced_sum_val = blockReduceSum<float>(sum_val);
    float reduced_sum_td_err = blockReduceSum<float>(sum_td_err);
    if (threadIdx.x == 0) {
        td_err[gidy] = reduced_sum_td_err * (-1.f);
        atomicAdd(loss, reduced_sum_val * (-1.f) / (float)batch_size);
    }
}

void __global__ distNStepTdBackwardKernel(unsigned int batch_size, unsigned int action_dim, unsigned int n_atom,
        const float* grad_loss, const float* grad_buf, const int64_t* action, float* grad_dist) {
    unsigned int gid = threadIdx.x + blockIdx.x*blockDim.x;

    if (gid < batch_size * action_dim * n_atom) {
        unsigned int atom_id = gid % n_atom;
        unsigned int action_id = (gid / n_atom) % action_dim;
        unsigned int batch_id = (gid / n_atom) / action_dim;

        float grad = (action_id == action[batch_id]) ? grad_buf[batch_id * n_atom + atom_id] : 0;
        grad_dist[gid] = (*grad_loss) * grad;
    }
}

}  // namespace cuda
}  // namespace rll
}  // namespace hpc
#endif // HPC_RLL_CUDA_DIST_NSTEP_TD_KERNEL_H_
