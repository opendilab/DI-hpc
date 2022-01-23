#ifndef HPC_RLL_CUDA_LOSS_H_
#define HPC_RLL_CUDA_LOSS_H_

#include "hpc/rll/cuda/common.h"

namespace hpc {
namespace rll {
namespace cuda {

void Pad1DForward(
    const std::vector<torch::Tensor>& inputs, 
    const torch::Tensor& shape,
    torch::Tensor& new_x, 
    torch::Tensor& mask, 
    const int& max_shape, 
    const int& value);

void GroupPad1DForward(
    const std::vector<torch::Tensor>& inputs, 
    const torch::Tensor& shape,
    std::vector<torch::Tensor>& new_x, 
    std::vector<torch::Tensor>& mask, 
    const torch::Tensor& max_shape, 
    const torch::Tensor& group_id, 
    const torch::Tensor* group_idx, 
    const int& value);

void Unpad1DForward(
    const torch::Tensor& inputs, 
    const torch::Tensor& shape, 
    std::vector<torch::Tensor>& outputs);

void Pad2DForward(
    const std::vector<torch::Tensor>& inputs, 
    const torch::Tensor& shape, 
    torch::Tensor& new_x,
    torch::Tensor& mask, 
    const int& max_shape0, 
    const int& max_shape1, 
    const int& value);

void GroupPad2DForward(
    const std::vector<torch::Tensor>& inputs, 
    const torch::Tensor& shape,
    std::vector<torch::Tensor>& new_x, 
    std::vector<torch::Tensor>& mask, 
    const torch::Tensor& max_shape, 
    const torch::Tensor& group_id, 
    const torch::Tensor* group_idx, 
    const int& value);

void Unpad2DForward(
    const torch::Tensor& inputs, 
    const torch::Tensor& shape, 
    std::vector<torch::Tensor>& outputs);

void Pad3DForward(
    const std::vector<torch::Tensor>& inputs, 
    torch::Tensor& shape, 
    torch::Tensor* new_x,
    torch::Tensor& mask, 
    const int max_shape0, 
    const int max_shape1, 
    const int max_shape2, 
    const int& value);

void GroupPad3DForward(
    const std::vector<torch::Tensor>& inputs, 
    const torch::Tensor& shape,
    std::vector<torch::Tensor>& new_x, 
    std::vector<torch::Tensor>& mask, 
    const torch::Tensor& max_shape, 
    const torch::Tensor& group_id, 
    const torch::Tensor* group_idx, 
    const int& value);

void Unpad3DForward(
    const torch::Tensor inputs, 
    const torch::Tensor shape, 
    std::vector<torch::Tensor>& outputs);

// gae
void GaeForward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs,
    float gamma,
    float lambda);

// td_lambda
void TdLambdaForward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs,
    float gamma,
    float lambda);

void TdLambdaBackward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs);

// dist_nstep_td
void DistNStepTdForward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs,
    float gamma,
    float v_min,
    float v_max);

void DistNStepTdBackward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs);

// q_nstep_td
void QNStepTdForward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs,
    float gamma);

void QNStepTdBackward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs);

// q_nstep_td_with_rescale
void QNStepTdRescaleForward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs,
    float gamma);

void QNStepTdRescaleBackward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs);

// upgo
void UpgoForward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs);

void UpgoBackward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs);

// vtrace
void VTraceForward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs,
    float gamma,
    float lambda,
    float rho_clip_ratio,
    float c_clip_ratio,
    float rho_pg_clip_ratio);

void VTraceBackward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs);

// ppo
void PPOForward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs,
    bool use_value_clip,
    float clip_ratio,
    float dual_clip);

void PPOBackward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs);

// iqn_nstep_td_error
void IQNNStepTDErrorForward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs,
    float gamma,
    float kappa);

void IQNNStepTDErrorBackward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs);

// qrdqn_nstep_td_error
void QRDQNNStepTDErrorForward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs,
    float gamma);

void QRDQNNStepTDErrorBackward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs);

}  // namespace cuda
}  // namespace rll
}  // namespace hpc

#endif // HPC_RLL_CUDA_LOSS_H_
