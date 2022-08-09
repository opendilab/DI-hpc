#ifndef HPC_RLL_CUDA_LOSS_H_
#define HPC_RLL_CUDA_LOSS_H_

#include "hpc/rll/cuda/common.h"

namespace hpc {
namespace rll {
namespace cuda {
std::vector<std::vector<int>> sample_split_group(const std::vector<torch::Tensor>& x, int group);
std::vector<std::vector<int>> oracle_split_group(const std::vector<torch::Tensor>& x, int group);

void SOFTARGMAXBackward(
    std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs);

void SOFTARGMAXForward(
    std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs);

void COMAForward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs,
    float gamma,
    float lambda_);

void COMABackward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs);

void RetraceForward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs,
    float gamma);

std::vector<torch::Tensor> Pad1DForward(
    const std::vector<torch::Tensor>& inputs, 
    const int& value);

std::vector<std::vector<torch::Tensor>> GroupPad1DForward(
    const std::vector<torch::Tensor>& inputs, 
    const std::vector<int>& group_cnt,
    const std::vector<int>& max_shape, 
    const std::vector<int>& group_id, 
    const std::vector<int>& group_idx, 
    const int& value);

std::vector<torch::Tensor> Unpad1DForward(
    const torch::Tensor& inputs, 
    const std::vector<int>& shape);

std::vector<torch::Tensor> Pad2DForward(
    const std::vector<torch::Tensor>& inputs, 
    const int& value);

std::vector<std::vector<torch::Tensor>> GroupPad2DForward(
    const std::vector<torch::Tensor>& inputs, 
    const std::vector<int>& group_cnt,
    const std::vector<int>& max_shape, 
    const std::vector<int>& group_id, 
    const std::vector<int>& group_idx, 
    const int& value);

std::vector<torch::Tensor> Unpad2DForward(
    const torch::Tensor& inputs, 
    const std::vector<int>& shape);

std::vector<torch::Tensor> Pad3DForward(
    const std::vector<torch::Tensor>& inputs, 
    const int& value);

std::vector<std::vector<torch::Tensor>> GroupPad3DForward(
    const std::vector<torch::Tensor>& inputs, 
    const std::vector<int>& group_cnt,
    const std::vector<int>& max_shape, 
    const std::vector<int>& group_id, 
    const std::vector<int>& group_idx, 
    const int& value);

std::vector<torch::Tensor> Unpad3DForward(
    const torch::Tensor& inputs, 
    const std::vector<int>& shape);

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

void PPOContinuousForward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs,
    bool use_value_clip,
    float clip_ratio,
    float dual_clip);

void PPOContinuousBackward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs);

void GRUForward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs,
    int TB,
    int input_dim);

void GRUBackward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs,
    int TB,
    int input_dim);
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
