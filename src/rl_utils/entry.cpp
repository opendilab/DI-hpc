#include <torch/extension.h>
#include "hpc/rll/cuda/rl_utils/entry.h"

namespace hpc {
namespace rll {
namespace cuda {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sample_split_group", &sample_split_group, "sample_split_group");
    m.def("oracle_split_group", &oracle_split_group, "oracle_split_group");
    m.def("RetraceForward", &RetraceForward, "Retrace forward (CUDA)");
    m.def("Pad1DForward", &Pad1DForward, "Pad1D forward (CUDA)");
    m.def("GroupPad1DForward", &GroupPad1DForward, "Pad1D forward (CUDA)");
    m.def("Unpad1DForward", &Unpad1DForward, "Unpad1D forward (CUDA)");
    m.def("Pad2DForward", &Pad2DForward, "Pad2D forward (CUDA)");
    m.def("GroupPad2DForward", &GroupPad2DForward, "Pad1D forward (CUDA)");
    m.def("Unpad2DForward", &Unpad2DForward, "Unpad2D forward (CUDA)");
    m.def("Pad3DForward", &Pad3DForward, "Pad2D forward (CUDA)");
    m.def("GroupPad3DForward", &GroupPad3DForward, "Pad1D forward (CUDA)");
    m.def("Unpad3DForward", &Unpad3DForward, "Unpad2D forward (CUDA)");
    m.def("DistNStepTdForward", &DistNStepTdForward, "dist_nstep_td forward (CUDA)");
    m.def("DistNStepTdBackward", &DistNStepTdBackward, "dist_nstep_td backward (CUDA)");
    m.def("GaeForward", &GaeForward, "gae forward (CUDA)");
    m.def("PPOForward", &PPOForward, "ppo forward (CUDA)");
    m.def("PPOBackward", &PPOBackward, "ppo backward (CUDA)");
    m.def("GRUForward", &GRUForward, "gru forward (CUDA)");
    m.def("GRUBackward", &GRUBackward, "gru backward (CUDA)");
    m.def("COMAForward", &COMAForward, "gru forward (CUDA)");
    m.def("COMABackward", &COMABackward, "gru backward (CUDA)");
    m.def("SOFTARGMAXForward", &SOFTARGMAXForward, "softargmax forward (CUDA)");
    m.def("SOFTARGMAXBackward", &SOFTARGMAXBackward, "softargmax backward (CUDA)");
    m.def("PPOContinuousForward", &PPOContinuousForward, "ppo forward (CUDA)");
    m.def("PPOContinuousBackward", &PPOContinuousBackward, "ppo backward (CUDA)");
    m.def("QNStepTdForward", &QNStepTdForward, "q_nstep_td forward (CUDA)");
    m.def("QNStepTdBackward", &QNStepTdBackward, "q_nstep_td backward (CUDA)");
    m.def("QNStepTdRescaleForward", &QNStepTdRescaleForward, "q_nstep_td_with_rescale forward (CUDA)");
    m.def("QNStepTdRescaleBackward", &QNStepTdRescaleBackward, "q_nstep_td_with_rescale backward (CUDA)");
    m.def("TdLambdaForward", &TdLambdaForward, "td_lambda forward (CUDA)");
    m.def("TdLambdaBackward", &TdLambdaBackward, "td_lambda backward (CUDA)");
    m.def("UpgoForward", &UpgoForward, "upgo forward (CUDA)");
    m.def("UpgoBackward", &UpgoBackward, "upgo backward (CUDA)");
    m.def("VTraceForward", &VTraceForward, "vtrace forward (CUDA)");
    m.def("VTraceBackward", &VTraceBackward, "vtrace backward (CUDA)");
    m.def("IQNNStepTDErrorForward", &IQNNStepTDErrorForward, "iqn_nstep_td_error forward (CUDA)");
    m.def("IQNNStepTDErrorBackward", &IQNNStepTDErrorBackward, "iqn_nstep_td_error backward (CUDA)");
    m.def("QRDQNNStepTDErrorForward", &QRDQNNStepTDErrorForward, "qrdqn_nstep_td_error forward (CUDA)");
    m.def("QRDQNNStepTDErrorBackward", &QRDQNNStepTDErrorBackward, "qrdqn_nstep_td_error backward (CUDA)");
}

}  // namespace cuda
}  // namespace rll
}  // namespace hpc

