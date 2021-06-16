#include "hpc/rll/cuda/models/entry.h"

namespace hpc {
namespace rll {
namespace cuda {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("actor_critic_update_ae", &actor_critic_update_ae, "actor critic model update autoregressive embedding (CUDA)");
    m.def("actor_critic_lstm_activation", &actor_critic_lstm_activation, "actor critic model lstm activation (CUDA)");
    m.def("actor_critic_pre_sample", &actor_critic_pre_sample, "actor critic model pre sample (CUDA)");
}

}  // namespace cuda
}  // namespace rll
}  // namespace hpc
