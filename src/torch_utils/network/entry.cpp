#include <torch/extension.h>
#include "hpc/rll/cuda/torch_utils/network/entry.h"

namespace hpc {
namespace rll {
namespace cuda {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("LstmForward", &LstmForward, "lstm forward (CUDA)");
    m.def("LstmBackward", &LstmBackward, "lstm backward (CUDA)");
    m.def("ScatterConnectionForward", &ScatterConnectionForward, "scatter_connection forward (CUDA)");
    m.def("ScatterConnectionBackward", &ScatterConnectionBackward, "scatter_connection backward (CUDA)");
}

}  // namespace cuda
}  // namespace rll
}  // namespace hpc
