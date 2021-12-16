#include "hpc/rll/cuda/torch_utils/network/entry.h"
#include "hpc/rll/cuda/torch_utils/network/scatter_connection_kernel.h"

namespace hpc {
namespace rll {
namespace cuda {

void ScatterConnectionForward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs,
    const char* scatter_type) {

    unsigned int index = 0;
    const torch::Tensor& in = inputs[index++];
    const torch::Tensor& loc = inputs[index++];
    index = 0;
    torch::Tensor& out = outputs[index++];

	const unsigned int B = in.size(0);
	const unsigned int M = in.size(1);
	const unsigned int N = in.size(2);
	const unsigned int H = out.size(2);
	const unsigned int W = out.size(3);

    // forward kernel is launched according to input size, some output element will not be set value
    unsigned int out_size = B * N * H * W;
    checkCudaErr(cudaMemsetAsync((float*)(out.data_ptr()), 0, out_size * sizeof(float)));

    if (std::string(scatter_type) == "cover") {
        /*
        // consider that the input data may overlap, keep sequence like cpu when cover data
        // note: even thought this kernel is launched according to output size,
        // there still may be some output element will not be set value according to the mapping index
        dim3 block_size = WARP_SIZE;
        dim3 grid_size = B * N * H * W;
        scatterConnectionCoverKeepSeqForwardKernel<<<grid_size, block_size>>>(
        B, M, N, H, W, (float*)(in.data_ptr()), (int64_t*)(loc.data_ptr()), (float*)(out.data_ptr()));
         */
        dim3 block_size = {WARP_SIZE, DEFAULT_WARP_NUM, 1};
        dim3 grid_size = {(N + block_size.x - 1) / block_size.x, (M + block_size.y - 1) / block_size.y, B};
        scatterConnectionCoverForwardKernel<<<grid_size, block_size>>>(
                B, M, N, H, W, (float*)(in.data_ptr()), (int64_t*)(loc.data_ptr()), (float*)(out.data_ptr()));
    } else if (std::string(scatter_type) == "add") {
        dim3 block_size = {WARP_SIZE, DEFAULT_WARP_NUM, 1};
        dim3 grid_size = {(N + block_size.x - 1) / block_size.x, (M + block_size.y - 1) / block_size.y, B};
        scatterConnectionAddForwardKernel<<<grid_size, block_size>>>(
                B, M, N, H, W, (float*)(in.data_ptr()), (int64_t*)(loc.data_ptr()), (float*)(out.data_ptr()));
    }
}

void ScatterConnectionBackward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs) {

    unsigned int index = 0;
    const torch::Tensor& grad_out = inputs[index++];
    const torch::Tensor& loc = inputs[index++];
    index = 0;
    torch::Tensor& grad_in = outputs[index++];

	const unsigned int B = grad_in.size(0);
	const unsigned int M = grad_in.size(1);
	const unsigned int N = grad_in.size(2);
	const unsigned int H = grad_out.size(2);
	const unsigned int W = grad_out.size(3);

    dim3 block_size = {WARP_SIZE, DEFAULT_WARP_NUM, 1};
    dim3 grid_size = {(N + block_size.x - 1) / block_size.x, (M + block_size.y - 1) / block_size.y, B};
    scatterConnectionBackwardKernel<<<grid_size, block_size>>>(
            B, M, N, H, W, (float*)(grad_out.data_ptr()), (int64_t*)(loc.data_ptr()), (float*)(grad_in.data_ptr()));
}

}  // namespace cuda
}  // namespace rll
}  // namespace hpc
