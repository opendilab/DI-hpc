#include "hpc/rll/cuda/models/entry.h"
#include "hpc/rll/cuda/models/actor_critic_kernel.h"

namespace hpc {
namespace rll {
namespace cuda {

void actor_critic_update_ae(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs) {
    unsigned index = 0;
    const torch::Tensor& key_embeddings = inputs[index++];
    const torch::Tensor& sample_result = inputs[index++];
    const torch::Tensor& true_entity_num = inputs[index++];
    index = 0;
    torch::Tensor& autoregressive_embedding = outputs[index++];

    int64_t batch_size = key_embeddings.size(0);
    int64_t entity_num = key_embeddings.size(1);
    int64_t input_dim = key_embeddings.size(2);
    {
        dim3 block_size = {DEFAULT_WARP_NUM * WARP_SIZE, 1, 1};
        unsigned int grid_size_x = (input_dim + block_size.x - 1) / block_size.x;
        unsigned int grid_size_y = batch_size;
        dim3 grid_size = {grid_size_x, grid_size_y, 1};
        autoregressive_embedding_fp<<<grid_size, block_size>>>(batch_size, entity_num, input_dim,
                sample_result.data_ptr<int64_t>(), true_entity_num.data_ptr<int64_t>(),
                key_embeddings.data_ptr<float>(), autoregressive_embedding.data_ptr<float>());
    }
}

void actor_critic_lstm_activation(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs) {
    unsigned index = 0;

    const torch::Tensor& lstm_ih = inputs[index++];
    const torch::Tensor& lstm_hh = inputs[index++];
    const torch::Tensor& lstm_bias = inputs[index++];
    index = 0;
    torch::Tensor& lstm_hx = outputs[index++];
    torch::Tensor& lstm_cx = outputs[index++];

    int64_t batch_size = lstm_ih.size(0);
    int64_t hidden_size = lstm_ih.size(1) / 4;
    {
        dim3 block_size = {DEFAULT_WARP_NUM * WARP_SIZE, 1, 1};
        unsigned int grid_size_x = (hidden_size + block_size.x - 1) / block_size.x;
        unsigned int grid_size_y = batch_size;
        dim3 grid_size = {grid_size_x, grid_size_y, 1};
        lstm_activation_fp<<<grid_size, block_size>>>(batch_size, hidden_size,
                lstm_ih.data_ptr<float>(), lstm_hh.data_ptr<float>(), lstm_bias.data_ptr<float>(),
                lstm_hx.data_ptr<float>(), lstm_cx.data_ptr<float>());
    }
}

void actor_critic_pre_sample(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs) {
    unsigned index = 0;

    const torch::Tensor& mat = inputs[index++];
    const torch::Tensor& vec = inputs[index++];
    const torch::Tensor& mask = inputs[index++];
    index = 0;
    torch::Tensor& output = outputs[index++];

    int64_t batch_size = mat.size(0);
    int64_t entity_num = mat.size(1);
    int64_t hidden_size = mat.size(2);
    {
        dim3 block_size = {WARP_SIZE, 1, 1};
        unsigned int grid_size_x = 1;
        unsigned int grid_size_y = batch_size * entity_num;
        dim3 grid_size = {grid_size_x, grid_size_y, 1};
        const float mask_value = -1e9;
        const float div_factor = 0.8;
        pre_sample_fp<<<grid_size, block_size>>>(batch_size, entity_num, hidden_size, mask_value, div_factor,
                mat.data_ptr<float>(), vec.data_ptr<float>(), mask.data_ptr<bool>(),
                output.data_ptr<float>());
    }
}

}  // namespace cuda
}  // namespace rll
}  // namespace hpc
