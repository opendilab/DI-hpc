#include "hpc/rll/cuda/rl_utils/entry.h"
#include "hpc/rll/cuda/rl_utils/padding_kernel.h"

namespace hpc {
namespace rll {
namespace cuda {


void Pad1DForward(const std::vector<torch::Tensor>& inputs, const torch::Tensor& shape,
                 torch::Tensor& new_x, torch::Tensor& mask, const int& max_shape, const int& value) {
    const int n = inputs.size();
    int* shape_ptr = (int*)shape.data_ptr();
    float** inputs_ptr = new float*[n];
    float* new_x_ptr = new_x.data_ptr<float>();
    int* mask_ptr = mask.data_ptr<int>();
    for(int i = 0; i < n; i++)
        inputs_ptr[i] = inputs[i].data_ptr<float>();
    dim3 grid(n, 1, 1);
    dim3 block(GetBlockSize(max_shape));
    Pad1D_kernel<<<grid, block>>>(inputs_ptr, shape_ptr, new_x_ptr, mask_ptr, max_shape, value);
}

void GroupPad1DForward(const std::vector<torch::Tensor>& inputs, const torch::Tensor& shape,
                 std::vector<torch::Tensor>& new_x, std::vector<torch::Tensor>& mask, 
                 const torch::Tensor& max_shape, const torch::Tensor& group_id, 
                 const torch::Tensor* group_idx, const int& value) {
    const int n = inputs.size();
    const int group_num = new_x.size();
    float** new_x_ptr = new float*[group_num];
    float** mask_ptr = new float*[group_num];
    int* max_shape_ptr = max_shape.data_ptr<int>();
    int* shape_ptr = shape.data_ptr<int>();
    int* group_id_ptr = group_id.data_ptr<int>();
    int* group_idx_ptr = group_idx.data_ptr<int>();
    int max_shape_global = 0;
    for(int i = 0; i < group_num; i++) {
        new_x_ptr[i] = new_x[i].data_ptr<float>();
        mask_ptr[i] = mask[i].data_ptr<int>();
        if(max_shape_global < max_shape_ptr[i]) {
            max_shape_global = max_shape_ptr[i];
        }
    }
    dim3 grid(n, 1, 1);
    dim3 block(GetBlockSize(max_shape_global));
    GroupPad1D_kernel<<<grid, block>>>(inputs, shape_ptr, new_x_ptr, mask_ptr, max_shape_ptr, 
                                        group_id_ptr, group_idx_ptr, value);
}

void Unpad1DForward(const torch::Tensor& inputs, const torch::Tensor& shape, std::vector<torch::Tensor>& outputs) {
    const int n = outputs.size();
    int max_shape = 0;
    const int* shape_ptr = shape.data_ptr<int>();
    float** outputs_ptr = new float*[n];
    float* inputs_ptr = inputs.data_ptr<float>();
    for(int i = 0; i < n; i++) {
        outputs_ptr[i] = outputs[i].data_ptr<float>();
        if(max_shape < shape_ptr[i])
            max_shape = shape_ptr[i];
    }
    dim3 grid(n, 1, 1);
    dim3 block(GetBlockSize(max_shape));
    Unpad1D_kernel<<<grid, block>>>(inputs_ptr, shape_ptr, outputs_ptr, max_shape);
}

void Pad2DForward(const std::vector<torch::Tensor>& inputs, const torch::Tensor& shape, torch::Tensor& new_x,
            torch::Tensor& mask, const int& max_shape0, const int& max_shape1, const int& value) {
    const int n = inputs.size();
    float** inputs_ptr = new float*[n];
    int* shape_ptr = shape.data_ptr<int>();
    float* new_x_ptr = new_x.data_ptr<float>();
    int* mask_ptr = mask.data_ptr<int>();
    for(int i = 0; i < n; i++) {
        inputs_ptr[i] = inputs[i].data_ptr<float>();
    }
    dim3 grid(n, 1, 1);
    dim3 block(GetBlockSize(max_shape0), GetBlockSize(max_shape1), 1);
    Pad2D_kernel<<<grid, block>>>(inputs_ptr, shape_ptr, new_x_ptr, mask_ptr, max_shape0, max_shape1, value);
}

void GroupPad2DForward(const std::vector<torch::Tensor>& inputs, const torch::Tensor& shape,
                 std::vector<torch::Tensor>& new_x, std::vector<torch::Tensor>& mask, 
                 const torch::Tensor& max_shape, const torch::Tensor& group_id, 
                 const torch::Tensor* group_idx, const int& value) {
    const int n = inputs.size();
    const int group_num = new_x.size();
    float** new_x_ptr = new float*[group_num];
    float** mask_ptr = new float*[group_num];
    int* max_shape_ptr = max_shape.data_ptr<int>();
    int* shape_ptr = shape.data_ptr<int>();
    int* group_id_ptr = group_id.data_ptr<int>();
    int* group_idx_ptr = group_idx.data_ptr<int>();
    int max_shape_global0 = 0;
    int max_shape_global1 = 0;
    for(int i = 0; i < group_num; i++) {
        new_x_ptr[i] = new_x[i].data_ptr<float>();
        mask_ptr[i] = mask[i].data_ptr<int>();
        if(max_shape_global0 < max_shape_ptr[i * 2]) {
            max_shape_global0 = max_shape_ptr[i * 2];
        }
        if(max_shape_global1 < max_shape_ptr[i * 2 + 1]) {
            max_shape_global1 = max_shape_ptr[i * 2 + 1];
        }
    }
    dim3 grid(n, 1, 1);
    dim3 block(GetBlockSize(max_shape_global1), GetBlockSize(max_shape_global0));
    GroupPad1D_kernel<<<grid, block>>>(inputs, shape_ptr, new_x_ptr, mask_ptr, max_shape_ptr, 
                                        group_id_ptr, group_idx_ptr, value);
}

void Unpad2DForward(const torch::Tensor& inputs, const torch::Tensor& shape, std::vector<torch::Tensor>& outputs) {
    int max_shape0 = 0;
    int max_shape1 = 0;
    const int n = inputs.size();
    float* inputs_ptr = inputs.data_ptr<float>();
    float** outputs_ptr = new float*[n]
    const int* shape_ptr = (int*)shape.data_ptr();
    for(int i = 0; i < n; i++) {
        outputs_ptr[i] = outputs[i].data_ptr<float>();
        if(max_shape0 < shape_ptr[i * 2]) {
            max_shape0 = shape_ptr[i * 2];
        }
        if(max_shape1 < shape_ptr[i * 2 + 1]) {
            max_shape1 = shape_ptr[i * 2 + 1];
        }
    }
    dim3 grid(n, 1, 1);
    dim3 block(GetBlockSize(max_shape1), GetBlockSize(max_shape0), 1);
    Unpad2D_kernel<<<grid, block>>>(inputs_ptr, shape_ptr, outputs_ptr, max_shape0, max_shape1);
}

void Pad3DForward(const std::vector<torch::Tensor>& inputs, torch::Tensor& shape, torch::Tensor* new_x,
            torch::Tensor& mask, const int max_shape0, const int max_shape1, const int max_shape2, const int& value) {
    const int n = inputs.size();
    float** inputs_ptr = new float*[n];
    const int* shape_ptr = shape.data_ptr<int>();
    float* new_x_ptr = new_x.data_ptr<float>();
    int* mask_ptr = mask.data_ptr<int>();
    for(int i = 0; i < n; i++) {
        inputs_ptr[i] = inputs[i].data_ptr<float>();
    }
    dim3 grid(n, 1, 1);
    dim3 block(GetBlockSize(max_shape2), GetBlockSize(max_shape1), GetBlockSize(max_shape0));
    Pad3D_kernel<<<grid, block>>>(inputs_ptr, shape_ptr, new_x_ptr, mask_ptr, max_shape0, max_shape1, max_shape2, value);
}

void GroupPad3DForward(const std::vector<torch::Tensor>& inputs, const torch::Tensor& shape,
                 std::vector<torch::Tensor>& new_x, std::vector<torch::Tensor>& mask, 
                 const torch::Tensor& max_shape, const torch::Tensor& group_id, 
                 const torch::Tensor* group_idx, const int& value) {
    const int n = inputs.size();
    const int group_num = new_x.size();
    float** new_x_ptr = new float*[group_num];
    float** mask_ptr = new float*[group_num];
    int* max_shape_ptr = max_shape.data_ptr<int>();
    int* shape_ptr = shape.data_ptr<int>();
    int* group_id_ptr = group_id.data_ptr<int>();
    int* group_idx_ptr = group_idx.data_ptr<int>();
    int max_shape_global0 = 0;
    int max_shape_global1 = 0;
    int max_shape_global2 = 0;
    for(int i = 0; i < group_num; i++) {
        new_x_ptr[i] = new_x[i].data_ptr<float>();
        mask_ptr[i] = mask[i].data_ptr<int>();
        if(max_shape_global0 < max_shape_ptr[i * 3]) {
            max_shape_global0 = max_shape_ptr[i * 3];
        }
        if(max_shape_global1 < max_shape_ptr[i * 3 + 1]) {
            max_shape_global1 = max_shape_ptr[i * 3 + 1];
        }
        if(max_shape_global2 < max_shape_ptr[i * 3 + 2]) {
            max_shape_global2 = max_shape_ptr[i * 3 + 2];
        }
    }
    dim3 grid(n, 1, 1);
    dim3 block(GetBlockSize(max_shape_global2), GetBlockSize(max_shape_global1), GetBlockSize(max_shape_global0));
    GroupPad3D_kernel<<<grid, block>>>(inputs, shape_ptr, new_x_ptr, mask_ptr, max_shape_ptr, 
                                        group_id_ptr, group_idx_ptr, value);
}

void Unpad3DForward(torch::Tensor inputs, torch::Tensor shape, std::vector<torch::Tensor>& outputs) {
    int max_shape0 = 0;
    int max_shape1 = 0;
    int max_shape2 = 0;
    const int n = inputs.size();
    float* inputs_ptr = inputs.data_ptr<float>();
    float** outputs_ptr = new float*[n]
    const int* shape_ptr = (int*)shape.data_ptr();
    for(int i = 0; i < n; i++) {
        outputs_ptr[i] = outputs[i].data_ptr<float>();
        if(max_shape0 < shape_ptr[i * 3]) {
            max_shape0 = shape_ptr[i * 3];
        }
        if(max_shape1 < shape_ptr[i * 3 + 1]) {
            max_shape1 = shape_ptr[i * 3 + 1];
        }
        if(max_shape2 < shape_ptr[i * 3 + 2]) {
            max_shape2 = shape_ptr[i * 3 + 2];
        }
    }
    dim3 grid(n, 1, 1);
    dim3 block(GetBlockSize(max_shape2), GetBlockSize(max_shape1), GetBlockSize(max_shape0));
    Unpad3D_kernel<<<grid, block>>>(inputs_ptr, shape_ptr, outputs_ptr, max_shape0, max_shape1, max_shape2);
}




}  // namespace cuda
}  // namespace rll
}  // namespace hpc
