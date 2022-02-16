#include "hpc/rll/cuda/rl_utils/entry.h"
#include "hpc/rll/cuda/rl_utils/padding_kernel.h"
#include <algorithm>

namespace hpc {
namespace rll {
namespace cuda {
std::vector<std::vector<int>> sample_split_group(const std::vector<torch::Tensor>& x, int group) {
    int N = x.size();
    int dim = x[0].sizes().size();
    std::vector<std::vector<int>> result;
    std::vector<int> sampled_idx;
    int last_rand = -1;
    int now_rand = -1;
    for(int i = 0; i < group - 1; i++) {
        while(now_rand == last_rand) {
            now_rand = (rand() % (N-2)) + 1;
        }
        sampled_idx.push_back(now_rand);
        last_rand = now_rand;
    }
    sort(sampled_idx.begin(), sampled_idx.end());
    sampled_idx.push_back(N-1);
    std::vector<int> group_idx;
    int last_idx = -1;
    for(auto idx : sampled_idx) {
        std::vector<int> group_shape(dim, -1);
        for(int i = last_idx + 1; i <= idx; i++) {
            for(int j = 0; j < dim; j++) {
                if(x[i].sizes()[j] > group_shape[j])
                    group_shape[j] = x[i].sizes()[j];
            }
        }
        if(!result.empty() && group_shape == result[result.size() - 1]) 
            continue;
        result.push_back(group_shape);
        group_idx.push_back(last_idx + 1);
        last_idx = idx;
    }
    group_idx.push_back(N);
    result.push_back(group_idx);
    return result;
}
std::vector<std::vector<int>> oracle_split_group(const std::vector<torch::Tensor>& x, int group) {
    int N = x.size();
    int M = group;
    int dim = x[0].sizes().size();
    int look_up[dim][N][N];
    int cost[N+1][M+1];
    int pos[N+1][M+1];
    memset(cost, 0, (N+1)*(M+1)*sizeof(int));
    memset(pos, 0, (N+1)*(M+1)*sizeof(int));
    memset(look_up, 0, dim*N*N*sizeof(int));
    for(int d = 0; d < dim; d++) {
        for(int i = 0; i < N; i++)
            look_up[d][i][i] = x[i].sizes()[d];
        for(int i = 0; i < N; i++) {
            for(int j = i + 1; j < N; j++) {
                look_up[d][i][j] = (x[j].sizes()[d] > look_up[d][i][j-1]) ? x[j].sizes()[d] : look_up[d][i][j-1];
            }
        }
    }
    for(int i = 1; i <= N; i++) {
        for(int j = 1; j <= M; j++) {
            bool flg = false;
            int min_cost = 1e8;
            int start_pos = 0;
            for(int k = 0; k < i; k++) {
                if(cost[k][j-1] != 0 || (k == 0 && j == 1)) {
                    flg = true;
                    int max_elems = 1;
                    for(int d = 0; d < dim; d++) max_elems *= look_up[d][k][i-1];
                    int now_cost = cost[k][j-1] + max_elems * (i - k);
                    if(now_cost < min_cost) {
                        min_cost = now_cost;
                        start_pos = k;
                    }
                }
            }
            if(flg) {
                cost[i][j] = min_cost;
                pos[i][j] = start_pos;
            }
        }
    }
    int last_pos = N;
    int last_cnt = M;
    std::vector<int> positions = {N};
    while(last_pos > 0) {
        last_pos = pos[last_pos][last_cnt];
        last_cnt--;
        positions.push_back(last_pos);
    }
    reverse(positions.begin(), positions.end());
    std::vector<std::vector<int>> result;
    int start_id = 0;
    int end_id = 0;
    for(int i = 1; i < positions.size(); i++) {
        end_id = positions[i] - 1;
        std::vector<int> res;
        for(int d = 0; d < dim; d++)
            res.push_back(look_up[d][start_id][end_id]);
        start_id = end_id + 1;
        result.push_back(res);
    }
    result.push_back(positions);
    return result;
}


std::vector<torch::Tensor> Pad1DForward(const std::vector<torch::Tensor>& inputs, const int& value) {
    const int n = inputs.size();
    int max_shape = 0;
    float** inputs_hptr = new float*[n];
    int* shape_hptr = new int[n];
    float** inputs_dptr = nullptr;
    int* shape_dptr = nullptr;
    cudaMalloc((void**)(&inputs_dptr), n * sizeof(float*));
    cudaMalloc((void**)(&shape_dptr), n * sizeof(int));
    for(int i = 0; i < n; i++) {
        inputs_hptr[i] = inputs[i].data_ptr<float>();
        int cur_shape = inputs[i].sizes()[0];
        if(cur_shape > max_shape) max_shape = cur_shape;
        shape_hptr[i] = cur_shape;
    }
    torch::Tensor new_x = at::empty({n,max_shape}, inputs[0].options());
    torch::Tensor mask = at::empty({n,max_shape}, inputs[0].options().dtype(torch::kInt32));
    float* new_x_ptr = new_x.data_ptr<float>();
    int* mask_ptr = mask.data_ptr<int>();
    cudaMemcpy(inputs_dptr, inputs_hptr, n*sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(shape_dptr, shape_hptr, n*sizeof(int), cudaMemcpyHostToDevice);
    delete [] inputs_hptr;
    delete [] shape_hptr;
    dim3 grid(n, 1, 1);
    dim3 block(GetBlockSize(max_shape));
    Pad1D_kernel<<<grid, block>>>(inputs_dptr, shape_dptr, new_x_ptr, mask_ptr, max_shape, value);
    cudaFree(inputs_dptr);
    cudaFree(shape_dptr);
    return {new_x, mask};
}

/*
Par:
group_cnt : count of tensor in each group
max_shape : max shape in each group
group_id : its group index of each tensor
group_idx : each group's start tensor index
*/
std::vector<std::vector<torch::Tensor>> GroupPad1DForward(const std::vector<torch::Tensor>& inputs, 
                 const std::vector<int>& group_cnt, const std::vector<int>& max_shape, 
                 const std::vector<int>& group_id, const std::vector<int>& group_idx, const int& value) {
    const int n = inputs.size();
    int max_shape_global = 0;
    const int group_num = group_cnt.size();
    std::vector<torch::Tensor> new_x;
    std::vector<torch::Tensor> mask;
    
    float** inputs_hptr = new float*[n];
    float** new_x_hptr = new float*[group_num];
    int** mask_hptr = new int*[group_num];
    int* shape_hptr = new int[n];
    int* max_shape_hptr = new int[group_num];
    int* group_id_hptr = new int[n];
    int* group_idx_hptr = new int[group_num];

    float** inputs_dptr = nullptr;
    float** new_x_dptr = nullptr;
    int** mask_dptr = nullptr;   
    int* shape_dptr = nullptr;
    int* max_shape_dptr = nullptr;
    int* group_id_dptr = nullptr;
    int* group_idx_dptr = nullptr;

    for(int i = 0; i < group_num; i++) {
        new_x.push_back(std::move(at::empty({group_cnt[i], max_shape[i]}, inputs[0].options())));
        mask.push_back(std::move(at::empty({group_cnt[i], max_shape[i]}, inputs[0].options().dtype(torch::kInt32))));
    }
    
    cudaMalloc((void**)(&inputs_dptr), n * sizeof(float*));
    cudaMalloc((void**)(&new_x_dptr), group_num * sizeof(float*));
    cudaMalloc((void**)(&mask_dptr), group_num * sizeof(int*));
    cudaMalloc((void**)(&shape_dptr), n * sizeof(int));
    cudaMalloc((void**)(&max_shape_dptr), group_num * sizeof(int));
    cudaMalloc((void**)(&group_id_dptr), n * sizeof(int));
    cudaMalloc((void**)(&group_idx_dptr), group_num * sizeof(int));

    for(int i = 0; i < n; i++) {
        inputs_hptr[i] = inputs[i].data_ptr<float>();
        shape_hptr[i] = inputs[i].sizes()[0];
        group_id_hptr[i] = group_id[i];
    }
    for(int i = 0; i < group_num; i++) {
        if(max_shape[i] > max_shape_global) max_shape_global = max_shape[i];
        new_x_hptr[i] = new_x[i].data_ptr<float>();
        mask_hptr[i] = mask[i].data_ptr<int>();
        max_shape_hptr[i] = max_shape[i];
        group_idx_hptr[i] = group_idx[i];
    }
    
    cudaMemcpy(inputs_dptr, inputs_hptr, n*sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(new_x_dptr, new_x_hptr, group_num*sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(mask_dptr, mask_hptr, group_num*sizeof(int*), cudaMemcpyHostToDevice);
    cudaMemcpy(shape_dptr, shape_hptr, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(max_shape_dptr, max_shape_hptr, group_num*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(group_id_dptr, group_id_hptr, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(group_idx_dptr, group_idx_hptr, group_num*sizeof(int), cudaMemcpyHostToDevice);

    delete [] inputs_hptr;
    delete [] new_x_hptr;
    delete [] mask_hptr;
    delete [] shape_hptr;
    delete [] max_shape_hptr;
    delete [] group_id_hptr;
    delete [] group_idx_hptr;

    dim3 grid(n, 1, 1);
    dim3 block(GetBlockSize(max_shape_global));
    GroupPad1D_kernel<<<grid, block>>>(inputs_dptr, shape_dptr, new_x_dptr, mask_dptr, max_shape_dptr, 
                                        group_id_dptr, group_idx_dptr, value);
    cudaFree(inputs_dptr);
    cudaFree(new_x_dptr);
    cudaFree(mask_dptr);
    cudaFree(shape_dptr);
    cudaFree(max_shape_dptr);
    cudaFree(group_id_dptr);
    cudaFree(group_idx_dptr);

    return {new_x, mask};
}

std::vector<torch::Tensor> Unpad1DForward(const torch::Tensor& inputs, const std::vector<int>& shape) {
    const int n = shape.size();
    int max_shape = 0;
    std::vector<torch::Tensor> outputs;
    for(int i = 0; i < n; i++) {
        outputs.push_back(std::move(at::empty({shape[i]}, inputs.options())));
    }
    float** outputs_hptr = new float*[n];
    int* shape_hptr = new int[n];

    float** outputs_dptr = nullptr;
    int* shape_dptr = nullptr;
    cudaMalloc((void**)(&outputs_dptr), n * sizeof(float*));
    cudaMalloc((void**)(&shape_dptr), n * sizeof(int));
    for(int i = 0; i < n; i++) {
        outputs_hptr[i] = outputs[i].data_ptr<float>();
        int cur_shape = shape[i];
        if(cur_shape > max_shape) max_shape = cur_shape;
        shape_hptr[i] = cur_shape;
    }
    cudaMemcpy(outputs_dptr, outputs_hptr, n*sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(shape_dptr, shape_hptr, n*sizeof(int), cudaMemcpyHostToDevice);
    delete [] outputs_hptr;
    delete [] shape_hptr;
    dim3 grid(n, 1, 1);
    dim3 block(GetBlockSize(max_shape));
    Unpad1D_kernel<<<grid, block>>>(inputs.data_ptr<float>(), shape_dptr, outputs_dptr, max_shape);
    cudaFree(outputs_dptr);
    cudaFree(shape_dptr);
    cudaError_t error = cudaGetLastError();
    return outputs;
}

std::vector<torch::Tensor> Pad2DForward(const std::vector<torch::Tensor>& inputs, const int& value) {
    const int n = inputs.size();
    int max_shape0 = 0;
    int max_shape1 = 0;
    float** inputs_hptr = new float*[n];
    int* shape_hptr = new int[2 * n];
    float** inputs_dptr = nullptr;
    int* shape_dptr = nullptr;
    cudaMalloc((void**)(&inputs_dptr), n * sizeof(float*));
    cudaMalloc((void**)(&shape_dptr), 2 * n * sizeof(int));
    for(int i = 0; i < n; i++) {
        inputs_hptr[i] = inputs[i].data_ptr<float>();
        int cur_shape0 = inputs[i].sizes()[0];
        int cur_shape1 = inputs[i].sizes()[1];
        if(cur_shape0 > max_shape0) max_shape0 = cur_shape0;
        if(cur_shape1 > max_shape1) max_shape1 = cur_shape1;
        shape_hptr[i * 2] = cur_shape0;
        shape_hptr[i * 2 + 1] = cur_shape1;
    } 
    torch::Tensor new_x = at::empty({n, max_shape0, max_shape1}, inputs[0].options());
    torch::Tensor mask = at::empty({n, max_shape0, max_shape1}, inputs[0].options().dtype(torch::kInt32));
    float* new_x_ptr = new_x.data_ptr<float>();
    int* mask_ptr = mask.data_ptr<int>();
    cudaMemcpy(inputs_dptr, inputs_hptr, n*sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(shape_dptr, shape_hptr, 2*n*sizeof(int), cudaMemcpyHostToDevice);
    delete [] inputs_hptr;
    delete [] shape_hptr;
    dim3 grid(n, 1, 1);
    dim3 block(32, 32, 1);
    Pad2D_kernel<<<grid, block>>>(inputs_dptr, shape_dptr, new_x_ptr, mask_ptr, max_shape0, max_shape1, value);
    cudaFree(inputs_dptr);
    cudaFree(shape_dptr);
    return {new_x, mask};
}

std::vector<std::vector<torch::Tensor>> GroupPad2DForward(const std::vector<torch::Tensor>& inputs, 
                 const std::vector<int>& group_cnt, const std::vector<int>& max_shape, 
                 const std::vector<int>& group_id, const std::vector<int>& group_idx, const int& value) {
    const int n = inputs.size();
    const int group_num = group_cnt.size();
    int max_shape_global0 = 0;
    int max_shape_global1 = 0;
    std::vector<torch::Tensor> new_x;
    std::vector<torch::Tensor> mask;
    float** inputs_hptr = new float*[n];
    float** new_x_hptr = new float*[group_num];
    int** mask_hptr = new int*[group_num];
    int* shape_hptr = new int[2*n];
    int* max_shape_hptr = new int[2*group_num];
    int* group_id_hptr = new int[n];
    int* group_idx_hptr = new int[group_num];

    float** inputs_dptr = nullptr;
    float** new_x_dptr = nullptr;
    int** mask_dptr = nullptr;   
    int* shape_dptr = nullptr;
    int* max_shape_dptr = nullptr;
    int* group_id_dptr = nullptr;
    int* group_idx_dptr = nullptr;

    for(int i = 0; i < group_num; i++) {
        new_x.push_back(std::move(at::empty({group_cnt[i], max_shape[i*2], max_shape[i*2+1]}, inputs[0].options())));
        mask.push_back(std::move(at::empty({group_cnt[i], max_shape[i*2], max_shape[i*2+1]}, inputs[0].options().dtype(torch::kInt32))));
    }
    
    cudaMalloc((void**)(&inputs_dptr), n * sizeof(float*));
    cudaMalloc((void**)(&new_x_dptr), group_num * sizeof(float*));
    cudaMalloc((void**)(&mask_dptr), group_num * sizeof(int*));
    cudaMalloc((void**)(&shape_dptr), 2 * n * sizeof(int));
    cudaMalloc((void**)(&max_shape_dptr), 2 * group_num * sizeof(int));
    cudaMalloc((void**)(&group_id_dptr), n * sizeof(int));
    cudaMalloc((void**)(&group_idx_dptr), group_num * sizeof(int));

    for(int i = 0; i < n; i++) {
        inputs_hptr[i] = inputs[i].data_ptr<float>();
        shape_hptr[i * 2] = inputs[i].sizes()[0];
        shape_hptr[i * 2 + 1] = inputs[i].sizes()[1];
        group_id_hptr[i] = group_id[i];
    }
    for(int i = 0; i < group_num; i++) {
        if(max_shape[i * 2] > max_shape_global0) max_shape_global0 = max_shape[i * 2];
        if(max_shape[i * 2 + 1] > max_shape_global1) max_shape_global1 = max_shape[i * 2 + 1];
        new_x_hptr[i] = new_x[i].data_ptr<float>();
        mask_hptr[i] = mask[i].data_ptr<int>();
        max_shape_hptr[i * 2] = max_shape[i * 2];
        max_shape_hptr[i * 2 + 1] = max_shape[i * 2 + 1];
        group_idx_hptr[i] = group_idx[i];
    }
    
    cudaMemcpy(inputs_dptr, inputs_hptr, n*sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(new_x_dptr, new_x_hptr, group_num*sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(mask_dptr, mask_hptr, group_num*sizeof(int*), cudaMemcpyHostToDevice);
    cudaMemcpy(shape_dptr, shape_hptr, 2*n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(max_shape_dptr, max_shape_hptr, 2*group_num*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(group_id_dptr, group_id_hptr, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(group_idx_dptr, group_idx_hptr, group_num*sizeof(int), cudaMemcpyHostToDevice);


    delete [] inputs_hptr;
    delete [] new_x_hptr;
    delete [] mask_hptr;
    delete [] shape_hptr;
    delete [] max_shape_hptr;
    delete [] group_id_hptr;
    delete [] group_idx_hptr;
    dim3 grid(n, 1, 1);
    dim3 block(32, 32);
    GroupPad2D_kernel<<<grid, block>>>(inputs_dptr, shape_dptr, new_x_dptr, mask_dptr, max_shape_dptr, 
                                        group_id_dptr, group_idx_dptr, value);
    cudaFree(inputs_dptr);
    cudaFree(new_x_dptr);
    cudaFree(mask_dptr);
    cudaFree(shape_dptr);
    cudaFree(max_shape_dptr);
    cudaFree(group_id_dptr);
    cudaFree(group_idx_dptr);

    return {new_x, mask};
}

std::vector<torch::Tensor> Unpad2DForward(const torch::Tensor& inputs, const std::vector<int>& shape) {
    const int n = shape.size() / 2;
    int max_shape0 = 0;
    int max_shape1 = 1;
    std::vector<torch::Tensor> outputs;
    for(int i = 0; i < n; i++) {
        outputs.push_back(std::move(at::empty({shape[i * 2], shape[i * 2 + 1]}, inputs.options())));
    }
    float** outputs_hptr = new float*[n];
    int* shape_hptr = new int[2 * n];
    float** outputs_dptr = nullptr;
    int* shape_dptr = nullptr;
    cudaMalloc((void**)(&outputs_dptr), n * sizeof(float*));
    cudaMalloc((void**)(&shape_dptr), 2 * n * sizeof(int));
    for(int i = 0; i < n; i++) {
        outputs_hptr[i] = outputs[i].data_ptr<float>();
        int cur_shape0 = shape[i * 2];
        int cur_shape1 = shape[i * 2 + 1];
        if(cur_shape0 > max_shape0) max_shape0 = cur_shape0;
        if(cur_shape1 > max_shape1) max_shape1 = cur_shape1;
        shape_hptr[i * 2] = cur_shape0;
        shape_hptr[i * 2 + 1] = cur_shape1;
    }
    cudaMemcpy(outputs_dptr, outputs_hptr, n*sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(shape_dptr, shape_hptr, 2*n*sizeof(int), cudaMemcpyHostToDevice);
    delete [] outputs_hptr;
    delete [] shape_hptr;
    dim3 grid(n, 1, 1);
    dim3 block(32, 32);
    Unpad2D_kernel<<<grid, block>>>(inputs.data_ptr<float>(), shape_dptr, outputs_dptr, max_shape0, max_shape1);
    cudaFree(outputs_dptr);
    cudaFree(shape_dptr);
    return outputs;
}

std::vector<torch::Tensor> Pad3DForward(const std::vector<torch::Tensor>& inputs, const int& value) {
    const int n = inputs.size();
    int max_shape0 = 0;
    int max_shape1 = 0;
    int max_shape2 = 0;
    float** inputs_hptr = new float*[n];
    int* shape_hptr = new int[3 * n];
    float** inputs_dptr = nullptr;
    int* shape_dptr = nullptr;
    cudaMalloc((void**)(&inputs_dptr), n * sizeof(float*));
    cudaMalloc((void**)(&shape_dptr), 3 * n * sizeof(int));
    for(int i = 0; i < n; i++) {
        inputs_hptr[i] = inputs[i].data_ptr<float>();
        int cur_shape0 = inputs[i].sizes()[0];
        int cur_shape1 = inputs[i].sizes()[1];
        int cur_shape2 = inputs[i].sizes()[2];
        if(cur_shape0 > max_shape0) max_shape0 = cur_shape0;
        if(cur_shape1 > max_shape1) max_shape1 = cur_shape1;
        if(cur_shape2 > max_shape2) max_shape2 = cur_shape2;
        shape_hptr[i * 3] = cur_shape0;
        shape_hptr[i * 3 + 1] = cur_shape1;
        shape_hptr[i * 3 + 2] = cur_shape2;
    } 
    torch::Tensor new_x = at::empty({n, max_shape0, max_shape1, max_shape2}, inputs[0].options());
    torch::Tensor mask = at::empty({n, max_shape0, max_shape1, max_shape2}, inputs[0].options().dtype(torch::kInt32));
    float* new_x_ptr = new_x.data_ptr<float>();
    int* mask_ptr = mask.data_ptr<int>();
    cudaMemcpy(inputs_dptr, inputs_hptr, n*sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(shape_dptr, shape_hptr, 3*n*sizeof(int), cudaMemcpyHostToDevice);
    delete [] inputs_hptr;
    delete [] shape_hptr;
    dim3 grid(n, 1, 1);
    dim3 block(16, 8, 8);
    Pad3D_kernel<<<grid, block>>>(inputs_dptr, shape_dptr, new_x_ptr, mask_ptr, max_shape0, max_shape1, max_shape2, value);
    cudaFree(inputs_dptr);
    cudaFree(shape_dptr);
    return {new_x, mask};
}

std::vector<std::vector<torch::Tensor>> GroupPad3DForward(const std::vector<torch::Tensor>& inputs, 
                 const std::vector<int>& group_cnt, const std::vector<int>& max_shape, 
                 const std::vector<int>& group_id, const std::vector<int>& group_idx, const int& value) {
    const int n = inputs.size();
    const int group_num = group_cnt.size();
    int max_shape_global0 = 0;
    int max_shape_global1 = 0;
    int max_shape_global2 = 0;
    std::vector<torch::Tensor> new_x;
    std::vector<torch::Tensor> mask;
    float** inputs_hptr = new float*[n];
    float** new_x_hptr = new float*[group_num];
    int** mask_hptr = new int*[group_num];
    int* shape_hptr = new int[3*n];
    int* max_shape_hptr = new int[3*group_num];
    int* group_id_hptr = new int[n];
    int* group_idx_hptr = new int[group_num];

    float** inputs_dptr = nullptr;
    float** new_x_dptr = nullptr;
    int** mask_dptr = nullptr;   
    int* shape_dptr = nullptr;
    int* max_shape_dptr = nullptr;
    int* group_id_dptr = nullptr;
    int* group_idx_dptr = nullptr;

    for(int i = 0; i < group_num; i++) {
        new_x.push_back(std::move(at::empty({group_cnt[i], max_shape[i*3], max_shape[i*3+1], max_shape[i*3+2]}, inputs[0].options())));
        mask.push_back(std::move(at::empty({group_cnt[i], max_shape[i*3], max_shape[i*3+1], max_shape[i*3+2]}, inputs[0].options().dtype(torch::kInt32))));
    }
    
    cudaMalloc((void**)(&inputs_dptr), n * sizeof(float*));
    cudaMalloc((void**)(&new_x_dptr), group_num * sizeof(float*));
    cudaMalloc((void**)(&mask_dptr), group_num * sizeof(int*));
    cudaMalloc((void**)(&shape_dptr), 3 * n * sizeof(int));
    cudaMalloc((void**)(&max_shape_dptr), 3 * group_num * sizeof(int));
    cudaMalloc((void**)(&group_id_dptr), n * sizeof(int));
    cudaMalloc((void**)(&group_idx_dptr), group_num * sizeof(int));

    for(int i = 0; i < n; i++) {
        inputs_hptr[i] = inputs[i].data_ptr<float>();
        shape_hptr[i * 3] = inputs[i].sizes()[0];
        shape_hptr[i * 3 + 1] = inputs[i].sizes()[1];
        shape_hptr[i * 3 + 2] = inputs[i].sizes()[2];
        group_id_hptr[i] = group_id[i];
    }
    for(int i = 0; i < group_num; i++) {
        if(max_shape[i * 3] > max_shape_global0) max_shape_global0 = max_shape[i * 3];
        if(max_shape[i * 3 + 1] > max_shape_global1) max_shape_global1 = max_shape[i * 3 + 1];
        if(max_shape[i * 3 + 2] > max_shape_global2) max_shape_global2 = max_shape[i * 3 + 2];
        new_x_hptr[i] = new_x[i].data_ptr<float>();
        mask_hptr[i] = mask[i].data_ptr<int>();
        max_shape_hptr[i * 3] = max_shape[i * 3];
        max_shape_hptr[i * 3 + 1] = max_shape[i * 3 + 1];
        max_shape_hptr[i * 3 + 2] = max_shape[i * 3 + 2];
        group_idx_hptr[i] = group_idx[i];
    }
    
    cudaMemcpy(inputs_dptr, inputs_hptr, n*sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(new_x_dptr, new_x_hptr, group_num*sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(mask_dptr, mask_hptr, group_num*sizeof(int*), cudaMemcpyHostToDevice);
    cudaMemcpy(shape_dptr, shape_hptr, 3*n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(max_shape_dptr, max_shape_hptr, 3*group_num*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(group_id_dptr, group_id_hptr, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(group_idx_dptr, group_idx_hptr, group_num*sizeof(int), cudaMemcpyHostToDevice);

    delete [] inputs_hptr;
    delete [] new_x_hptr;
    delete [] mask_hptr;
    delete [] shape_hptr;
    delete [] max_shape_hptr;
    delete [] group_id_hptr;
    delete [] group_idx_hptr;
    dim3 grid(n, 1, 1);
    dim3 block(16, 8, 8);
    GroupPad3D_kernel<<<grid, block>>>(inputs_dptr, shape_dptr, new_x_dptr, mask_dptr, max_shape_dptr, 
                                        group_id_dptr, group_idx_dptr, value);
    cudaFree(inputs_dptr);
    cudaFree(new_x_dptr);
    cudaFree(mask_dptr);
    cudaFree(shape_dptr);
    cudaFree(max_shape_dptr);
    cudaFree(group_id_dptr);
    cudaFree(group_idx_dptr);
    return {new_x, mask};
}

std::vector<torch::Tensor> Unpad3DForward(const torch::Tensor& inputs, const std::vector<int>& shape) {
    const int n = shape.size() / 3;
    int max_shape0 = 0;
    int max_shape1 = 1;
    int max_shape2 = 2;
    std::vector<torch::Tensor> outputs;
    for(int i = 0; i < n; i++) {
        outputs.push_back(std::move(at::empty({shape[i * 3], shape[i * 3 + 1], shape[i * 3 + 2]}, inputs.options())));
    }
    float** outputs_hptr = new float*[n];
    int* shape_hptr = new int[3 * n];
    float** outputs_dptr = nullptr;
    int* shape_dptr = nullptr;
    cudaMalloc((void**)(&outputs_dptr), n * sizeof(float*));
    cudaMalloc((void**)(&shape_dptr), 3 * n * sizeof(int));
    for(int i = 0; i < n; i++) {
        outputs_hptr[i] = outputs[i].data_ptr<float>();
        int cur_shape0 = shape[i * 3];
        int cur_shape1 = shape[i * 3 + 1];
        int cur_shape2 = shape[i * 3 + 2];
        if(cur_shape0 > max_shape0) max_shape0 = cur_shape0;
        if(cur_shape1 > max_shape1) max_shape1 = cur_shape1;
        if(cur_shape2 > max_shape2) max_shape2 = cur_shape2;
        shape_hptr[i * 3] = cur_shape0;
        shape_hptr[i * 3 + 1] = cur_shape1;
        shape_hptr[i * 3 + 2] = cur_shape2;
    }
    cudaMemcpy(outputs_dptr, outputs_hptr, n*sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(shape_dptr, shape_hptr, 3*n*sizeof(int), cudaMemcpyHostToDevice);
    delete [] outputs_hptr;
    delete [] shape_hptr;
    dim3 grid(n, 1, 1);
    dim3 block(16, 8, 8);
    Unpad3D_kernel<<<grid, block>>>(inputs.data_ptr<float>(), shape_dptr, outputs_dptr, max_shape0, max_shape1, max_shape2);
    cudaFree(outputs_dptr);
    cudaFree(shape_dptr);
    return outputs;
}




}  // namespace cuda
}  // namespace rll
}  // namespace hpc
