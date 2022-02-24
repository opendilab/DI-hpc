#include <unistd.h>
#include "hpc/rll/cuda/torch_utils/network/entry.h"
#include "hpc/rll/cuda/torch_utils/network/lstm_kernel.h"

namespace hpc {
namespace rll {
namespace cuda {

// random seeding, keep up with caffe
int64_t cluster_seedgen(void) {
    int64_t s, seed, pid;
    FILE* f = fopen("/dev/urandom", "rb");
    if (f && fread(&seed, 1, sizeof(seed), f) == sizeof(seed)) {
        fclose(f);
        return seed;
    }

    LOG(INFO) << "System entropy source not available, "
        "using fallback algorithm to generate seed instead.";
    if (f)
        fclose(f);

    pid = getpid();
    s = time(NULL);
    seed = std::abs(((s * 181) * ((pid - 83) * 359)) % 104729);
    return seed;
}

void LstmForward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs,
    float dropout_threshold) {

    unsigned int index = 0;
    const torch::Tensor& x0 = inputs[index++];
    const torch::Tensor& h0 = inputs[index++];
    const torch::Tensor& c0 = inputs[index++];
    const torch::Tensor& wx = inputs[index++];
    const torch::Tensor& wh = inputs[index++];
    const torch::Tensor& bias = inputs[index++];
    const torch::Tensor& ln_gamma = inputs[index++];
    const torch::Tensor& ln_beta = inputs[index++];
    index = 0;
    torch::Tensor& xbuf = outputs[index++];
    torch::Tensor& hbuf = outputs[index++];
    torch::Tensor& hn = outputs[index++];
    torch::Tensor& cn = outputs[index++];
    torch::Tensor& ifog = outputs[index++];
    torch::Tensor& ym = outputs[index++];
    torch::Tensor& ln_in = outputs[index++];
    torch::Tensor& ln_mean = outputs[index++];
    torch::Tensor& ln_rstd = outputs[index++];
    torch::Tensor& dropout_mask = outputs[index++];

    const unsigned int seq_len = x0.size(0);
    const unsigned int batch_size = x0.size(1);
    const unsigned int input_size = x0.size(2);
	const unsigned int num_layers = h0.size(0);
    const unsigned int hidden_size = h0.size(2);

    const float* inputptr = (float*)(x0.data_ptr());
    const float* h0ptr = (float*)(h0.data_ptr());
    const float* c0ptr = (float*)(c0.data_ptr());
    const float* wxptr = (float*)(wx.data_ptr());
    const float* whptr = (float*)(wh.data_ptr());
    const float* biasptr = (float*)(bias.data_ptr());
    const float* ln_gammaptr = (float*)(ln_gamma.data_ptr());
    const float* ln_betaptr = (float*)(ln_beta.data_ptr());
    float* xbufptr = (float*)(xbuf.data_ptr());
    float* hbufptr = (float*)(hbuf.data_ptr());
    float* hnptr = (float*)(hn.data_ptr());
    float* cnptr = (float*)(cn.data_ptr());
    float* ifogptr = (float*)(ifog.data_ptr());
    float* outputptr = (float*)(ym.data_ptr());
    float* ln_in_ptr = (float*)(ln_in.data_ptr());
    float* ln_mean_ptr = (float*)(ln_mean.data_ptr());
    float* ln_rstd_ptr = (float*)(ln_rstd.data_ptr());
    unsigned int* maskptr = reinterpret_cast<unsigned int*>((int32_t*)(dropout_mask.data_ptr()));
    float onedata = 1;
    float zerodata = 0;

    // create handles
	cublasHandle_t cublas_handle;
    checkCublasErr(cublasCreate(&cublas_handle));

    curandGenerator_t gen;
    if (dropout_threshold > 0) {
        checkCurandErr(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
        checkCurandErr(curandSetPseudoRandomGeneratorSeed(gen, cluster_seedgen()));
    }

    // TODO pay attention to wx shape change
    unsigned int wxidx[num_layers];
    wxidx[0] = input_size;
    for (int l = 0; l < num_layers - 1; l++) {
        wxidx[l + 1] = hidden_size;
    }
    unsigned int wxoffset[num_layers];
    wxoffset[0] = 0;
    for (int l = 0; l < num_layers - 1; l++) {
        wxoffset[l + 1] = wxoffset[l] + wxidx[l] * wxidx[l + 1] * 4;
    }

    for (int l = 0; l < num_layers; l++) {
        const float* xdata = (l == 0 ? inputptr : (outputptr + (l - 1) * seq_len * batch_size * hidden_size));
        const float* wxdata = wxptr + wxoffset[l];
        const float* whdata = whptr + l * hidden_size * (hidden_size * 4);
        const float* biasdata = biasptr + l * (hidden_size * 4);

        const float* ln_gamma_x = ln_gammaptr + l * hidden_size * 4 * 2;
        const float* ln_gamma_h = ln_gammaptr + l * hidden_size * 4 * 2 + hidden_size * 4;
        const float* ln_beta_x = ln_betaptr + l * hidden_size * 4 * 2;
        const float* ln_beta_h = ln_betaptr + l * hidden_size * 4 * 2 + hidden_size * 4;
        float* ln_x = ln_in_ptr + l * seq_len * batch_size * hidden_size * 4 * 2;
        float* ln_h = ln_in_ptr + l * seq_len * batch_size * hidden_size * 4 * 2 + seq_len * batch_size * hidden_size * 4;
        float* ln_mean_x = ln_mean_ptr + l * seq_len * batch_size * 2;
        float* ln_mean_h = ln_mean_ptr + l * seq_len * batch_size * 2 + seq_len * batch_size;
        float* ln_rstd_x = ln_rstd_ptr + l * seq_len * batch_size * 2;
        float* ln_rstd_h = ln_rstd_ptr + l * seq_len * batch_size * 2 + seq_len * batch_size;

        checkCublasErr(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    hidden_size * 4, seq_len * batch_size, wxidx[l],
                    &onedata, wxdata, hidden_size * 4, xdata, wxidx[l], &zerodata, ln_x, hidden_size * 4));

        // layernorm
        unsigned int block_size = DEFAULT_WARP_NUM * WARP_SIZE;
        unsigned int grid_size = seq_len * batch_size;
        layernorm<<<grid_size, block_size>>>(
                hidden_size * 4, ln_x, ln_gamma_x, ln_beta_x, ln_mean_x, ln_rstd_x, xbufptr);

        for (int s = 0; s < seq_len; s++) {
            const float* xbufdata = xbufptr + s * batch_size * (hidden_size * 4);
            const float* prehdata = (s == 0 ? (h0ptr + l * batch_size * hidden_size)
                    : (hnptr + (s - 1) * num_layers * batch_size * hidden_size + l * batch_size * hidden_size));
            const float* precdata = (s == 0 ? (c0ptr + l * batch_size * hidden_size)
                    : (cnptr + (s - 1) * num_layers * batch_size * hidden_size + l * batch_size * hidden_size));
            float* hdata = hnptr + s * num_layers * batch_size * hidden_size + l * batch_size * hidden_size;
            float* cdata = cnptr + s * num_layers * batch_size * hidden_size + l * batch_size * hidden_size;
            float* ifogdata = ifogptr + l * seq_len * batch_size * hidden_size * 4 + s * batch_size * hidden_size * 4;
            float* outputdata = outputptr + l * seq_len * batch_size * hidden_size + s * batch_size * hidden_size;
            float* ln_h_s = ln_h + s * batch_size * hidden_size * 4;
            float* ln_mean_h_s = ln_mean_h + s * batch_size;
            float* ln_rstd_h_s = ln_rstd_h + s * batch_size;

            checkCublasErr(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        hidden_size * 4, batch_size, hidden_size,
                        &onedata, whdata, hidden_size * 4, prehdata, hidden_size, &zerodata, ln_h_s, hidden_size * 4));

            // layernorm
            {
                unsigned int block_size = DEFAULT_WARP_NUM * WARP_SIZE;
                unsigned int grid_size = batch_size;
                layernorm<<<grid_size, block_size>>>(
                        hidden_size * 4, ln_h_s, ln_gamma_h, ln_beta_h, ln_mean_h_s, ln_rstd_h_s, hbufptr);
            }
            {
                dim3 block_size = {DEFAULT_WARP_NUM * WARP_SIZE, 1, 1};
                dim3 grid_size = {(hidden_size + block_size.x - 1) / block_size.x, batch_size, 1};
                activation<<<grid_size, block_size>>>(
                        batch_size, hidden_size, xbufdata , hbufptr, biasdata,
                        prehdata, precdata, hdata, cdata, ifogdata, outputdata);
            }
        }

        // dropout
        if (dropout_threshold > 0 && l != num_layers - 1) {
            float* dropoutdata = outputptr + l * seq_len * batch_size * hidden_size;
            unsigned int maskstride = seq_len * batch_size * hidden_size;
            unsigned int* maskdata = maskptr + l * maskstride;
            checkCurandErr(curandGenerate(gen, maskdata, maskstride));

            float dropout_scale = 1. / (1. - dropout_threshold);
            unsigned int uint_threshold = static_cast<unsigned int>(UINT_MAX * dropout_threshold);
            unsigned int block_size = DEFAULT_WARP_NUM * WARP_SIZE;
            unsigned int grid_size = (maskstride + block_size - 1) / block_size;
            dropout<<<grid_size, block_size>>>(
                    maskstride, uint_threshold, dropout_scale, maskdata, dropoutdata);
        }
    }

    // destroy handles
    checkCublasErr(cublasDestroy(cublas_handle));
    if (dropout_threshold > 0) {
        checkCurandErr(curandDestroyGenerator(gen));
    }
}

void LstmBackward(
    const std::vector<torch::Tensor>& inputs,
    std::vector<torch::Tensor>& outputs,
    float dropout_threshold) {

    unsigned int index = 0;
    const torch::Tensor& x0 = inputs[index++];
    const torch::Tensor& h0 = inputs[index++];
    const torch::Tensor& c0 = inputs[index++];
    const torch::Tensor& wx = inputs[index++];
    const torch::Tensor& wh = inputs[index++];
    const torch::Tensor& hn = inputs[index++];
    const torch::Tensor& cn = inputs[index++];
    const torch::Tensor& ifogbuf = inputs[index++];
    const torch::Tensor& ym = inputs[index++];
    const torch::Tensor& ln_in = inputs[index++];
    const torch::Tensor& ln_mean = inputs[index++];
    const torch::Tensor& ln_rstd = inputs[index++];
    const torch::Tensor& ln_gamma = inputs[index++];
    const torch::Tensor& dropout_mask = inputs[index++];
    index = 0;
    torch::Tensor& dgatebuf = outputs[index++];
    torch::Tensor& xbuf = outputs[index++];
    torch::Tensor& hbuf = outputs[index++];
    torch::Tensor& dx = outputs[index++];
    torch::Tensor& dwx = outputs[index++];
    torch::Tensor& dwh = outputs[index++];
    torch::Tensor& dbias = outputs[index++];
    torch::Tensor& d_ln_gamma = outputs[index++];
    torch::Tensor& d_ln_beta = outputs[index++];
    torch::Tensor& dy = outputs[index++];
    torch::Tensor& dh = outputs[index++];
    torch::Tensor& dc = outputs[index++];

    const unsigned int seq_len = x0.size(0);
    const unsigned int batch_size = x0.size(1);
    const unsigned int input_size = x0.size(2);
	const unsigned int num_layers = h0.size(0);
    const unsigned int hidden_size = h0.size(2);

    const float* x0ptr = (float*)(x0.data_ptr());
    const float* ifogptr = (float*)(ifogbuf.data_ptr());
    const float* ymptr = (float*)(ym.data_ptr());
    const float* h0ptr = (float*)(h0.data_ptr());
    const float* c0ptr = (float*)(c0.data_ptr());
    const float* hnptr = (float*)(hn.data_ptr());
    const float* cnptr = (float*)(cn.data_ptr());
    const float* wxptr = (float*)(wx.data_ptr());
    const float* whptr = (float*)(wh.data_ptr());
    const float* ln_in_ptr = (float*)(ln_in.data_ptr());
    const float* ln_mean_ptr = (float*)(ln_mean.data_ptr());
    const float* ln_rstd_ptr = (float*)(ln_rstd.data_ptr());
    const float* ln_gammaptr = (float*)(ln_gamma.data_ptr());
    const unsigned int* maskptr = reinterpret_cast<unsigned int*>((int32_t*)(dropout_mask.data_ptr()));
    float* dgatebufptr = (float*)(dgatebuf.data_ptr());
    float* xbufptr = (float*)(xbuf.data_ptr());
    float* hbufptr = (float*)(hbuf.data_ptr());
    float* dyptr = (float*)(dy.data_ptr());
    float* dxptr = (float*)(dx.data_ptr());
    float* dhptr = (float*)(dh.data_ptr());
    float* dcptr = (float*)(dc.data_ptr());
    float* dwxptr = (float*)(dwx.data_ptr());
    float* dwhptr = (float*)(dwh.data_ptr());
    float* dbiasptr = (float*)(dbias.data_ptr());
    float* ln_dgammaptr = (float*)(d_ln_gamma.data_ptr());
    float* ln_dbetaptr = (float*)(d_ln_beta.data_ptr());
    float onedata = 1;
    float zerodata = 0;

	cublasHandle_t cublas_handle;
    checkCublasErr(cublasCreate(&cublas_handle));

    // TODO pay attention to wx shape change
    unsigned int wxidx[num_layers + 1];
    wxidx[0] = input_size;
    for (int l = 0; l < num_layers; l++) {
        wxidx[l + 1] = hidden_size;
    }
    unsigned int wxoffset[num_layers + 1];
    wxoffset[0] = 0;
    unsigned int totalwx = 0;
    for (int l = 0; l < num_layers; l++) {
        totalwx += wxidx[l] * wxidx[l + 1] * 4;
        wxoffset[l + 1] = wxoffset[l] + wxidx[l] * wxidx[l + 1] * 4;
    }

    checkCudaErr(cudaMemsetAsync(dxptr, 0, seq_len * batch_size * input_size * sizeof(float)));
    checkCudaErr(cudaMemsetAsync(dwxptr, 0, totalwx * sizeof(float)));
    checkCudaErr(cudaMemsetAsync(dwhptr, 0, num_layers * hidden_size * hidden_size * 4 * sizeof(float)));
    checkCudaErr(cudaMemsetAsync(dbiasptr, 0, num_layers * hidden_size * 4 * sizeof(float)));
    checkCudaErr(cudaMemsetAsync(ln_dgammaptr, 0, num_layers * hidden_size * 4 * 2 * sizeof(float)));
    checkCudaErr(cudaMemsetAsync(ln_dbetaptr, 0, num_layers * hidden_size * 4 * 2 * sizeof(float)));
    for (int l = num_layers - 1; l >= 0; l--) {
        // dropout
        if (dropout_threshold > 0 && l != num_layers - 1) {
            float* dropoutdata = dyptr;
            unsigned int maskstride = seq_len * batch_size * hidden_size;
            const unsigned int* maskdata = maskptr + l * maskstride;

            float dropout_scale = 1. / (1. - dropout_threshold);
            unsigned int uint_threshold = static_cast<unsigned int>(UINT_MAX * dropout_threshold);
            unsigned int block_size = DEFAULT_WARP_NUM * WARP_SIZE;
            unsigned int grid_size = (maskstride + block_size - 1) / block_size;
            dropout_backward<<<grid_size, block_size>>>(
                    maskstride, uint_threshold, dropout_scale, maskdata, dropoutdata);
        }

        // layernorm
        const float* ln_gamma_x = ln_gammaptr + l * hidden_size * 4 * 2;
        const float* ln_gamma_h = ln_gammaptr + l * hidden_size * 4 * 2 + hidden_size * 4;
        float* ln_dgamma_x = ln_dgammaptr + l * hidden_size * 4 * 2;
        float* ln_dgamma_h = ln_dgammaptr + l * hidden_size * 4 * 2 + hidden_size * 4;
        float* ln_dbeta_x = ln_dbetaptr + l * hidden_size * 4 * 2;
        float* ln_dbeta_h = ln_dbetaptr + l * hidden_size * 4 * 2 + hidden_size * 4;

        // lstm
        const float* wxdata = wxptr + wxoffset[l];
        float* dwxdata = dwxptr + wxoffset[l];
        const float* whdata = whptr + l * hidden_size * hidden_size * 4;
        float* dwhdata = dwhptr + l * hidden_size * hidden_size * 4;
        float* dbiasdata = dbiasptr + l * hidden_size * 4;
        checkCudaErr(cudaMemsetAsync(dhptr, 0, batch_size * hidden_size * sizeof(float)));
        checkCudaErr(cudaMemsetAsync(dcptr, 0, batch_size * hidden_size * sizeof(float)));

        float* dxlayer = (l == 0 ? dxptr : dyptr);
        const float* xlayer = (l == 0 ? x0ptr : (ymptr + (l - 1) * seq_len * batch_size * hidden_size));
        for (int s = seq_len - 1; s >= 0; s--) {
            const float* cdata = cnptr + s * num_layers * batch_size * hidden_size + l * batch_size * hidden_size;
            const float* prehdata = (s == 0 ? (h0ptr + l * batch_size * hidden_size)
                    : (hnptr + (s - 1) * num_layers * batch_size * hidden_size + l * batch_size * hidden_size));
            const float* precdata = (s == 0 ? (c0ptr + l * batch_size * hidden_size)
                    : (cnptr + (s - 1) * num_layers * batch_size * hidden_size + l * batch_size * hidden_size));
            const float* ifogdata = ifogptr + l * seq_len * batch_size * hidden_size * 4 + s * batch_size * hidden_size * 4;
            const float* dydata = dyptr + s * batch_size * hidden_size;
            const float* xdata = xlayer + s * batch_size * wxidx[l];
            float* dxdata = dxlayer + s * batch_size * wxidx[l];
            {
                dim3 block_size = {DEFAULT_WARP_NUM * WARP_SIZE, 1, 1};
                dim3 grid_size = {(hidden_size + block_size.x - 1) / block_size.x, batch_size, 1};
                activation_backward<<<grid_size, block_size>>>(
                        batch_size, hidden_size, dydata, cdata, precdata, ifogdata,
                        dgatebufptr, dhptr, dcptr, dbiasdata);
            }

            // layernorm
            const float* ln_x = ln_in_ptr + l * seq_len * batch_size * hidden_size * 4 * 2 +
                s * batch_size * hidden_size * 4;
            const float* ln_h = ln_in_ptr + l * seq_len * batch_size * hidden_size * 4 * 2 +
                seq_len * batch_size * hidden_size * 4 + s * batch_size * hidden_size * 4;
            const float* ln_mean_x = ln_mean_ptr + l * seq_len * batch_size * 2 + s * batch_size;
            const float* ln_mean_h = ln_mean_ptr + l * seq_len * batch_size * 2 + seq_len * batch_size + s * batch_size;
            const float* ln_rstd_x = ln_rstd_ptr + l * seq_len * batch_size * 2 + s * batch_size;
            const float* ln_rstd_h = ln_rstd_ptr + l * seq_len * batch_size * 2 + seq_len * batch_size + s * batch_size;
            {
                unsigned int block_size = DEFAULT_WARP_NUM * WARP_SIZE;
                unsigned int grid_size = batch_size;
                // xbufptr has seq_len blocks(for fp), bp only use the first block
                layernorm_backward<<<grid_size, block_size>>>(
                        hidden_size * 4, dgatebufptr, ln_x, ln_mean_x, ln_rstd_x, ln_gamma_x, ln_dgamma_x, ln_dbeta_x, xbufptr);
                layernorm_backward<<<grid_size, block_size>>>(
                        hidden_size * 4, dgatebufptr, ln_h, ln_mean_h, ln_rstd_h, ln_gamma_h, ln_dgamma_h, ln_dbeta_h, hbufptr);
            }

            // dwx += torch.matmul(x_t, d_gate)
            checkCublasErr(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                        hidden_size * 4, wxidx[l], batch_size,
                        &onedata, xbufptr, hidden_size * 4, xdata, wxidx[l],
                        &onedata, dwxdata, hidden_size * 4));

            // dwh += torch.matmul(h_t, d_gate)
            checkCublasErr(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                        hidden_size * 4, hidden_size, batch_size,
                        &onedata, hbufptr, hidden_size * 4, prehdata, hidden_size,
                        &onedata, dwhdata, hidden_size * 4));

            // dx = torch.matmul(d_gate, wx_t)
            checkCublasErr(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        wxidx[l], batch_size, hidden_size * 4,
                        &onedata, wxdata, hidden_size * 4, xbufptr, hidden_size * 4,
                        &zerodata, dxdata, wxidx[l]));

            // dh = torch.matmul(d_gate, wh_t)
            checkCublasErr(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        hidden_size, batch_size, hidden_size * 4,
                        &onedata, whdata, hidden_size * 4, hbufptr, hidden_size * 4,
                        &zerodata, dhptr, hidden_size));
        }
    }

    // destroy handles
    checkCublasErr(cublasDestroy(cublas_handle));
}

}  // namespace cuda
}  // namespace rll
}  // namespace hpc
