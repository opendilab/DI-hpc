#ifndef HPC_RLL_CUDA_LSTM_KERNEL_H_
#define HPC_RLL_CUDA_LSTM_KERNEL_H_

#include "hpc/rll/cuda/common.h"
#include "hpc/rll/cuda/reduce.h"
#include "hpc/rll/cuda/basic_math.h"

namespace hpc {
namespace rll {
namespace cuda {

__global__ void layernorm(unsigned int N, const float* x, const float* gamma, const float* beta,
        float* x_mean, float* x_rstd, float* y) {
	unsigned int block_start = blockIdx.x * N;
    unsigned int start = block_start + threadIdx.x;
	unsigned int end = block_start + N;

	// step 0. compute mean
	float val = 0.0;
	for (int i = start; i < end; i += blockDim.x) {
		val += x[i];
	}

	__shared__ float s_mean;
	float reduce_sum = blockReduceSum<float>(val);
	if (threadIdx.x == 0)
        s_mean = reduce_sum / float(N);
	__syncthreads();

	// step 1. compute variance
	val = 0.0;
	for (int i = start; i < end; i += blockDim.x) {
        val += x[i] * x[i];
	}
	__shared__ float s_rstd;
	float reduce_sum_sequare = blockReduceSum<float>(val);
	if (threadIdx.x == 0) {
        float var = reduce_sum_sequare / float(N) - s_mean * s_mean;
        var = max(var, 0.f);
        s_rstd = 1.f / sqrt(var + EPSILON);
    }
	__syncthreads();

	if (threadIdx.x == 0) {
        x_mean[blockIdx.x] = s_mean;
        x_rstd[blockIdx.x] = s_rstd;
    }

	// step 2. layer norm
	for (int i = start; i < end; i += blockDim.x) {
        y[i] = ((x[i] - s_mean) * s_rstd) * gamma[i - block_start] + beta[i - block_start];
	}
}

__global__ void activation(unsigned int batch_size, unsigned int hidden_size,
        const float* normx, const float* normh, const float* bias,
        const float* pre_h, const float* pre_c, float* h, float* c,
        float* ifogdata, float* output) {
    unsigned int gidx = blockIdx.x * blockDim.x + threadIdx.x; // hidden_size
    unsigned int gidy = blockIdx.y; // batch_size
    unsigned int start = gidy * hidden_size * 4;
    if (gidx < hidden_size) {
        float val[4];
        for (int i = 0; i < 4; i++) {
            val[i] = normx[start + i * hidden_size + gidx] + normh[start + i * hidden_size + gidx]
                + bias[i * hidden_size + gidx];
        }

        float i = sigmoid(val[0]);
        float f = sigmoid(val[1]);
        float o = sigmoid(val[2]);
        float g = tanh(val[3]);
        float old_c = pre_c[gidy * hidden_size + gidx];
        float new_c = f * old_c + i * g;
        float new_h = o * tanh(new_c);

        h[gidy * hidden_size + gidx] = new_h;
        c[gidy * hidden_size + gidx] = new_c;

        ifogdata[gidy * hidden_size * 4 + hidden_size * 0 + gidx] = i;
        ifogdata[gidy * hidden_size * 4 + hidden_size * 1 + gidx] = f;
        ifogdata[gidy * hidden_size * 4 + hidden_size * 2 + gidx] = o;
        ifogdata[gidy * hidden_size * 4 + hidden_size * 3 + gidx] = g;
        output[gidy * hidden_size + gidx] = new_h;
    }
}

__global__ void dropout(unsigned int stride, const unsigned int threshold,
        const float scale, const unsigned int* mask, float* data) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x; // maskstride
    if (gid < stride) {
        data[gid] = data[gid] * (mask[gid] > threshold) * scale;
    }
}

__global__ void layernorm_backward(unsigned int N, const float* dy,
        const float* x, const float* x_mean, const float* x_rstd,
        const float* gamma, float* dgamma, float* dbeta, float* dx) {
	unsigned int block_start = blockIdx.x * N;
    unsigned int start = block_start + threadIdx.x;
	unsigned int end = block_start + N;

    float ds = 0.0;
    float db = 0.0;
    for (int i = start; i < end; i += blockDim.x) {
        ds += dy[i] * x[i] * gamma[i - block_start];
        db += dy[i] * gamma[i - block_start];
    }

	__shared__ float s_ds;
	__shared__ float s_db;
    float reduced_ds = blockReduceSum<float>(ds);
    float reduced_db = blockReduceSum<float>(db);
	if (threadIdx.x == 0) {
        s_ds = reduced_ds;
        s_db = reduced_db;
    }
	__syncthreads();

    float scale = 1.f / N;
    float mean = x_mean[blockIdx.x];
    float rstd = x_rstd[blockIdx.x];
    float a = (s_db * mean - s_ds) * rstd * rstd * rstd * scale;
    float b = -a * mean - s_db * rstd * scale;
    for (int i = start; i < end; i += blockDim.x) {
        dx[i] = rstd * dy[i] * gamma[i - block_start] + a * x[i] + b;
        atomicAdd(&dgamma[i - block_start], dy[i] * (x[i] - mean) * rstd);
        atomicAdd(&dbeta[i - block_start], dy[i]);
    }
}

__global__ void activation_backward(unsigned int batch_size, unsigned int hidden_size,
        const float* dy, const float* c, const float* pre_c, const float* ifogdata,
        float* dgate, float* dh, float* dc, float* dbias) {
    unsigned int gidx = blockIdx.x * blockDim.x + threadIdx.x; // hidden_size
    unsigned int gidy = blockIdx.y; // batch_size
    if (gidx < hidden_size) {
        float i = ifogdata[gidy * hidden_size * 4 + hidden_size * 0 + gidx];
        float f = ifogdata[gidy * hidden_size * 4 + hidden_size * 1 + gidx];
        float o = ifogdata[gidy * hidden_size * 4 + hidden_size * 2 + gidx];
        float g = ifogdata[gidy * hidden_size * 4 + hidden_size * 3 + gidx];
        float dhdata = dh[gidy * hidden_size + gidx];
        float dydata = dy[gidy * hidden_size + gidx];
        float dcdata = dc[gidy * hidden_size + gidx];
        float cdata = c[gidy * hidden_size + gidx];
        float pre_cdata = pre_c[gidy * hidden_size + gidx];
        float tanh_c = tanh(cdata);

        dhdata += dydata;
        dcdata += dhdata * o * (1 - tanh_c * tanh_c);

        float didata = dcdata * g;
        float dfdata = dcdata * pre_cdata;
        float dodata = dhdata * tanh_c;
        float dgdata = dcdata * i;

        float dai = didata * i * (1 - i);
        float daf = dfdata * f * (1 - f);
        float dao = dodata * o * (1 - o);
        float dag = dgdata * (1 - g * g);

        dgate[gidy * hidden_size * 4 + hidden_size * 0 + gidx] = dai;
        dgate[gidy * hidden_size * 4 + hidden_size * 1 + gidx] = daf;
        dgate[gidy * hidden_size * 4 + hidden_size * 2 + gidx] = dao;
        dgate[gidy * hidden_size * 4 + hidden_size * 3 + gidx] = dag;

        dc[gidy * hidden_size + gidx] = dcdata * f;
        dh[gidy * hidden_size + gidx] = dhdata;

        atomicAdd(&dbias[hidden_size * 0 + gidx], dai);
        atomicAdd(&dbias[hidden_size * 1 + gidx], daf);
        atomicAdd(&dbias[hidden_size * 2 + gidx], dao);
        atomicAdd(&dbias[hidden_size * 3 + gidx], dag);
    }
}

__global__ void dropout_backward(unsigned int stride, const unsigned int threshold,
        const float scale, const unsigned int* mask, float* data) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x; // maskstride
    if (gid < stride) {
        data[gid] = data[gid] * (mask[gid] > threshold) * scale;
    }
}

}  // namespace cuda
}  // namespace rll
}  // namespace hpc
#endif // HPC_RLL_CUDA_LSTM_KERNEL_H_
