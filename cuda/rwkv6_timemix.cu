#include <torch/extension.h>
#include "ATen/ATen.h"

__global__ void time_mixing_kernel(
    int batch_size, int seq_len, int n_embd,
    const float* x, const float* xx, const float* time_maa_x,
    const float* time_maa_w, const float* time_maa_k,
    const float* time_maa_v, const float* time_maa_r,
    const float* time_maa_w1, const float* time_maa_w2,
    float* xw, float* xk, float* xv, float* xr
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * seq_len * n_embd) return;

    int i = idx / (seq_len * n_embd);
    int j = (idx / n_embd) % seq_len;
    int k = idx % n_embd;

    // Shared memory for caching, sized dynamically
    extern __shared__ float smem[];
    float* shared_time_maa_x = smem;
    float* shared_time_maa_w = &shared_time_maa_x[n_embd];
    float* shared_time_maa_k = &shared_time_maa_w[n_embd];
    float* shared_time_maa_v = &shared_time_maa_k[n_embd];
    float* shared_time_maa_r = &shared_time_maa_v[n_embd];

    // Load into shared memory (only the first thread in each block does this)
    if (threadIdx.x == 0) {
        for (int l = 0; l < n_embd; l++) { 
            shared_time_maa_x[l] = time_maa_x[l];
            shared_time_maa_w[l] = time_maa_w[l];
            shared_time_maa_k[l] = time_maa_k[l];
            shared_time_maa_v[l] = time_maa_v[l];
            shared_time_maa_r[l] = time_maa_r[l];
        }
    }
    __syncthreads();// Make sure all threads have loaded from global memory

    float x_val = x[idx];
    float xx_val = xx[idx];

    // Access from shared memory
    float xxx = x_val + xx_val * shared_time_maa_x[k]; 
    float mw = tanh(xxx * time_maa_w1[idx]) * time_maa_w2[idx];  // Coalesced access
    float mk = tanh(xxx * time_maa_w1[idx + batch_size * seq_len * n_embd]) * time_maa_w2[idx + batch_size * seq_len * n_embd]; 
    float mv = tanh(xxx * time_maa_w1[idx + 2 * batch_size * seq_len * n_embd]) * time_maa_w2[idx + 2 * batch_size * seq_len * n_embd]; 
    float mr = tanh(xxx * time_maa_w1[idx + 3 * batch_size * seq_len * n_embd]) * time_maa_w2[idx + 3 * batch_size * seq_len * n_embd]; 

    xw[idx] = x_val + xx_val * (shared_time_maa_w[k] + mw); 
    xk[idx] = x_val + xx_val * (shared_time_maa_k[k] + mk);
    xv[idx] = x_val + xx_val * (shared_time_maa_v[k] + mv);
    xr[idx] = x_val + xx_val * (shared_time_maa_r[k] + mr);
}

extern "C" void time_mixing_forward(
    int batch_size, int seq_len, int n_embd,
    const torch::Tensor& x, const torch::Tensor& xx,
    const torch::Tensor& time_maa_x, const torch::Tensor& time_maa_w,
    const torch::Tensor& time_maa_k, const torch::Tensor& time_maa_v,
    const torch::Tensor& time_maa_r, const torch::Tensor& time_maa_w1,
    const torch::Tensor& time_maa_w2,
    torch::Tensor& xw, torch::Tensor& xk, torch::Tensor& xv, torch::Tensor& xr
) {
    int num_elements = batch_size * seq_len * n_embd;
    int block_size = n_embd;
    int num_blocks = (num_elements + block_size - 1) / block_size;

    size_t shared_mem_size = 5 * n_embd * sizeof(float);

    time_mixing_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        batch_size, seq_len, n_embd,
        x.data_ptr<float>(), xx.data_ptr<float>(),
        time_maa_x.data_ptr<float>(), time_maa_w.data_ptr<float>(),
        time_maa_k.data_ptr<float>(), time_maa_v.data_ptr<float>(),
        time_maa_r.data_ptr<float>(), time_maa_w1.data_ptr<float>(),
        time_maa_w2.data_ptr<float>(),
        xw.data_ptr<float>(), xk.data_ptr<float>(),
        xv.data_ptr<float>(), xr.data_ptr<float>()
    );

    // Error checking after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in time_mixing_forward: %s\n", cudaGetErrorString(err));
    }

    // Synchronize to check for errors during kernel execution
    cudaDeviceSynchronize(); 
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error during kernel execution: %s\n", cudaGetErrorString(err));
    }
}