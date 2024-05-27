#include <torch/extension.h>
#include "ATen/ATen.h"

extern "C" void time_mixing_forward(
    int batch_size, int seq_len, int n_embd,
    const torch::Tensor& x, const torch::Tensor& xx,
    const torch::Tensor& time_maa_x, const torch::Tensor& time_maa_w,
    const torch::Tensor& time_maa_k, const torch::Tensor& time_maa_v,
    const torch::Tensor& time_maa_r, const torch::Tensor& time_maa_w1,
    const torch::Tensor& time_maa_w2,
    torch::Tensor& xw, torch::Tensor& xk, torch::Tensor& xv, torch::Tensor& xr
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("time_mixing_forward", &time_mixing_forward, "Time Mixing Forward (CUDA)");
}