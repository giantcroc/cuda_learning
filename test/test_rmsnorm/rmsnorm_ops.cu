#include "rmsnorm.h"
#include <torch/extension.h>

void torchRmsnormNaive(torch::Tensor& input, torch::Tensor&output, torch::Tensor&residual, torch::Tensor&weights, int batch_size, int hidden_size, float eps){
    dim3 grid((batch_size + 31)/32);
    dim3 block(32);

    rmsnorm_residual_naive<<<grid,block>>>((float*)input.data_ptr(),(float*)output.data_ptr(),(float*)residual.data_ptr(),(float*)weights.data_ptr(), batch_size, hidden_size, eps);
}

void torchRmsnormBlock(torch::Tensor& input, torch::Tensor&output, torch::Tensor&residual, torch::Tensor&weights, int batch_size, int hidden_size, float eps){
    dim3 grid(batch_size);
    dim3 block(128);

    rmsnorm_residual_block<128><<<grid,block>>>((float*)input.data_ptr(),(float*)output.data_ptr(),(float*)residual.data_ptr(),(float*)weights.data_ptr(), batch_size, hidden_size, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("torchRmsnormNaive", &torchRmsnormNaive, "torchRmsnormNaive");
    m.def("torchRmsnormBlock", &torchRmsnormBlock, "torchRmsnormBlock");
};