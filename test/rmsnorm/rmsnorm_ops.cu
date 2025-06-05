#include "rmsnorm.h"
#include <torch/extension.h>

void torchRmsnormNaive(torch::Tensor& input,torch::Tensor& output,const int M, const int N, const int Block_size) {

    dim3 griddim((M+Block_size-1)/Block_size);
    dim3 blockdim(Block_size);
    rmsnorm_naive<<<griddim,blockdim>>>((float*)input.data_ptr(),(float*)output.data_ptr(),M,N);
}

void torchRmsnormShared(torch::Tensor& input,torch::Tensor& output,const int M, const int N, const int Block_size) {

    dim3 griddim(M);
    dim3 blockdim(Block_size);
    rmsnorm_shared<<<griddim,blockdim,Block_size*sizeof(float)>>>((float*)input.data_ptr(),(float*)output.data_ptr(),M,N);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torchRmsnormNaive",
          &torchRmsnormNaive,
          "torchRmsnormNaive");
    m.def("torchRmsnormShared",
          &torchRmsnormShared,
          "torchRmsnormShared");
}