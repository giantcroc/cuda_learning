#include "softmax.h"
#include <torch/extension.h>

void torchSoftmaxNaive(torch::Tensor& input,torch::Tensor& output,const int M, const int N, const int Block_size) {

    dim3 griddim((M+Block_size-1)/Block_size);
    dim3 blockdim(Block_size);
    softmax_naive<<<griddim,blockdim>>>((float*)input.data_ptr(),(float*)output.data_ptr(),M,N);
}

void torchSoftmaxShared(torch::Tensor& input,torch::Tensor& output,const int M, const int N, const int Block_size) {

    dim3 griddim(M);
    dim3 blockdim(Block_size);
    softmax_shm<<<griddim,blockdim,Block_size * sizeof(float)>>>((float*)input.data_ptr(),(float*)output.data_ptr(),M,N);
}

void torchSoftmaxWarp(torch::Tensor& input,torch::Tensor& output,const int M, const int N, const int Block_size) {

    dim3 griddim(M);
    dim3 blockdim(Block_size);
    softmax_warp<<<griddim,blockdim>>>((float*)input.data_ptr(),(float*)output.data_ptr(),M,N);
}

void torchSoftmax7(torch::Tensor& input,torch::Tensor& output,const int M, const int N, const int Block_size) {

    dim3 griddim(M);
    dim3 blockdim(Block_size);
    softmax_forward_kernel7<<<griddim,blockdim,2*Block_size/32 * sizeof(float)>>>((float*)output.data_ptr(),(float*)input.data_ptr(),M,N);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torchSoftmaxNaive",
          &torchSoftmaxNaive,
          "torchSoftmaxNaive");
    m.def("torchSoftmaxShared",
          &torchSoftmaxShared,
          "torchSoftmaxShared");
    m.def("torchSoftmaxWarp",
          &torchSoftmaxWarp,
          "torchSoftmaxWarp");
    m.def("torchSoftmax7",
          &torchSoftmax7,
          "torchSoftmax7");
}