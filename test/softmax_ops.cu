#include "softmax.h"
#include <torch/extension.h>

void torchSoftmaxNaive(torch::Tensor& input,torch::Tensor& output,const int M, const int N, const int Block_size) {

    dim3 griddim((M+Block_size-1)/Block_size);
    dim3 blockdim(Block_size);
    softmax_naive<<<griddim,blockdim>>>((float*)input.data_ptr(),(float*)output.data_ptr(),M,N);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torchSoftmaxNaive",
          &torchSoftmaxNaive,
          "torchSoftmaxNaive");
}