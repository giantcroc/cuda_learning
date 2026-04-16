#include "rmsnorm.h"
#include <torch/extension.h>

// void torchRmsnormNaive(torch::Tensor& input,torch::Tensor& output,const int M, const int N, const int Block_size) {

//     dim3 griddim((M+Block_size-1)/Block_size);
//     dim3 blockdim(Block_size);
//     rmsnorm_naive<<<griddim,blockdim>>>((float*)input.data_ptr(),(float*)output.data_ptr(),M,N);
// }

void torchRmsnormShared(torch::Tensor& input,torch::Tensor& output,const int M, const int N, const int Block_size) {

    dim3 griddim(M);
    dim3 blockdim(Block_size);
    rmsnorm_shared<<<griddim,blockdim,Block_size*sizeof(float)>>>((float*)input.data_ptr(),(float*)output.data_ptr(),M,N);
}

void torchRmsnormWarp(torch::Tensor& input,torch::Tensor& residual, torch::Tensor& output,torch::Tensor& weight, int M, int N, const int Block_size) {

    dim3 griddim(M);
    dim3 blockdim(Block_size);

    if((N+Block_size-1)/Block_size==32){
        rms_norm_residual<32><<<griddim,blockdim>>>((float*)input.data_ptr(),(float*)residual.data_ptr(),(float*)output.data_ptr(),(float*)weight.data_ptr(),N,N,1,1e-5);
    }
    
}

void torchRmsnormWarpPack(torch::Tensor& input,torch::Tensor& residual, torch::Tensor& output,torch::Tensor& weight, int M, int N) {

    if(input.scalar_type()==torch::ScalarType::Float){
        rms_norm_residual_kernel<float>((float*)input.data_ptr(),(float*)residual.data_ptr(),(float*)output.data_ptr(),(float*)weight.data_ptr(),M,N,1,1e-5);
    }else if(input.scalar_type()==torch::ScalarType::Half){
        rms_norm_residual_kernel<half>((half*)input.data_ptr(),(half*)residual.data_ptr(),(half*)output.data_ptr(),(half*)weight.data_ptr(),M,N,1,1e-5);
    }


}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("torchRmsnormNaive",
    //       &torchRmsnormNaive,
    //       "torchRmsnormNaive");
    m.def("torchRmsnormShared",
          &torchRmsnormShared,
          "torchRmsnormShared");
    m.def("torchRmsnormWarp",
          &torchRmsnormWarp,
          "torchRmsnormWarp");
    m.def("torchRmsnormWarpPack",
          &torchRmsnormWarpPack,
          "torchRmsnormWarpPack");
}