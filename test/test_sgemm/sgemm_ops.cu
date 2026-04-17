#include <torch/extension.h>
#include <cuda.h>
#include "sgemm.h"

void torchSgemmNaive(torch::Tensor& A,torch::Tensor& B,torch::Tensor& C, const int M, const int N, const int K){
    dim3 grid(M/32,N/32);
    dim3 block(32,32);

    sgemm_naive<<<grid,block>>>((float*)A.data_ptr(), (float*)B.data_ptr(), (float*)C.data_ptr(), M, N, K);
}

void torchSgemmShared(torch::Tensor& A,torch::Tensor& B,torch::Tensor& C, const int M, const int N, const int K){
    dim3 grid(M/32,N/32);
    dim3 block(32,32);

    sgemm_shared<<<grid,block>>>((float*)A.data_ptr(), (float*)B.data_ptr(), (float*)C.data_ptr(), M, N, K);
}

void torchSgemmThreadTiled(torch::Tensor& A,torch::Tensor& B,torch::Tensor& C, const int M, const int N, const int K){
    dim3 grid(M/128,N/128);
    dim3 block(16,16);

    sgemm_thread_tiled<<<grid,block>>>((float*)A.data_ptr(), (float*)B.data_ptr(), (float*)C.data_ptr(), M, N, K);
}

void torchSgemmThreadTiledBCF(torch::Tensor& A,torch::Tensor& B,torch::Tensor& C, const int M, const int N, const int K){
    dim3 grid(M/128,N/128);
    dim3 block(16,16);

    sgemm_thread_tiled_bcf<<<grid,block>>>((float*)A.data_ptr(), (float*)B.data_ptr(), (float*)C.data_ptr(), M, N, K);
}

void torchSgemmThreadTiledBCFDbuf(torch::Tensor& A,torch::Tensor& B,torch::Tensor& C, const int M, const int N, const int K){
    dim3 grid(M/128,N/128);
    dim3 block(16,16);

    sgemm_thread_tiled_bcf_dbuf<<<grid,block>>>((float*)A.data_ptr(), (float*)B.data_ptr(), (float*)C.data_ptr(), M, N, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("torchSgemmNaive",
    &torchSgemmNaive,
    "torchSgemmNaive");
    m.def("torchSgemmShared",
    &torchSgemmShared,
    "torchSgemmShared");
    m.def("torchSgemmThreadTiled",
    &torchSgemmThreadTiled,
    "torchSgemmThreadTiled");
    m.def("torchSgemmThreadTiledBCF",
    &torchSgemmThreadTiledBCF,
    "torchSgemmThreadTiledBCF");
    m.def("torchSgemmThreadTiledBCFDbuf",
    &torchSgemmThreadTiledBCFDbuf,
    "torchSgemmThreadTiledBCFDbuf");
}