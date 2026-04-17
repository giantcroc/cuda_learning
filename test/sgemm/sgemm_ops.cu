#include "sgemm.h"
#include <torch/extension.h>

void torchSgemmNaive(torch::Tensor& A, torch::Tensor& B, torch::Tensor& C,const int M, const int N, const int K) {

    const int block_size=32;
    dim3 griddim((N+block_size-1)/block_size,(M+block_size-1)/block_size);
    dim3 blockdim(block_size,block_size);
    sgemm_naive<<<griddim,blockdim>>>((float*)A.data_ptr(),(float*)B.data_ptr(),(float*)C.data_ptr(),M,N,K);
}

void torchSgemmShared(torch::Tensor& A, torch::Tensor& B, torch::Tensor& C,const int M, const int N, const int K) {

    const int block_size=32;
    dim3 griddim((N+block_size-1)/block_size,(M+block_size-1)/block_size);
    dim3 blockdim(block_size,block_size);
    sgemm_shared<32,32,32><<<griddim,blockdim>>>((float*)A.data_ptr(),(float*)B.data_ptr(),(float*)C.data_ptr(),M,N,K);
}

void torchSgemmThreadTiled(torch::Tensor& A, torch::Tensor& B, torch::Tensor& C,const int M, const int N, const int K) {

    const int BM=128;
    const int BN=128;
    const int TM=8;
    const int TN=8;
    dim3 griddim((N+BN-1)/BN,(M+BM-1)/BM);
    dim3 blockdim(BN/TN,BM/TM);
    sgemm_thread_tiled<<<griddim,blockdim>>>((float*)A.data_ptr(),(float*)B.data_ptr(),(float*)C.data_ptr(),M,N,K);
}

void torchSgemmThreadTiledBCF(torch::Tensor& A, torch::Tensor& B, torch::Tensor& C,const int M, const int N, const int K) {

    const int BM=128;
    const int BN=128;
    const int TM=8;
    const int TN=8;
    dim3 griddim((N+BN-1)/BN,(M+BM-1)/BM);
    dim3 blockdim(BN/TN,BM/TM);
    sgemm_thread_tiled_bcf<<<griddim,blockdim>>>((float*)A.data_ptr(),(float*)B.data_ptr(),(float*)C.data_ptr(),M,N,K);
}
void torchSgemmThreadTiledBCFDbuf(torch::Tensor& A, torch::Tensor& B, torch::Tensor& C,const int M, const int N, const int K) {

    const int BM=128;
    const int BN=128;
    const int TM=8;
    const int TN=8;
    dim3 griddim((N+BN-1)/BN,(M+BM-1)/BM);
    dim3 blockdim(BN/TN,BM/TM);
    sgemm_thread_tiled_bcf_dbuf<<<griddim,blockdim>>>((float*)A.data_ptr(),(float*)B.data_ptr(),(float*)C.data_ptr(),M,N,K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torchSgemmNaive",
          &torchSgemmNaive,
          "torchSgemmNaive");
    m.def("torchSgemmThreadTiled",
          &torchSgemmThreadTiled,
          "torchSgemmThreadTiled");
    m.def("torchSgemmThreadTiledBCF",
          &torchSgemmThreadTiledBCF,
          "torchSgemmThreadTiledBCF");
    m.def("torchSgemmThreadTiledBCFDbuf",
          &torchSgemmThreadTiledBCFDbuf,
          "torchSgemmThreadTiledBCFDbuf");
    m.def("torchSgemmShared",
          &torchSgemmShared,
          "torchSgemmShared");
}