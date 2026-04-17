#include "cuda_runtime.h"


__global__ void sgemm_naive(float* A, float* B, float* C, const int M, const int N, const int K);

__global__ void sgemm_shared(float* A, float* B, float* C, const int M, const int N, const int K);

__global__ void sgemm_thread_tiled(float* A, float* B, float* C, const int M, const int N, const int K);

__global__ void sgemm_thread_tiled_bcf(float* A, float* B, float* C, const int M, const int N, const int K);

