#include "cuda_runtime.h"


__global__ void softmax_naive(float* in, float* out, const int M, const int N);

__global__ void softmax_shm(float* in, float* out, const int M, const int N);

__global__ void softmax_warp(float* in, float* out, const int M, const int N);

__global__ void softmax_forward_kernel7(float* out, const float* inp, int N, int C);