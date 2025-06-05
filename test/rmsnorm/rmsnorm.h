#include "cuda_runtime.h"


__global__ void rmsnorm_naive(float* in, float* out, const int M, const int N, float eps=1e-5);

__global__ void rmsnorm_shared(float* in, float* out, const int M, const int N, float eps=1e-5);
