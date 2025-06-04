#include "cuda_runtime.h"


__global__ void rmsnorm_naive(float* in, float* out, const int M, const int N, float eps);
