#include<cuda.h>
#include "stdio.h"

#define WarpSize 32

__global__ void rmsnorm_residual_naive(float* input, float* output, float* residual, float* weights, int batch_size, int hidden_size, float eps){
    const int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<batch_size){
        const int offset = idx*hidden_size;
        input += offset;
        output += offset;
        residual += offset;

        float x2_sum=0.0f;
        float var =0.0f;
        for(int i=0;i<hidden_size;i++){
            float temp = input[i] + residual[i];
            residual[i]=temp;
            x2_sum += temp*temp;
        }

        var = rsqrtf(x2_sum/hidden_size+eps);

        for(int i=0;i<hidden_size;i++){
            output[i] = residual[i]*var*weights[i];
        }
    }
}

__device__ float warp_reduce_sum(float val){
    for(int i=WarpSize/2;i>0;i/=2){
        val+=__shfl_xor_sync(0xffffffff,val,i);
    }
    return val;
}

template<int NUM_THREADS>
__device__ float block_reduce_sum(float val){
    constexpr int NUM_WARP = (NUM_THREADS + WarpSize -1 )/WarpSize;
    __shared__ float shared[NUM_WARP];
    const int laneid = threadIdx.x % WarpSize;
    const int warpid = threadIdx.x / WarpSize;

    val = warp_reduce_sum(val);

    if(laneid == 0){
        shared[warpid] = val;
    }

    __syncthreads();

    val = laneid < NUM_WARP? shared[laneid]:0.0f;

    val=warp_reduce_sum(val);
    return val;
}

template<int NUM_THREADS>
__global__ void rmsnorm_residual_block(float* input, float* output, float* residual, float* weights, int batch_size, int hidden_size, float eps){
    const int idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;
    const int offset = idx*hidden_size;

    input += offset;
    output += offset;
    residual += offset;

    float x2_sum =0.0f;
    float var = 0.0f;
    for(int i=tid;i<hidden_size;i+=bdim){
        float val = input[i] + residual[i];
        residual[i] = val;
        x2_sum += val*val;
    }

    x2_sum = block_reduce_sum<NUM_THREADS>(x2_sum);
    var = rsqrtf(x2_sum/hidden_size+eps);

    for(int i=tid;i<hidden_size;i+=bdim){
        output[i] = residual[i]*var*weights[i];
    }    
}