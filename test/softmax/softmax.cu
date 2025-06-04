#include "softmax.h"
#include "stdio.h"

#define OFFSET(row, id, col) (row*col+id)
#define FLOAT4(data) (reinterpret_cast<float4*>(&(data))[0])

__global__ void softmax_naive(float* in, float* out, const int M, const int N){
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    if(idx<M){
        float* x=in + idx*N;
        float maxv=-INFINITY;
        for(int i=0;i<N;i++){
            maxv=max(maxv,x[i]);
        }
        float* y=out+idx*N;
        float sum=0;
        for(int i=0;i<N;i++){
            sum+=expf(x[i]-maxv);
        }
        for(int i=0;i<N;i++){
            y[i]=expf(x[i]-maxv)/sum;
        }
    }
}

__global__ void softmax_shm(float* in, float* out, const int M, const int N){
    extern __shared__ float shared[];
    int tid=threadIdx.x;
    int idx=blockIdx.x;
    const int Block_size=blockDim.x;

    float* x=in+idx*N;
    float maxv=-INFINITY;

    for(int i=tid;i<N;i+=Block_size){
        maxv=fmaxf(maxv,x[i]);
    }
    shared[tid]=maxv;
    __syncthreads();


    for(int stride=Block_size/2;stride>0;stride>>=1){
                __syncthreads();
        if(tid<stride){
            shared[tid]=fmaxf(shared[tid],shared[tid+stride]);
        }
    }

    __syncthreads();

    maxv=shared[0];

    float gsum=0;

    for(int i=tid;i<N;i+=Block_size){
        gsum+=expf(x[i]-maxv);
    }
    shared[tid]=gsum;
    __syncthreads();


    for(int stride=Block_size/2;stride>0;stride>>=1){
                __syncthreads();
        if(tid<stride){
            shared[tid]+=shared[tid+stride];
        }
    }

    __syncthreads();

    gsum=shared[0];

    float* y=out+idx*N;

    for(int i=tid;i<N;i+=Block_size){
        y[i]=expf(x[i]-maxv)/gsum;
    }
}

__device__ float warp_reduce_max(float val){
    for(int i=16;i>0;i>>=1){
        val=fmax(val,__shfl_down_sync(0xffffffff,val,i));
    }
    return val;
}

__device__ float warp_reduce_sum(float val){
    for(int i=16;i>0;i>>=1){
        val+=__shfl_down_sync(0xffffffff,val,i);
    }
    return val;
}

__global__ void softmax_warp(float* in, float* out, const int M, const int N){
    int tid=threadIdx.x;
    int idx=blockIdx.x;
    const int block_size=blockDim.x;
    const float* x=in+idx*N;
    float maxv=-INFINITY;
    for(int i=tid;i<N;i+=block_size){
        maxv=fmax(maxv,x[i]);
    }
    maxv=warp_reduce_max(maxv);

    maxv = __shfl_sync(0xFFFFFFFF, maxv, 0);

    float sum=0;
    for(int i=tid;i<N;i+=block_size){
        sum+=expf(x[i]-maxv);
    }
    sum=warp_reduce_sum(sum);

    sum = __shfl_sync(0xFFFFFFFF, sum, 0);

    float* y=out+idx*N;
    for(int i=tid;i<N;i+=block_size){
        y[i]=expf(x[i]-maxv)/sum;
    }
}
// warp-level reduction for finding the maximum value
__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// warp-level reduction for summing values
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void softmax_forward_kernel7(float* out, const float* inp, int N, int C) {
    // out is (N, C) just like inp. Each row of inp will get softmaxed.
    // same as kernel4, but optimised for very large Cs with advanced unrolling

    // The trick is to read into a register array (all indices known at compile time)
    // and always read UNROLL_FACTOR values to maximise memory level parallelism
    // even if we would be out of bounds, we set the index to min(C-1, idx)
    // so we just do some unnecessary reads (obviously bad for small C)
    // the writes are in a separate loop with a conditional check for out of bounds
    // making it separate is necessary to convince the compiler to do the right thing
    const int UNROLL_FACTOR = 8;
    const int warpsPerBlock = blockDim.x / 32;

    extern __shared__ float shared[];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int warpId = threadIdx.x / 32; // warp index within a block
    int laneId = threadIdx.x % 32; // thread index within a warp

    // shared[] must be allocated to have 2 * warpsPerBlock elements
    // first half for max values, the second half for sum values
    float* maxvals = shared;
    float* sumvals = &shared[warpsPerBlock];

    if (tid >= C) {
        maxvals[warpId] = -INFINITY;
        sumvals[warpId] = 0.0f;
        return;
    }

    const float* x = inp + idx * C; // input
    float* y = out + idx * C; // output

    // first, thread coarsening by directly accessing global memory in series
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x * UNROLL_FACTOR) {
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            maxval = fmaxf(maxval, x[min(C - 1, i + u*blockDim.x)]);
        }
    }

    // now within-warp reductions for maxval
    maxval = warpReduceMax(maxval);
    // the 0th thread of each warp writes the maxval of that warp to shared memory
    if (laneId == 0) maxvals[warpId] = maxval;
    __syncthreads();
    // now the 0th thread reduces the maxvals in shared memory, i.e. across warps
    if (tid == 0) {
        float val = maxvals[tid];
        #pragma unroll
        for (int i = 1; i < warpsPerBlock; i++) {
            val = fmaxf(val, maxvals[i]);
        }
        // store the final max in the first position
        maxvals[0] = val;
    }
    __syncthreads();
    // broadcast the max to all threads
    float offset = maxvals[0];

    // compute expf and write the result to global memory
    // + thread coarsening for sum
    float sumval = 0.0f;
    for (int i = tid; i < C; i += blockDim.x * UNROLL_FACTOR) {
        float reg_array[UNROLL_FACTOR];
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            reg_array[u] = __ldcs(&x[min(C - 1, i + u*blockDim.x)]);
        }
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            if (i + u*blockDim.x < C) {
                float output = expf(reg_array[u] - offset);
                y[min(C - 1, i + u*blockDim.x)] = output; // compiler likes redundant min()?!
                sumval += output; // combined into the same loop unlike kernel3
            }
        }
    }

    // okay now we calculated exp(x - max(x))
    // step 2: sum all the values and divide by the sum

    // within-warp reduction for sumval
    sumval = warpReduceSum(sumval);
    // write sumval to shared memory
    if (laneId == 0) sumvals[warpId] = sumval;
    __syncthreads();
    // inter-thread reduction of sum
    if (tid == 0) {
        float val = sumvals[tid];
        #pragma unroll
        for (int i = 1; i < warpsPerBlock; ++i) {
            val += sumvals[i];
        }
        sumvals[0] = val;
    }
    __syncthreads();
    // broadcast the sum to all threads
    float sum = sumvals[0];

    // divide the whole row by the sum
    for (int i = tid; i < C; i += blockDim.x * UNROLL_FACTOR) {
        float reg_array[UNROLL_FACTOR];
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            reg_array[u] = y[min(C - 1, i + u*blockDim.x)];
        }
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            if (i + u*blockDim.x < C) {
                y[i + u*blockDim.x] = reg_array[u] / sum;
            }
        }
    }
}


int main(void){
    const int M=1,N=4;
    const int Block_size=32, data_size=M*N*sizeof(float);
    float* hin,*hout,*din,*dout, *dhout;
    hin=(float*)malloc(data_size);
    hout=(float*)malloc(data_size);
    dhout=(float*)malloc(data_size);
    for(int i=0;i<M*N;i++){
        hin[i]=i+1;
    }

    cudaMalloc(&din, data_size);
    cudaMalloc(&dout,data_size);

    cudaMemcpy(din,hin,data_size,cudaMemcpyDefault);
    dim3 griddim(M);
    dim3 blockdim(Block_size);
    // softmax_naive<<<griddim,blockdim>>>(din,dout,M,N);
    // softmax_shm<<<griddim,blockdim,Block_size>>>(din,dout,M,N);

    softmax_warp<<<griddim,blockdim>>>(din,dout,M,N);

    cudaMemcpy(dhout, dout, data_size,cudaMemcpyDefault);

    for(int i=0;i<M*N;i++){
        printf("%f ",dhout[i]);
    }

    cudaFree(din);
    cudaFree(dout);
    free(hin);
    free(hout);
    free(dhout);
    return 0;
}