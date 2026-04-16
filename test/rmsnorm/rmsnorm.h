#pragma once
// #include "cuda_runtime.h"
#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include "cuda.h"
#include "stdio.h"


#define OFFSET(row, id, col) (row*col+id)
#define FLOAT4(data) (reinterpret_cast<float4*>(&(data))[0])

__global__ void rmsnorm_naive(float* in, float* out, const int M, const int N, float eps=1e-5){
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    if(idx<M){
        float* x=in + idx*N;
        float sum=0;
        for(int i=0;i<N;i++){
            float temp=x[i];
            sum+=temp*temp;
        }
        sum=sqrtf(sum/N+eps);
        float* y=out+idx*N;
        for(int i=0;i<N;i++){
            y[i]=x[i]/sum;
        }
    }
}

__global__ void rmsnorm_shared(float* in, float* out, const int M, const int N, float eps=1e-5){
    int Block_size = blockDim.x;
    extern __shared__ float shared[];
    int idx=blockIdx.x;
    int tid=threadIdx.x;

    float* x=in + idx*N;
    float sum=0;

    for(int i=tid;i<N;i+=Block_size){
        sum+=x[i]*x[i];
    }
    shared[tid]=sum;
    __syncthreads();

    for(int stride=Block_size/2;stride>0;stride/=2){
        __syncthreads();
        if(tid<stride){
            shared[tid]+=shared[tid+stride];
        }
    }
    sum=shared[0];
    sum=sqrtf(sum/N+eps);
    float* y=out+idx*N;
    for(int i=0;i<N;i++){
        y[i]=x[i]/sum;
    }

}

__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

template<int UNROLL_FACTOR>
__global__ void rms_norm_residual(float* input, float* residual, float* output, float* weight, int in_stride, int hidden_size, float alpha, float eps){
    const int laneid = threadIdx.x;
    const int idx = blockIdx.x;
    float val[UNROLL_FACTOR];
    float x2_sum = 0;
    float s_var = 0;

    input = input + idx*in_stride;
    residual = residual + idx*in_stride;
    output = output + idx*in_stride;

    for(int i=laneid;i<hidden_size;i+=blockDim.x){
        val[i] = input[i] + residual[i]*alpha;
        x2_sum += val[i]*val[i];
    }

    x2_sum = warpReduceSum(x2_sum);

    s_var = rsqrtf(x2_sum/hidden_size+eps);

    s_var = __shfl_sync(0xffffffff, s_var, 0);


    for(int i=laneid;i<hidden_size;i+=blockDim.x){
        output[i] = val[i]*s_var*weight[i];
    }

}

template<typename T>
struct TypeTraits {};

template<>
struct TypeTraits<float> {
    typedef float2 Type;
    static const int packed{2};
    static __device__ float F2T(float x){return x;}
    static __device__ float T2F(float x){return x;}
};

template<>
struct TypeTraits<half> {
    typedef half2 Type;
    static const int packed{2};
    static __device__ half F2T(float x){return __float2half(x);}
    static __device__ float T2F(half x){return __half2float(x);}
};

#define WarpSize 32

template<int NUM_THREADS>
__device__ float BlockReduceSum(float val){
    constexpr int NUM_WARPS = (NUM_THREADS + WarpSize - 1) / WarpSize;
    static __shared__ float shared[NUM_WARPS];
    const int warpid = threadIdx.x / WarpSize;
    const int laneid = threadIdx.x % WarpSize;
    const int num_real_warps = blockDim.x/WarpSize;

    val = warpReduceSum(val);

    if(laneid==0){
        shared[warpid]=val;
    }

    __syncthreads();

    val = (laneid < num_real_warps?shared[laneid]:0.0f);

    val = warpReduceSum(val);

    return val;

}

template<typename scalar_t, int UNROLL_FACTOR, int NUM_THREADS>
__global__ void rms_norm_residual_pack(scalar_t* input, scalar_t* residual, scalar_t* output, scalar_t* weights, int hidden_size, float alpha, float eps){
    using load_type = typename TypeTraits<scalar_t>::Type;
    constexpr int PACKED = TypeTraits<scalar_t>::packed;
    constexpr auto F2T = TypeTraits<scalar_t>::F2T;
    constexpr auto T2F = TypeTraits<scalar_t>::T2F;
    load_type vals[UNROLL_FACTOR];
    const int warpid = threadIdx.x / WarpSize;
    const int laneid = threadIdx.x % WarpSize;
    const int idx = blockIdx.x;
    const int read_offset = warpid*WarpSize*UNROLL_FACTOR*PACKED;

    float x2_sum = 0;
    float s_var = 0;

    input = input + idx*hidden_size + read_offset;
    residual = residual + idx*hidden_size +read_offset;
    output = output + idx*hidden_size + read_offset;

    weights += read_offset;

#pragma unroll
    for(int i=0;i<UNROLL_FACTOR;i++){
        int index=(i*WarpSize+laneid)*PACKED;
        if(read_offset+index<hidden_size){
            load_type value = reinterpret_cast<load_type*>(input+index)[0];
            load_type resi = reinterpret_cast<load_type*>(residual+index)[0];

#pragma unroll
            for(int j=0;j<PACKED;j++){
                reinterpret_cast<scalar_t*>(&(value))[j]=F2T(T2F(reinterpret_cast<scalar_t*>(&value)[j]) + T2F(reinterpret_cast<scalar_t*>(&resi)[j])*alpha);
                reinterpret_cast<scalar_t*>(&resi)[j]=reinterpret_cast<scalar_t*>(&(value))[j];
            }

            vals[i] = value;
            reinterpret_cast<load_type*>(residual+index)[0] = resi;

            
#pragma unroll
            for(int j=0;j<PACKED;j++){
                float val = T2F(reinterpret_cast<scalar_t*>(&(value))[j]);
                x2_sum += val*val; 
            }

        }
    }

    x2_sum = BlockReduceSum<NUM_THREADS>(x2_sum);
    s_var = rsqrtf(x2_sum/hidden_size+eps);


    #pragma unroll
    for(int i=0;i<UNROLL_FACTOR;i++){
        int index=(i*WarpSize+laneid)*PACKED;
        if(read_offset+index<hidden_size){
            load_type weight = reinterpret_cast<load_type*>(weights+index)[0];
            load_type res;

#pragma unroll
            for(int j=0;j<PACKED;j++){
                reinterpret_cast<scalar_t*>(&(res))[j]=F2T(T2F(reinterpret_cast<scalar_t*>(&vals[i])[j])*s_var*T2F(reinterpret_cast<scalar_t*>(&weight)[j]));
            }

            reinterpret_cast<load_type*>(output+index)[0] = res;
        }
    }
}

template<typename scalar_t>
void rms_norm_residual_kernel(scalar_t* input, scalar_t* residual, scalar_t* output, scalar_t* weights, int batch_size, int hidden_size, float alpha, float eps){
    dim3 grid(batch_size);
    int padding_size=(hidden_size + 2047)/2048;
    constexpr int packed = 2;
    constexpr int padding_threads = 1024;
    switch (padding_size){
        case 1: {
            constexpr int unroll = 1;
            int threads=hidden_size/(unroll*packed);
            threads = (threads + WarpSize-1)/WarpSize*WarpSize;
            rms_norm_residual_pack<scalar_t, 1, padding_threads><<<grid,threads>>>(input,residual,output,weights,hidden_size,alpha,eps);
            break;
        }
        case 2: {
            constexpr int unroll = 2;
            int threads=hidden_size/(unroll*packed);
            threads = (threads + WarpSize-1)/WarpSize*WarpSize;
            rms_norm_residual_pack<scalar_t, 2, padding_threads><<<grid,threads>>>(input,residual,output,weights,hidden_size,alpha,eps);
            break;
        }
        case 3: {
            constexpr int unroll = 3;
            int threads=hidden_size/(unroll*packed);
            threads = (threads + WarpSize-1)/WarpSize*WarpSize;
            rms_norm_residual_pack<scalar_t, 3, padding_threads><<<grid,threads>>>(input,residual,output,weights,hidden_size,alpha,eps);
            break;
        }
        case 4: {
            constexpr int unroll = 4;
            int threads=hidden_size/(unroll*packed);
            threads = (threads + WarpSize-1)/WarpSize*WarpSize;
            rms_norm_residual_pack<scalar_t, 4, padding_threads><<<grid,threads>>>(input,residual,output,weights,hidden_size,alpha,eps);
            break;
        }
        default:
            return;
    }
}