#include<cuda.h>
#include "stdio.h"

#define WARPSIZE 32

__global__ void rms_norm_residual_naive(float* input, float* residual, float* residual_output,float* output, float* weights, int batch_size, int hidden_size, float eps){
    const int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<batch_size){
        input += idx*hidden_size;
        residual += idx*hidden_size;
        output += idx*hidden_size;

        float sum=0.0f;
        float sum_var=0.0f;
        for(int i=0;i<hidden_size;i++){
            float val=input[i]+residual[i];
            residual_output[i]=val;
            sum+=val*val;
        }

        sum_var = rsqrtf(sum/hidden_size+eps);

        for(int i=0;i<hidden_size;i++){
            sum=input[i]+residual[i];
            output[i]=sum*sum_var*weights[i];
        }

    }
}

__device__ float WarpReduceSum(float val){
    for(int offset=WARPSIZE/2;offset>0;offset/=2){
        val+=__shfl_xor_sync(0xffffffff,val,offset);
    }
    return val;
}

template<int NUM_THREADS>
__device__ float BlockReduceSum(float val){
    constexpr int NUM_WARPS=(NUM_THREADS+WARPSIZE-1)/WARPSIZE;
    __shared__ float shared[NUM_WARPS];
    const int laneid = threadIdx.x % WARPSIZE;
    const int warpid = threadIdx.x / WARPSIZE;

    val = WarpReduceSum(val);

    if(laneid==0){
        shared[warpid]=val;
    }

    __syncthreads();

    val=laneid<NUM_WARPS?shared[laneid]:0.0f;

    val=WarpReduceSum(val);
    return val;
}

template<int NUM_THREADS>
__global__ void rms_norm_residual_multi_warps(float* input, float* residual, float* residual_output, float* output, float* weights, int batch_size, int hidden_size, float eps){
    float sum=0.0f;
    float sum_var=0.0f;
    const int tid=threadIdx.x;
    const int idx=blockIdx.x;

    input += idx*hidden_size;
    residual += idx*hidden_size;
    output += idx*hidden_size;

    for(int i=tid;i<hidden_size;i+=blockDim.x){
        float val=input[i]+residual[i];
        residual_output[i]=val;
        sum+=val*val;
    }

    __syncthreads();
    sum = BlockReduceSum<NUM_THREADS>(sum);

    sum_var = rsqrtf(sum/hidden_size+eps);

    for(int i=tid;i<hidden_size;i+=blockDim.x){
        sum=input[i]+residual[i];
        output[i]=sum*sum_var*weights[i];
    }   
}

template<int NUM_THREADS,int UNROLL>
__global__ void rms_norm_residual_multi_warps_unroll(float* input, float* residual, float* residual_output, float* output, float* weights, int batch_size, int hidden_size, float eps){
    float sum=0.0f;
    float sum_var=0.0f;
    float vals[UNROLL];
    const int tid=threadIdx.x;
    const int idx=blockIdx.x;
    const int laneid=tid % WARPSIZE;
    const int warpid=tid / WARPSIZE;

    const int read_offset=warpid*WARPSIZE*UNROLL;

    input += idx*hidden_size;
    residual += idx*hidden_size;
    output += idx*hidden_size;


    for(int i=0;i<UNROLL;i++){
        int index=read_offset+i*WARPSIZE+laneid;
        if(index<hidden_size){
            float val=input[index]+residual[index];
            vals[i]=val;
            residual_output[index]=val;
            sum+=val*val;
        }
    }

    __syncthreads();
    sum = BlockReduceSum<NUM_THREADS>(sum);

    sum_var = rsqrtf(sum/hidden_size+eps);

    for(int i=0;i<UNROLL;i++){
        int index=i*WARPSIZE+laneid;
        if(index+read_offset<hidden_size){
            sum=vals[i];
            output[index+read_offset]=sum*sum_var*weights[i];
        }   
    }
}

template<typename scalar_t>
struct TypeTraits{};

template<>
struct TypeTraits<float>{
    typedef float2 Type;
    static const int Packed{2};
    static __device__ float F2T(float x){return x;}
    static __device__ float T2F(float x){return x;}
};


template<int NUM_THREADS,int UNROLL>
__global__ void rms_norm_residual_multi_warps_unroll_packed(float* input, float* residual, float* residual_output, float* output, float* weights, int batch_size, int hidden_size, float eps){
    using load_type = TypeTraits<float>::Type;
    const int VEC_SIZE = TypeTraits<float>::Packed;
    auto F2T = TypeTraits<float>::F2T;
    auto T2F = TypeTraits<float>::T2F;
    const int tid=threadIdx.x;
    const int idx=blockIdx.x;
    const int laneid=tid % WARPSIZE;
    const int warpid=tid / WARPSIZE;

    float sum=0.0f;
    float sum_var=0.0f;
    load_type vals[UNROLL];

    const int read_offset=warpid*WARPSIZE*UNROLL*VEC_SIZE;

    input += idx*hidden_size;
    residual += idx*hidden_size;
    output += idx*hidden_size;


    for(int i=0;i<UNROLL;i++){
        int index=read_offset+(i*WARPSIZE+laneid)*VEC_SIZE;
        if(index<hidden_size){
            load_type value = reinterpret_cast<load_type*>(input+index)[0];
            load_type resi = reinterpret_cast<load_type*>(residual+index)[0];
#pragma unroll
            for(int j=0;j<VEC_SIZE;j++){
                reinterpret_cast<float*>(&value)[j]=F2T(T2F(reinterpret_cast<float*>(&value)[j])+T2F(reinterpret_cast<float*>(&resi)[j]));
                reinterpret_cast<float*>(&resi)[j]+=reinterpret_cast<float*>(&value)[j];
            }

            vals[i]=value;

            reinterpret_cast<load_type*>(residual+index)[0]=resi;
#pragma unroll
            for(int j=0;j<VEC_SIZE;j++){
                float val = T2F(reinterpret_cast<float*>(&value)[j]);
                sum += val*val;
            }
        }
    }

    __syncthreads();
    sum = BlockReduceSum<NUM_THREADS>(sum);

    sum_var = rsqrtf(sum/hidden_size+eps);

    for(int i=0;i<UNROLL;i++){
        int index=(i*WARPSIZE+laneid)*VEC_SIZE;
        if(index+read_offset<hidden_size){
            load_type weight = reinterpret_cast<load_type*>(weights+index)[0];
            load_type res;
#pragma unroll
            for(int j=0;j<VEC_SIZE;j++){            
                reinterpret_cast<float*>(&res)[j]=F2T(T2F(reinterpret_cast<float*>(&vals[i])[j])*T2F(reinterpret_cast<float*>(&weight)[j])*sum_var);
            }
            reinterpret_cast<load_type*>(output+index+read_offset)[0] = res;
        }   
    }
}
