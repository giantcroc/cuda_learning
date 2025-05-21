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

int main(void){
    const int M=1024,N=1024;
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
    dim3 griddim((M+Block_size-1)/Block_size);
    dim3 blockdim(Block_size);
    softmax_naive<<<griddim,blockdim>>>(din,dout,M,N);

    cudaMemcpy(dhout, dout, data_size,cudaMemcpyDefault);

    // for(int i=0;i<M*N;i++){
    //     printf("%f ",dhout[i]);
    // }

    cudaFree(din);
    cudaFree(dout);
    free(hin);
    free(hout);
    free(dhout);
    return 0;
}