#include "rmsnorm.h"
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

    rmsnorm_naive<<<griddim,blockdim>>>(din,dout,M,N);

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