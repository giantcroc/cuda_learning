#include "rmsnorm.h"
#include "stdio.h"
#include <iostream>

int main(void){
    const int M=1024,N=8192;
    const int Block_size=32, data_size=M*N*sizeof(half);
    half* hin,*hout,*din,*dout, *dhout, *weight, *residual, *dweight,*dresidual;
    hin=(half*)malloc(data_size);
    residual=(half*)malloc(data_size);
    hout=(half*)malloc(data_size);
    dhout=(half*)malloc(data_size);
    weight=(half*)malloc(data_size);
    for(int i=0;i<M*N;i++){
        hin[i]=i+1;
        residual[i] = 0;
    }

    for(int i=0;i<N;i++){
        weight[i] =1;
    }

    cudaMalloc(&din, data_size);
    cudaMalloc(&dout,data_size);
    cudaMalloc(&dweight,data_size);
    cudaMalloc(&dresidual,data_size);

    cudaMemcpy(din,hin,data_size,cudaMemcpyDefault);
    cudaMemcpy(dweight,weight,data_size,cudaMemcpyDefault);
    cudaMemcpy(dresidual,residual,data_size,cudaMemcpyDefault);

    rms_norm_residual_kernel<half>(din,dresidual,dout,dweight,M,N,1,1e-5);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }
  
    cudaDeviceSynchronize();

    // 再次检查错误，确保设备同步
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA Error after synchronization: " << cudaGetErrorString(err) << std::endl;
    }

    cudaMemcpy(dhout, dout, data_size,cudaMemcpyDefault);

    // for(int i=0;i<M*N;i++){
    //     printf("%f ",__half2float(dhout[i]));
    // }

    cudaFree(din);
    cudaFree(dout);
    cudaFree(dweight);
    cudaFree(dresidual);
    free(hin);
    free(hout);
    free(dhout);
    free(residual);
    free(weight);
    return 0;
}