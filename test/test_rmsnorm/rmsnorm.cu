#include "rmsnorm.h"
#include "stdio.h"

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

int main(void){
    const int M=1,N=1025,data_size=M*N*sizeof(float);
    const float eps = 1e-5;
    float* hinput,*hresidual,*houtput1,*hweights,*houtput2,*dinput,*dresidual,*dresidual_output, *doutput1, *doutput2,*dweights;
    hinput=(float*)malloc(data_size);
    houtput1=(float*)malloc(data_size);
    houtput2=(float*)malloc(data_size);
    hweights=(float*)malloc(N*sizeof(float));
    hresidual=(float*)malloc(data_size);
    for(int i=0;i<M*N;i++){
        hinput[i]=i+1;
        hresidual[i]=0;
        if(i<N){
            hweights[i]=float(i)/20;
        }
    }

    cudaMalloc(&dinput, data_size);
    cudaMalloc(&dresidual, data_size);
    cudaMalloc(&dresidual_output, data_size);
    cudaMalloc(&doutput1, data_size);
    cudaMalloc(&doutput2, data_size);
    cudaMalloc(&dweights, N*sizeof(float));

    cudaMemcpy(dinput,hinput, data_size, cudaMemcpyDefault);
    cudaMemcpy(dresidual,hresidual, data_size, cudaMemcpyDefault);
    cudaMemcpy(dweights,hweights, N*sizeof(float), cudaMemcpyDefault);

    // {
    //     dim3 grid(M);
    //     dim3 block(32);

    //     rmsnorm_residual_naive<<<grid,block>>>(dinput, dresidual, dresidual_output, doutput1, dweights, M, N, eps);

    //     cudaError_t err = cudaGetLastError();
    //     if(err!=cudaSuccess){
    //         printf(" cuda error %s\n", cudaGetErrorString(err));
    //     }

    //     cudaDeviceSynchronize();

    //     if(err!=cudaSuccess){
    //         printf(" cuda error %s\n", cudaGetErrorString(err));
    //     }

    //     cudaMemcpy(houtput1,doutput1, data_size, cudaMemcpyDefault);
    // }

    // {
    //     dim3 grid(M);
    //     dim3 block(64);

    //     rms_norm_residual_multi_warps<64><<<grid,block>>>(dinput, dresidual, dresidual_output, doutput2, dweights, N, eps);

    //     cudaError_t err = cudaGetLastError();
    //     if(err!=cudaSuccess){
    //         printf(" cuda error %s\n", cudaGetErrorString(err));
    //     }

    //     cudaDeviceSynchronize();

    //     if(err!=cudaSuccess){
    //         printf(" cuda error %s\n", cudaGetErrorString(err));
    //     }        

    //     cudaMemcpy(houtput2,doutput2, data_size, cudaMemcpyDefault);
    // }

    for(int i=0;i<8;i++){
        printf("%f ",houtput1[i]);
    }
    printf("\n");

    for(int i=0;i<8;i++){
        printf("%f ",houtput2[i]);
    }
    printf("\n");

    for(int i=0;i<M*N;i++){
        if(houtput1[i]>houtput2[i]+1e-2||houtput1[i]<houtput2[i]-1e-2){
            printf("error data in (%d %d) %f %f \n", i/N,i%N,houtput1[i],houtput2[i]);
        }
    }

    cudaFree(dinput);
    cudaFree(doutput1);
    cudaFree(doutput2);
    cudaFree(dresidual);
    cudaFree(dresidual_output);
    cudaFree(dweights);
    free(hinput);
    free(houtput1);
    free(houtput2);
    free(hresidual);
    free(hweights);
    return 0;
}