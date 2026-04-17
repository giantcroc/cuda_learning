#include<cuda.h>
#include<iostream>

#define OFFSET(rid,cid,col) (rid*col+cid)
#define FLOAT4(data) (reinterpret_cast<float4*>(&data)[0])

__global__ void sgemm_naive(float* A, float* B, float* C, int M, int N, int K){
    const int idx_m = blockIdx.y*blockDim.y+threadIdx.y;
    const int idx_n = blockIdx.x*blockDim.x+threadIdx.x;

    if(idx_m<M&&idx_n<N){
        float val = 0.0f;
        for(int k=0;k<K;k++){
            val+=A[OFFSET(idx_m,k,K)]*B[OFFSET(k,idx_n,N)];
        }
        C[OFFSET(idx_m,idx_n,N)]=val;
    }
}

__global__ void sgemm_shared(float* A, float* B, float* C, int M, int N, int K){

    const int BM = 32;
    const int BN = 32;
    const int BK = 32;
    __shared__ float sA[BM][BK];
    __shared__ float sB[BK][BN];
    float sum=0.0f;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    int load_a_m_gmem = by*BM + ty;

    int load_b_n_gmem = bx*BN + tx;
    for(int bk=0;bk<K/BK;bk++){
        int load_a_k_gmem = bk*BK+tx;
        int load_b_k_gmem = bk*BK+ty;
        sA[ty][tx] = A[OFFSET(load_a_m_gmem,load_a_k_gmem,K)];
        sB[ty][tx] = B[OFFSET(load_b_k_gmem,load_b_n_gmem,N)];

        __syncthreads();

#pragma unroll
        for(int k=0;k<BK;k++){
            sum+=sA[ty][k]*sB[k][tx];
        }

        __syncthreads();
    }

    int store_c_m_gmem = load_a_m_gmem;
    int store_c_n_gmem = load_b_n_gmem;
    C[OFFSET(store_c_m_gmem,store_c_n_gmem,N)] = sum;

}

__global__ void sgemm_thread_tiled(float* A, float* B, float* C, int M, int N, int K){
    const int BM=128;
    const int BN=128;
    const int BK=8;
    const int TM=8;
    const int TN=8;

    __shared__ float sA[BM][BK];
    __shared__ float sB[BK][BN];

    float rC[TM][TN]={0.0f};

    const int tx=threadIdx.x;
    const int ty=threadIdx.y;
    const int bx=blockIdx.x;
    const int by=blockIdx.y;
    const int tid=ty*blockDim.x+tx;

    int load_a_m_smem = tid>>1;
    int load_a_k_smem = (tid&1)<<2;
    int load_b_k_smem = tid>>5;
    int load_b_n_smem = (tid&31)<<2;

    int load_a_m_gmem = by*BM+load_a_m_smem;
    int load_b_n_gmem = bx*BN+load_b_n_smem;

    for(int bk=0;bk<K/BK;bk++){
        int load_a_k_gmem = bk*BK+load_a_k_smem;
        int load_b_k_gmem = bk*BK+load_b_k_smem;
        FLOAT4(sA[load_a_m_smem][load_a_k_smem]) = FLOAT4(A[OFFSET(load_a_m_gmem,load_a_k_gmem,K)]);
        FLOAT4(sB[load_b_k_smem][load_b_n_smem]) = FLOAT4(B[OFFSET(load_b_k_gmem,load_b_n_gmem,N)]);

        __syncthreads();

#pragma unroll
        for(int k=0;k<BK;k++){
#pragma unroll
            for(int m=0;m<TM;m++){
                int load_a_comp_m=ty*TM+m;
#pragma unroll
                for(int n=0;n<TN;n++){
                    int load_b_comp_n=tx*TN+n;
                    rC[m][n]+=sA[load_a_comp_m][k]*sB[k][load_b_comp_n];
                }
            }            
        }

        __syncthreads();
    }
#pragma unroll
    for(int m=0;m<TM;m++){
        int store_c_m_gmem=by*BM+ty*TM+m;
#pragma unroll
        for(int n=0;n<TN;n+=4){
            int store_c_n_gmem=bx*BN+tx*TN+n;
            FLOAT4(C[OFFSET(store_c_m_gmem,store_c_n_gmem,N)]) = FLOAT4(rC[m][n]);
        }
    }

}

__global__ void sgemm_thread_tiled_bcf(float* A, float* B, float* C, int M, int N, int K){
    const int BM=128;
    const int BN=128;
    const int BK=8;
    const int TM=8;
    const int TN=8;

    __shared__ float sA[BK][BM];
    __shared__ float sB[BK][BN];

    float rC[TM][TN]={0.0f};
    float r_load_a[TM/2];
    float r_load_b[TN/2];
    float r_comp_a[TM];
    float r_comp_b[TN];


    const int tx=threadIdx.x;
    const int ty=threadIdx.y;
    const int bx=blockIdx.x;
    const int by=blockIdx.y;
    const int tid=ty*blockDim.x+tx;

    int load_a_m_smem = tid>>1;
    int load_a_k_smem = (tid&1)<<2;
    int load_b_k_smem = tid>>5;
    int load_b_n_smem = (tid&31)<<2;

    int load_a_m_gmem = by*BM+load_a_m_smem;
    int load_b_n_gmem = bx*BN+load_b_n_smem;

    for(int bk=0;bk<K/BK;bk++){
        int load_a_k_gmem = bk*BK+load_a_k_smem;
        int load_b_k_gmem = bk*BK+load_b_k_smem;
        FLOAT4(r_load_a[0]) = FLOAT4(A[OFFSET(load_a_m_gmem,load_a_k_gmem,K)]);
        FLOAT4(r_load_b[0]) = FLOAT4(B[OFFSET(load_b_k_gmem,load_b_n_gmem,N)]);

        sA[load_a_k_smem][load_a_m_smem] = r_load_a[0];
        sA[load_a_k_smem+1][load_a_m_smem] = r_load_a[1];
        sA[load_a_k_smem+2][load_a_m_smem] = r_load_a[2];
        sA[load_a_k_smem+3][load_a_m_smem] = r_load_a[3];

        FLOAT4(sB[load_b_k_smem][load_b_n_smem]) = FLOAT4(r_load_b[0]);

        __syncthreads();

#pragma unroll
        for(int k=0;k<BK;k++){
            FLOAT4(r_comp_a[0]) = FLOAT4(sA[k][ty*TM/2]);
            FLOAT4(r_comp_a[4]) = FLOAT4(sA[k][ty*TM/2+BM/2]);

            FLOAT4(r_comp_b[0]) = FLOAT4(sB[k][tx*TN/2]);
            FLOAT4(r_comp_b[4]) = FLOAT4(sB[k][tx*TN/2+BN/2]);
#pragma unroll
            for(int m=0;m<TM;m++){
#pragma unroll
                for(int n=0;n<TN;n++){
                    rC[m][n]+=r_comp_a[m]*r_comp_b[n];
                }
            }            
        }

        __syncthreads();
    }
#pragma unroll
    for(int m=0;m<TM/2;m++){
        int store_c_m_gmem=by*BM+ty*TM/2+m;
        int store_c_n_gmem=bx*BN+tx*TN/2;
        FLOAT4(C[OFFSET(store_c_m_gmem,store_c_n_gmem,N)]) = FLOAT4(rC[m][0]);
        FLOAT4(C[OFFSET(store_c_m_gmem,store_c_n_gmem+BN/2,N)]) = FLOAT4(rC[m][4]);
    }

#pragma unroll
    for(int m=0;m<TM/2;m++){
        int store_c_m_gmem=by*BM+ty*TM/2+BM/2+m;
        int store_c_n_gmem=bx*BN+tx*TN/2;
        FLOAT4(C[OFFSET(store_c_m_gmem,store_c_n_gmem,N)]) = FLOAT4(rC[m+TM/2][0]);
        FLOAT4(C[OFFSET(store_c_m_gmem,store_c_n_gmem+BN/2,N)]) = FLOAT4(rC[m+TM/2][4]);
    }

}

__global__ void sgemm_thread_tiled_bcf_dbuf(float* A, float* B, float* C, int M, int N, int K){
    const int BM=128;
    const int BN=128;
    const int BK=8;
    const int TM=8;
    const int TN=8;

    __shared__ float sA[2][BK][BM];
    __shared__ float sB[2][BK][BN];

    float rC[TM][TN]={0.0f};
    float r_load_a[TM/2];
    float r_load_b[TN/2];
    float r_comp_a[TM];
    float r_comp_b[TN];


    const int tx=threadIdx.x;
    const int ty=threadIdx.y;
    const int bx=blockIdx.x;
    const int by=blockIdx.y;
    const int tid=ty*blockDim.x+tx;

    int load_a_m_smem = tid>>1;
    int load_a_k_smem = (tid&1)<<2;
    int load_b_k_smem = tid>>5;
    int load_b_n_smem = (tid&31)<<2;

    int load_a_m_gmem = by*BM+load_a_m_smem;
    int load_b_n_gmem = bx*BN+load_b_n_smem;

    {
        int load_a_k_gmem = load_a_k_smem;
        int load_b_k_gmem = load_b_k_smem;
        FLOAT4(r_load_a[0]) = FLOAT4(A[OFFSET(load_a_m_gmem,load_a_k_gmem,K)]);
        FLOAT4(r_load_b[0]) = FLOAT4(B[OFFSET(load_b_k_gmem,load_b_n_gmem,N)]);

        sA[0][load_a_k_smem][load_a_m_smem] = r_load_a[0];
        sA[0][load_a_k_smem+1][load_a_m_smem] = r_load_a[1];
        sA[0][load_a_k_smem+2][load_a_m_smem] = r_load_a[2];
        sA[0][load_a_k_smem+3][load_a_m_smem] = r_load_a[3];    

        
        FLOAT4(sB[0][load_b_k_smem][load_b_n_smem]) = FLOAT4(r_load_b[0]);

        __syncthreads();
    }

    int cur=0;
    int next=1;

    for(int bk=1;bk<K/BK;bk++){
      cur = (bk-1)&1;
      next = 1-cur;
#pragma unroll
        for(int k=0;k<BK;k++){
            FLOAT4(r_comp_a[0]) = FLOAT4(sA[cur][k][ty*TM/2]);
            FLOAT4(r_comp_a[4]) = FLOAT4(sA[cur][k][ty*TM/2+BM/2]);

            FLOAT4(r_comp_b[0]) = FLOAT4(sB[cur][k][tx*TN/2]);
            FLOAT4(r_comp_b[4]) = FLOAT4(sB[cur][k][tx*TN/2+BN/2]);
#pragma unroll
            for(int m=0;m<TM;m++){
#pragma unroll
                for(int n=0;n<TN;n++){
                    rC[m][n]+=r_comp_a[m]*r_comp_b[n];
                }
            }            
        }

        int load_a_k_gmem = bk*BK+load_a_k_smem;
        int load_b_k_gmem = bk*BK+load_b_k_smem;
        FLOAT4(r_load_a[0]) = FLOAT4(A[OFFSET(load_a_m_gmem,load_a_k_gmem,K)]);
        FLOAT4(r_load_b[0]) = FLOAT4(B[OFFSET(load_b_k_gmem,load_b_n_gmem,N)]);

        sA[next][load_a_k_smem][load_a_m_smem] = r_load_a[0];
        sA[next][load_a_k_smem+1][load_a_m_smem] = r_load_a[1];
        sA[next][load_a_k_smem+2][load_a_m_smem] = r_load_a[2];
        sA[next][load_a_k_smem+3][load_a_m_smem] = r_load_a[3];    

        
        FLOAT4(sB[next][load_b_k_smem][load_b_n_smem]) = FLOAT4(r_load_b[0]);

        __syncthreads();
    }

#pragma unroll
    for(int k=0;k<BK;k++){
        FLOAT4(r_comp_a[0]) = FLOAT4(sA[next][k][ty*TM/2]);
        FLOAT4(r_comp_a[4]) = FLOAT4(sA[next][k][ty*TM/2+BM/2]);

        FLOAT4(r_comp_b[0]) = FLOAT4(sB[next][k][tx*TN/2]);
        FLOAT4(r_comp_b[4]) = FLOAT4(sB[next][k][tx*TN/2+BN/2]);
#pragma unroll
        for(int m=0;m<TM;m++){
#pragma unroll
            for(int n=0;n<TN;n++){
                rC[m][n]+=r_comp_a[m]*r_comp_b[n];
            }
        }            
    }
#pragma unroll
    for(int m=0;m<TM/2;m++){
        int store_c_m_gmem=by*BM+ty*TM/2+m;
        int store_c_n_gmem=bx*BN+tx*TN/2;
        FLOAT4(C[OFFSET(store_c_m_gmem,store_c_n_gmem,N)]) = FLOAT4(rC[m][0]);
        FLOAT4(C[OFFSET(store_c_m_gmem,store_c_n_gmem+BN/2,N)]) = FLOAT4(rC[m][4]);
    }

#pragma unroll
    for(int m=0;m<TM/2;m++){
        int store_c_m_gmem=by*BM+ty*TM/2+BM/2+m;
        int store_c_n_gmem=bx*BN+tx*TN/2;
        FLOAT4(C[OFFSET(store_c_m_gmem,store_c_n_gmem,N)]) = FLOAT4(rC[m+TM/2][0]);
        FLOAT4(C[OFFSET(store_c_m_gmem,store_c_n_gmem+BN/2,N)]) = FLOAT4(rC[m+TM/2][4]);
    }

}

int main(void){
    const int M=128,N=128,K=32,block_size=32;
    float* hA,*hB,*hC,*dA,*dB,*dC,*hdC1, *hdC2;
    const int sizeA=M*K*sizeof(float);
    const int sizeB=K*N*sizeof(float);
    const int sizeC=M*N*sizeof(float);

    hA=(float*)malloc(sizeA);
    hB=(float*)malloc(sizeB);
    hC=(float*)malloc(sizeC);
    hdC1=(float*)malloc(sizeC);
    hdC2=(float*)malloc(sizeC);
    for(int i=0;i<M*K;i++){
        hA[i]=i+1;
    }
    for(int i=0;i<K*N;i++){
        hB[i]=i+1;
    }

    cudaMalloc(&dA, sizeA);
    cudaMalloc(&dB, sizeB);
    cudaMalloc(&dC, sizeC);

    cudaMemcpy(dA,hA,sizeA,cudaMemcpyDefault);
    cudaMemcpy(dB,hB,sizeB,cudaMemcpyDefault);

    {
        dim3 griddim(N/32,M/32);
        dim3 blockdim(32,32);

        sgemm_shared<<<griddim,blockdim>>>(dA,dB,dC,M,N,K);

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

        cudaMemcpy(hdC1, dC, sizeC,cudaMemcpyDefault);
    }

    {
        dim3 griddim(N/32,M/32);
        dim3 blockdim(32,32);

        sgemm_naive<<<griddim,blockdim>>>(dA,dB,dC,M,N,K);

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

        cudaMemcpy(hdC2, dC, sizeC,cudaMemcpyDefault);
    }



    for(int i=0;i<M*N;i++){
        if(i<10){
            printf("%f %f\n",hdC1[i],hdC2[i]);
        }
        if(hdC1[i]!=hdC2[i]){
            printf("%f %f\n",hdC1[i],hdC2[i]);
        }
    }
    printf("all same\n");
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    free(hA);
    free(hB);
    free(hC);
    free(hdC1);
    free(hdC2);
    return 0;
}