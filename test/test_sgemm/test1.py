import torch
from torch.utils.cpp_extension import load

sgemm_kernels = load("sgemm",
                     sources=["sgemm.cu","sgemm_ops.cu"],
                     extra_include_paths="./",
                     verbose=True)

M = 2048 
N = 2048
K = 1024

A = torch.rand(M,K,dtype=torch.float32, device="cuda:0")
B = torch.rand(K,N,dtype=torch.float32, device="cuda:0")

A@B

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
tC = A@B
end.record()
torch.cuda.synchronize()

print(f"torch matmul: {start.elapsed_time(end)}")

tC1 = torch.zeros_like(tC,device="cuda:0",dtype=torch.float32)
start.record()
sgemm_kernels.torchSgemmNaive(A,B,tC1,M,N,K)
end.record()
torch.cuda.synchronize()

print(f"torchSgemmNaive: {start.elapsed_time(end)}")

print(torch.allclose(tC,tC1,1e-2,1e-2))

tC2 = torch.zeros_like(tC,device="cuda:0",dtype=torch.float32)
start.record()
sgemm_kernels.torchSgemmShared(A,B,tC2,M,N,K)
end.record()
torch.cuda.synchronize()

print(f"torchSgemmShared: {start.elapsed_time(end)}")

print(torch.allclose(tC,tC2,1e-2,1e-2))

tC3 = torch.zeros_like(tC,device="cuda:0",dtype=torch.float32)
start.record()
sgemm_kernels.torchSgemmThreadTiled(A,B,tC3,M,N,K)
end.record()
torch.cuda.synchronize()

print(f"torchSgemmThreadTiled: {start.elapsed_time(end)}")

print(torch.allclose(tC,tC3,1e-2,1e-2))

tC4 = torch.zeros_like(tC,device="cuda:0",dtype=torch.float32)
start.record()
sgemm_kernels.torchSgemmThreadTiledBCF(A,B,tC4,M,N,K)
end.record()
torch.cuda.synchronize()

print(f"torchSgemmThreadTiledBCF: {start.elapsed_time(end)}")

print(torch.allclose(tC,tC4,1e-2,1e-2))


tC5 = torch.zeros_like(tC,device="cuda:0",dtype=torch.float32)
start.record()
sgemm_kernels.torchSgemmThreadTiledBCFDbuf(A,B,tC5,M,N,K)
end.record()
torch.cuda.synchronize()

print(f"torchSgemmThreadTiledBCFDbuf: {start.elapsed_time(end)}")

print(torch.allclose(tC,tC5,1e-2,1e-2))