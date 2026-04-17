import torch
from torch.utils.cpp_extension import load

ext_module = load(name="torchSgemm",
                  extra_include_paths=["./"] ,
                  sources=["sgemm.cu", "sgemm_ops.cu"],
                  verbose=True)


M=1024
N=1024
K=1024
A=torch.rand(M,K,dtype=torch.float32, device='cuda:0')
B=torch.rand(K,N,dtype=torch.float32, device='cuda:0')
# print(A)
# print(B)

#warm up
A @ B

start=torch.cuda.Event(enable_timing=True)
end=torch.cuda.Event(enable_timing=True)

start.record()
tC=A@B
end.record()
torch.cuda.synchronize()
print(f"torch sgemm: {start.elapsed_time(end)}")

# print(tC)

C1=tC.clone()

start.record()
ext_module.torchSgemmNaive(A,B,C1,M,N,K)
end.record()
torch.cuda.synchronize()
print(f"torchSgemmNaive: {start.elapsed_time(end)}")

# print(C1)
print(torch.allclose(tC,C1, 1e-4,1e-4))


C2=torch.zeros_like(tC,dtype=torch.float32,device="cuda:0")

start.record()
ext_module.torchSgemmShared(A,B,C2,M,N,K)
end.record()
torch.cuda.synchronize()
print(f"torchSgemmShared: {start.elapsed_time(end)}")

# print(C2)
print(torch.allclose(tC,C2, 1e-4,1e-4))

# C3=torch.zeros_like(tC,dtype=torch.float32,device="cuda:0")

# start.record()
# ext_module.torchSgemmThreadTiledBCF(A,B,C3,M,N,K)
# end.record()
# torch.cuda.synchronize()
# print(f"torchSgemmThreadTiledBCF: {start.elapsed_time(end)}")

# print(torch.allclose(tC,C3, 1e-4,1e-4))

# C4=torch.zeros_like(tC,dtype=torch.float32,device="cuda:0")

# start.record()
# ext_module.torchSgemmThreadTiledBCFDbuf(A,B,C4,M,N,K)
# end.record()
# torch.cuda.synchronize()
# print(f"torchSgemmThreadTiledBCFDbuf: {start.elapsed_time(end)}")

# print(torch.allclose(tC,C3, 1e-4,1e-4))
