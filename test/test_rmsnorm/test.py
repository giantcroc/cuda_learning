import torch
from torch.utils.cpp_extension import load

rmsnorm_kernels = load("rmsnorm", sources="rmsnorm_ops.cu", extra_include_paths="./",verbose=True)

def rmsnorm_residual(I:torch.Tensor,R,W,eps):
    R += I
    S = torch.rsqrt(torch.mean(torch.square(R),dim=1,keepdim=True)+eps)
    return R*S*W

B = 1024
H = 1024
eps =1e-5
I = torch.rand(B,H,dtype=torch.float32, device="cuda:0")
R = torch.zeros(B,H,dtype=torch.float32, device="cuda:0")
W = torch.ones(H,dtype=torch.float32, device="cuda:0")

I@R.T

R1 = R.clone()
O1 = rmsnorm_residual(I,R1,W,eps)


O2 = torch.zeros_like(O1,dtype=torch.float32,device="cuda:0")
R2 = R.clone()

rmsnorm_kernels.torchRmsnormNaive(I,O2,R2,W,B,H,eps)

print(torch.allclose(O1,O2,1e-2,1e-2))
print(torch.allclose(R1,R2,1e-2,1e-2))

O3 = torch.zeros_like(O1,dtype=torch.float32,device="cuda:0")
R3 = R.clone()

rmsnorm_kernels.torchRmsnormBlock(I,O3,R3,W,B,H,eps)

print(torch.allclose(O1,O3,1e-2,1e-2))
print(torch.allclose(R1,R3,1e-2,1e-2))