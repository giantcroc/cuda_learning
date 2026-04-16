import torch
from typing import Tuple
from torch.utils.cpp_extension import load

ext_module = load(name="torchRmsnormNaive",
                  extra_include_paths=["./"] ,
                  sources=["rmsnorm_ops.cu"],
                  verbose=True)

# M=1024
# N=4
# data=torch.arange(1,5,dtype=torch.float16, device='cuda:0').reshape(1,4)

M=1024
N=1024
data=torch.rand(M,N,dtype=torch.float16,device="cuda:0")
residual = torch.zeros_like(data,dtype=torch.float16,device="cuda:0")
weight = torch.ones(N,dtype=torch.float16,device="cuda:0")



#warm up
torch.nn.functional.rms_norm(data,[N])

def rms_norm_residual(input:torch.Tensor,residual:torch.Tensor,eps:float) -> Tuple[torch.Tensor, torch.Tensor]:
    residual_out = input + residual
    output = residual_out*torch.rsqrt(torch.mean(residual_out.square(),1)+eps)
    return output,residual_out

start=torch.cuda.Event(enable_timing=True)
end=torch.cuda.Event(enable_timing=True)

start.record()
toutput=torch.nn.functional.rms_norm(data,[N],eps=1e-5)
end.record()
torch.cuda.synchronize()
print(f"torch rmsnorm: {start.elapsed_time(end)}")

start.record()
toutput1,residual_out=rms_norm_residual(data,residual,eps=1e-5)
end.record()
torch.cuda.synchronize()
print(f"rms_norm_residual: {start.elapsed_time(end)}")
print(toutput1)

# start.record()
# ext_module.torchRmsnormNaive(data,toutput, M,N,32)
# end.record()
# torch.cuda.synchronize()
# print(f"torchRmsnormNaive: {start.elapsed_time(end)}")

# # print(toutput)
# toutput1 = torch.zeros_like(data,dtype=torch.float16,device="cuda:0")
# start.record()
# ext_module.torchRmsnormShared(data,toutput1, M,N,32)
# end.record()
# torch.cuda.synchronize()
# print(f"torchRmsnormShared: {start.elapsed_time(end)}")

print(torch.allclose(toutput, toutput1))

toutput2 = torch.zeros_like(data,dtype=torch.float16,device="cuda:0")
start.record()
ext_module.torchRmsnormWarpPack(data,residual,toutput2,weight, M,N)
end.record()
torch.cuda.synchronize()
print(f"torchRmsnormWarpPack: {start.elapsed_time(end)}")

print(torch.allclose(toutput, toutput2,1e-2,1e-2))
print(torch.allclose(residual, data,1e-2,1e-2))
# print(toutput)
# print(toutput2)

