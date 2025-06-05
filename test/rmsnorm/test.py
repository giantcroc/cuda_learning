import torch
from torch.utils.cpp_extension import load

ext_module = load(name="torchRmsnormNaive",
                  extra_include_paths=["./"] ,
                  sources=["rmsnorm.cu", "rmsnorm_ops.cu"],
                  verbose=True)

M=1024
N=1024
data=torch.rand(M,N,dtype=torch.float32,device="cuda:0")

# M=1
# N=4
# data=torch.arange(1,5,dtype=torch.float32, device='cuda:0').reshape(1,4)

#warm up
torch.nn.functional.rms_norm(data,[N])

start=torch.cuda.Event(enable_timing=True)
end=torch.cuda.Event(enable_timing=True)

start.record()
toutput=torch.nn.functional.rms_norm(data,[N],eps=1e-5)
end.record()
torch.cuda.synchronize()
print(f"torch rmsnorm: {start.elapsed_time(end)}")

# print(toutput)

start.record()
ext_module.torchRmsnormNaive(data,toutput, M,N,32)
end.record()
torch.cuda.synchronize()
print(f"torchRmsnormNaive: {start.elapsed_time(end)}")

# print(toutput)

start.record()
ext_module.torchRmsnormShared(data,toutput, M,N,32)
end.record()
torch.cuda.synchronize()
print(f"torchRmsnormShared: {start.elapsed_time(end)}")

# print(toutput)

