import torch
from torch.utils.cpp_extension import load

# ext_module = load(name="torchSoftmax",
#                   extra_include_paths=["./"] ,
#                   sources=["softmax.cu", "softmax_ops.cu"],
#                   verbose=True)

# M=1024*8
# N=50257
# data=torch.rand(M,N,dtype=torch.float32,device="cuda:0")

M=1
N=4
data=torch.arange(1,5,dtype=torch.float32, device='cuda:0').reshape(1,4)

#warm up
torch.nn.functional.rms_norm(data,[4])

start=torch.cuda.Event(enable_timing=True)
end=torch.cuda.Event(enable_timing=True)

start.record()
toutput=torch.nn.functional.rms_norm(data,[4],eps=1e-5)
end.record()
torch.cuda.synchronize()
print(f"torch rmsnorm: {start.elapsed_time(end)}")

print(toutput)

