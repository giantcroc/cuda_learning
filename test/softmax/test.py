import torch
from torch.utils.cpp_extension import load

ext_module = load(name="torchSoftmax",
                  extra_include_paths=["./"] ,
                  sources=["softmax.cu", "softmax_ops.cu"],
                  verbose=True)

M=32
N=256
data=torch.rand(M,N,dtype=torch.float32,device="cuda:0")

#warm up
torch.nn.functional.softmax(data,dim=-1)

start=torch.cuda.Event(enable_timing=True)
end=torch.cuda.Event(enable_timing=True)

start.record()
toutput=torch.nn.functional.softmax(data,dim=-1)
end.record()
torch.cuda.synchronize()
print(f"torch softmax: {start.elapsed_time(end)}")

# print(toutput)
output=data.clone()

start.record()
ext_module.torchSoftmaxWarp(data,output,M,N,32)
end.record()
torch.cuda.synchronize()
print(f"torchSoftmaxWarp: {start.elapsed_time(end)}")
# print(output1)

print(torch.allclose(output,toutput,1e-5,1e-5))

# output2=data.clone()

start.record()
ext_module.torchSoftmaxShared(data,output,M,N,32)
end.record()
torch.cuda.synchronize()
print(f"SoftmaxShared: {start.elapsed_time(end)}")

# print(output2)

print(torch.allclose(output,toutput,1e-5,1e-5))

# output3=data.clone()

start.record()
ext_module.torchSoftmax7(data,output,M,N,32)
end.record()
torch.cuda.synchronize()
print(f"torchSoftmax7: {start.elapsed_time(end)}")

# print(output2)

print(torch.allclose(output,toutput,1e-5,1e-5))