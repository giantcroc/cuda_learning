import torch
from torch.utils.cpp_extension import load

ext_module = load(name="torchSoftmaxNaive",
                  extra_include_paths=["./"] ,
                  sources=["softmax.cu", "softmax_ops.cu"],
                  verbose=True)

data=torch.arange(1,5,dtype=torch.float32,device="cuda:0").reshape(1,4)
output=data.clone()

ext_module.torchSoftmaxNaive(data,output,1,4,32)

print(output)



data=torch.nn.functional.softmax(data,dim=-1)

print(data)

