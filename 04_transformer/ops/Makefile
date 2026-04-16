CUDA_DIR := /usr/local/cuda
CXX := $(CUDA_DIR)/bin/nvcc
CXXFLAGS := -std=c++11 -O3 -I$(CUDA_DIR)/include
LDFLAGS := -L$(CUDA_DIR)/lib64 -lcuda -lcudart -lcublas -lcublasLt 

FILES := softmax_forward

.PHONY: all 
all: $(FILES)

%: %.cu
	LD_LIBRARY_PATH=$(CUDA_DIR)/lib64 $(CXX) $(CXXFLAGS) $(LDFLAGS) $< -o $@