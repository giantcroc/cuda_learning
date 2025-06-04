CUDA_DIR := /usr/local/cuda
CXX := $(CUDA_DIR)/bin/nvcc
CXXFLAGS := -std=c++11 -O3 -I$(CUDA_DIR)/include
LDFLAGS := -L$(CUDA_DIR)/lib64 -lcuda -lcudart 

FILES := softmax

.PHONY: all 
all: $(FILES)

%: %.cu
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $< -o $@