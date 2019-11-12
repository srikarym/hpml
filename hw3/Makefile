all: ./lab3

CUDADIR :=  /share/apps/cuda/9.0.176
CUDNNDIR := /share/apps/cudnn/9.0v7.0.5

CPPFLAGS := -O2 --std=c++11 -I$(CUDADIR)/include -I$(CUDNNDIR)/include
LDFLAGS := -L$(CUDADIR)/lib -L$(CUDNNDIR)/lib64
LDLIBS := -lcublas -lcudnn

NVCC := nvcc
CC := $(NVCC)

%: %.cu
	@$(NVCC) $(CPPFLAGS) $(LDFLAGS) $< -o $@ $(LDLIBS)

.PHONY: clean
clean:
	@rm  -f ./lab3
