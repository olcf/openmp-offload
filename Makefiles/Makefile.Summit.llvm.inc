#module load llvm/14.0.0 cuda

CC  = clang
CXX = clang++
FC  = flang

CFLAGS    = -Ofast -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_70
CXXFLAGS  = -Ofast -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_70
FFLAGS    = -Ofast -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_70

LIBS  =-Wl,-rpath=$(OLCF_LLVM_ROOT)/lib -lm

COMPILER = llvm
