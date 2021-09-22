#module purge
#module load cgpu
#module load PrgEnv-llvm/13_rc3

CC  = clang
CXX = clang++
FC  = flang

CFLAGS    = -Ofast -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_70
CXXFLAGS  = -Ofast -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_70
FFLAGS    = -Ofast -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_70

LIBS  = -Wl,-rpath=$(LLVM_ROOT)/lib -lm

COMPILER = llvm
