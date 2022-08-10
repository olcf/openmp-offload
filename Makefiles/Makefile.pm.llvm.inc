#module use /global/cfs/cdirs/nstaff/cookbg/pe/modulefiles
#module load npe cudatoolkit/11.5 craype-accel-nvidia80 PrgEnv-llvm

CC  = clang
CXX = clang++
#flang is not available yet
FC  = flang

CFLAGS    = -Ofast -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_80
CXXFLAGS  = -Ofast -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_80
FFLAGS    = -Ofast -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_80

LIBS  = -Wl,-rpath=$(LLVM_ROOT)/lib -lm

COMPILER = llvm
