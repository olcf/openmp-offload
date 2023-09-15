#module load PrgEnv-llvm

CC  = clang
CXX = clang++
#flang does not support OpenMP target offload yet
FC  = flang

CFLAGS    = -Ofast -fopenmp -fopenmp-targets=nvptx64
CXXFLAGS  = -Ofast -fopenmp -fopenmp-targets=nvptx64
FFLAGS    = -Ofast -fopenmp -fopenmp-targets=nvptx64

COMPILER = llvm
