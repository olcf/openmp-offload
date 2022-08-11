#module use /global/cfs/cdirs/nstaff/cookbg/pe/modulefiles
#module load npe/22.08 cudatoolkit/11.5 craype-accel-nvidia80 PrgEnv-llvm llvm

CC  = mpicc
CXX = mpic++
#flang does not support OpenMP target offload yet
FC  = flang

CFLAGS    = -Ofast -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda
CXXFLAGS  = -Ofast -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda
FFLAGS    = -Ofast -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda

COMPILER = llvm
