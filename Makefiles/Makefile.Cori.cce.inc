#module purge
#module load cgpu
#module load PrgEnv-cray
#module load cdt
#module load craype-x86-skylake
#module switch cce cce/10.0.3
#module load cuda/10.1.243
#module unload cray-libsci
#module list

#or just do: "source Cori_setup_cce"

CC=cc
CXX=CC
FC=ftn

CFLAGS=-Ofast -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target=nvptx64 -march=sm_70
CXXFLAGS=-Ofast -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target=nvptx64 -march=sm_70
FFLAGS=-O3 -homp -h noacc -haccel=nvidia70

COMPILER = cce
