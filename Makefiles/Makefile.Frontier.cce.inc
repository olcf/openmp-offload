# module load PrgEnv-cray cce craype-accel-amd-gfx90a rocm
$(info Make sure the following modules are loaded: PrgEnv-cray cce craype-accel-amd-gfx90a )

CC  = cc
CXX = CC
FC  = ftn

CFLAGS    = -Ofast -fopenmp 
CXXFLAGS  = -Ofast -fopenmp 
FFLAGS    = -Ofast -fopenmp 

LIBS  = -lm

COMPILER = cce
