# module load PrgEnv-amd  amd 

$(info Make sure the following modules are loaded: PrgEnv-amd  amd )

CC  = cc
CXX = CC
FC  = ftn

CFLAGS    = -Ofast -fopenmp 
CXXFLAGS  = -Ofast -fopenmp 
FFLAGS    = -Ofast -fopenmp 

LIBS  = -lm

COMPILER = rocm
