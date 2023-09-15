#module load PrgEnv-cray

CC  = cc
CXX = CC
FC  = ftn 

CFLAGS    = -Ofast -fopenmp 
CXXFLAGS  = -Ofast -fopenmp
FFLAGS    = -O3 -homp -hnoacc

COMPILER = cce
