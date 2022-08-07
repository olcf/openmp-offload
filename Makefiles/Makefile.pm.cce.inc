#module load PrgEnv-cray
#module load cudatoollkit craype-accel-nvidia80

CC  = cc
CXX = CC
FC  = ftn 

CFLAGS    = -Ofast -fopenmp 
CXXFLAGS  = -Ofast -fopenmp
FFLAGS    = -O3 -homp -hnoacc

COMPILER = cce
