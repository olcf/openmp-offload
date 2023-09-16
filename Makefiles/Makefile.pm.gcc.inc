#module load PrgEnv-gnu
#module use /global/cfs/cdirs/m1759/wwei/Modules/perlmutter/modulefiles
#module load gcc/13.1.0

CC  = gcc
CXX = g++
FC  = gfortran

CFLAGS   = -Ofast -fopenmp 
CXXFLAGS = -Ofast -fopenmp
FFLAGS	 = -Ofast -fopenmp

LIBS = -Wl,-rpath=$(GCC_ROOT)/lib64 -lm -foffload=nvptx-none="-Ofast -lm -latomic -misa=sm_80"

COMPILER = gcc
