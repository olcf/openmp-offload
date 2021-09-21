#module purge
#module load cgpu
#module load gcc/11.2.0

#module load  gcc/10-devel-omp_20201218

CC  = gcc
CXX = g++
FC  = gfortran

CFLAGS   = -O3 -fopenmp 
CXXFLAGS = -O3 -fopenmp
FFLAGS	 = -O3 -fopenmp

LIBS = -Wl,-rpath=$(GCC_ROOT)/lib64 -lm -foffload=nvptx-none="-Ofast -lm -latomic -misa=sm_35"

COMPILER = gcc