CC  = gcc
CXX = g++
FC  = gfortran

CFLAGS   = -O3 -fopenmp 
CXXFLAGS = -O3 -fopenmp
FFLAGS	 = -O3 -fopenmp

LIBS     = -Wl,-rpath=$(OLCF_GCC_ROOT)/lib64 -lm -foffload=nvptx-none="-Ofast -lm -latomic -misa=sm_35" 

COMPILER = gcc
