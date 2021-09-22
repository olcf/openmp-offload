CC  = gcc
CXX = g++
FC  = gfortran

CFLAGS   = -Ofast -fopenmp 
CXXFLAGS = -Ofast -fopenmp
FFLAGS	 = -Ofast -fopenmp

LIBS     = -Wl,-rpath=$(OLCF_GCC_ROOT)/lib64 -lm -foffload=nvptx-none="-Ofast -lm -latomic -misa=sm_35" 

COMPILER = gcc
