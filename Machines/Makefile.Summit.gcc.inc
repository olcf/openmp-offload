CC  = gcc
CXX = g++
FC  = gfortran

CFLAGS   = -O3 -fopenmp 
CXXFLAGS = -O3 -fopenmp
FFLAGS	 = -O3 -fopenmp

LIBS     = -Wl,-rpath=$(OLCF_GCC_ROOT)/lib64 -lm

COMPILER = gcc
