# module use /sw/crusher/ums/compilers/modulefiles
# module load  module load gcc/13.2.1-dev-latest

$(info Make sure the following cammends were executed first: )
$(info   module use /sw/crusher/ums/compilers/modulefiles)
$(info   module load gcc/13.2.1-dev-latest )

CC  = gcc
CXX = g++
FC  = gfortran

CFLAGS    = -Ofast -fopenmp -foffload="-march=gfx90a -lm" 
CXXFLAGS  = -Ofast -fopenmp -foffload="-march=gfx90a -lm"
FFLAGS    = -Ofast -fopenmp -foffload="-march=gfx90a -lm"

LIBS  = -lm

COMPILER = gcc
