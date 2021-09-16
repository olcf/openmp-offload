#module load nvhpc/21.7 

CC  = nvc
CXX = nvc++
FC  = nvfortran

CFLAGS    =-O3 -mp=gpu -gpu=cc70 -Minfo=accel -Minfo=mp
CXXFLAGS  =-O3 -mp=gpu -gpu=cc70 
FFLAGS    =-O3 -mp=gpu -gpu=cc70 

LIBS      =-Wl,-rpath=$(OLCF_NVHPC_ROOT)/compilers/lib \
           -Wl,-rpath=$(OLCF_NVHPC_ROOT)/math_libs/lib64 -lm

COMPILER = nvhpc
