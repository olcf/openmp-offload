#module load nvhpc/21.7 

CC  = nvc
CXX = nvc++
FC  = nvfortran

CFLAGS    = -O3 -mp=gpu -gpu=cc70 -Minfo=accel -Minfo=mp
CXXFLAGS  = -O3 -mp=gpu -gpu=cc70 
FFLAGS    = -O3 -mp=gpu -gpu=cc70 

NVHPC_HOME=$(NVHPC)/Linux_x86_64/21.7
LIBS      = -Wl,-rpath=$(NVHPC_HOME)/compilers/lib \
            -Wl,-rpath=$(NVHPC_HOME)/math_libs/lib64 -lm

COMPILER = nvhpc
