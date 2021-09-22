#module purge
#module load cgpu
#module load nvhpc/21.7 

CC  = nvc
CXX = nvc++
FC  = nvfortran

CFLAGS    = -fast -mp=gpu -gpu=cc70 -Minfo=mp,accel
CXXFLAGS  = -fast -mp=gpu -gpu=cc70 -Minfo=mp,accel 
FFLAGS    = -fast -mp=gpu -gpu=cc70 -Minfo=mp,accel

NVHPC_HOME=$(NVHPC)/Linux_x86_64/21.7
LIBS      = -Wl,-rpath=$(NVHPC_HOME)/compilers/lib \
            -Wl,-rpath=$(NVHPC_HOME)/math_libs/lib64 -lm

COMPILER = nvhpc
