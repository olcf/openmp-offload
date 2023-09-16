#module load PrgEnv-nvidia

CC  = cc
CXX = CC
FC  = ftn

CFLAGS    = -fast -mp=gpu -Minfo=mp,accel
CXXFLAGS  = -fast -mp=gpu -Minfo=mp,accel 
FFLAGS    = -fast -mp=gpu -Minfo=mp,accel

COMPILER = nvhpc
