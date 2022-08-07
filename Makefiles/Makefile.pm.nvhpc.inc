#module load PrgEnv-nvidia
#module load cudatoolkit craype-accel-nvidia80

CC  = cc
CXX = CC
FC  = ftn

CFLAGS    = -fast -mp=gpu -Minfo=mp,accel
CXXFLAGS  = -fast -mp=gpu -Minfo=mp,accel 
FFLAGS    = -fast -mp=gpu -Minfo=mp,accel

COMPILER = nvhpc
