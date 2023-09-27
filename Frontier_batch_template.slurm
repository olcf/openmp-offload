#!/bin/bash
#SBATCH -A ABC123
#SBATCH -J omptutorial
#SBATCH -o %x-%j.out
#SBATCH -t 00:10:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --reservation=openmp

#For CCE
module load PrgEnv-cray rocm craype-accel-amd-gfx90a

# For rocm
# module load PrgEnv-amd amd

# For gcc
# module use /sw/crusher/ums/compilers/modulefiles
# module load gcc/13.2.1-dev-latest

cd /PATH/TO/TUTORIAL/openmp-offload/C/1-openmp-cpu/
export OMP_NUM_THREADS=7    # for CPU OpenMP
./jacobi.C.cce.exe <args>
