#!/bin/bash
#SBATCH -A ABC123
#SBATCH -J omptutorial
#SBATCH -o %x-%j.out
#SBATCH -t 00:10:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --reservation=openmp

ulimit -s 300000   # Needed for implicit mapping example

#For CCE
module reset
module load PrgEnv-cray cpe/23.09 cce/16.0.1 rocm craype-accel-amd-gfx90a

# For rocm
# module load PrgEnv-amd amd

# For gcc
# module use /sw/crusher/ums/compilers/modulefiles
# module load gcc/13.2.1-dev-latest

cd /PATH/TO/TUTORIAL/openmp-offload/C/1-openmp-cpu/
export OMP_NUM_THREADS=7    # for CPU OpenMP
./jacobi.C.cce.exe <args>
