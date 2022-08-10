#!/bin/bash
#SBATCH -N 1
#SBATCH -q debug
#SBATCH -C gpu
#SBATCH -t 10:00
#SBATCH -c 128
#SBATCH -G 1
#SBATCH -A ntrain4
#SBATCH --reservation=omp_day1

module load PrgEnv-nvidia 
module load cudatoolkit craype-accel-nvidia80 
cd /PATH/TO/TUTORIAL/openmp-offload/C/4-openmp-gpu-data
cc -fast -mp=gpu -Minfo=mp,accel -o jacobi.C.nvhpc.exe jacobi.c     # or just do: make
export OMP_NUM_THREADS=8    # for CPU OpenMP
./jacobi.C.nvhpc.exe <args>
