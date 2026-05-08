#!/bin/bash
#SBATCH --job-name=p64_nx120
#SBATCH --output=out_p64_nx120.txt
#SBATCH --error=err_p64_nx120.txt
#SBATCH --ntasks-per-node=32
#SBATCH -N 2
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --partition=cpu

module load compiler/oneapi-2024/mpi

for run in {1..5}; do
    mpirun -np 64 ./src 7 32 4 4 4 120 120 120 5 1000 2 500
done
