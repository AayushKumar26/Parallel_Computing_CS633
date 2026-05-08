#!/bin/bash
#SBATCH --job-name=p48_nx240
#SBATCH --output=out_p48_nx240.txt
#SBATCH --error=err_p48_nx240.txt
#SBATCH --ntasks-per-node=48
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --partition=cpu

module load compiler/oneapi-2024/mpi

for run in {1..5}; do
    mpirun -np 48 ./src 7 48 6 4 2 240 240 240 5 1000 2 500
done
