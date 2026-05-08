#!/bin/bash
#SBATCH --job-name=p_8
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --partition=cpu
#SBATCH --output=results_p8_%j.log
#SBATCH --error=results_p8_%j.log

module load compiler/oneapi-2024/mpi 

P=8
D1=2
D2=4
T=10
SEED=1000

for run in {1..5}; do
    for M in 262144 1048576; do
        mpirun -np $P ./src $M $D1 $D2 $T $SEED
    done
done
