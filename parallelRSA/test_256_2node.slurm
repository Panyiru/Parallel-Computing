#!/bin/bash
#SBATCH -p physical
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:50:00

module load GCC/4.9.2
module load OpenMPI/1.10.2-intel-2017.u2
module load GMP/5.1.3-GCC-4.9.2
make
export OMP_NUM_THREADS=8
time mpiexec ./main 20K-keys.txt 20000
