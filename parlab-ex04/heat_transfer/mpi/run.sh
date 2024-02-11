#!/bin/bash

## Give the Job a descriptive name
#PBS -N run_jacobi_mpi

## Output and error files
#PBS -o run.out
#PBS -e run.err

## How many machines should we get?
#PBS -l nodes=8:ppn=8

## Start
## Run make in the src folder (modify properly)

module load openmpi/1.8.3
cd /home/parallel/parlab03/heat_transfer/mpi
mpirun -np 4 --mca btl tcp,self ./jacobi_mpi 6144 6144 2 2
