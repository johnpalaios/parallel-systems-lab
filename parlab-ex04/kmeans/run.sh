#!/bin/bash

## Give the Job a descriptive name
#PBS -N run_kmeans_mpi

## Output and error files
#PBS -o run_kmeans.out
#PBS -e run_kmeans.err

## How many machines should we get? 
#PBS -l nodes=8:ppn=8

## Start 
## Run make in the src folder (modify properly)

mpi_procs=(1 2 4 8 16 32 64)
for num_of_mpi_procs in ${mpi_procs[@]}; do
	module load openmpi/1.8.3
	cd /home/parallel/parlab03/parlab-ex04/kmeans
	export GOMP_CPU_AFFINITY=0-63
	mpirun -np $num_of_mpi_procs --mca btl tcp,self ./kmeans_mpi -s 256 -n 16 -c 16 -l 10
done
