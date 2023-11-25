#!/bin/bash

## Give the Job a descriptive name
#PBS -N run_fw

## Output and error files
#PBS -o run_fw.out
#PBS -e run_fw.err

## How many machines should we get? 
#PBS -l nodes=sandman:ppn=64

##How long should the job run for?
#PBS -l walltime=00:10:00

## Start 
## Run make in the src folder (modify properly)

module load openmp
cd /home/parallel/parlab03/parlab-ex02/FW
export OMP_NUM_THREADS=32
export GOMP_CPU_AFFINITY=0-63
#./fw 1024
./fw_sr 4096 64
# ./fw_tiled <SIZE> <BSIZE>
