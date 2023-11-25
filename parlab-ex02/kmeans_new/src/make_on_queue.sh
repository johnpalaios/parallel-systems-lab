#!/bin/bash

## Give the Job a descriptive name
#PBS -N make_kmeans

## Output and error files
#PBS -o make_kmeans.out
#PBS -e make_kmeans.err

## How many machines should we get? 
#PBS -l nodes=sandman:ppn=1

##How long should the job run for?
#PBS -l walltime=00:10:00

## Start 
## Run make in the src folder (modify properly)

module load openmp
cd /home/parallel/parlab03/parlab-ex02/kmeans_new
make
