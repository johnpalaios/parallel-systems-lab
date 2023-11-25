#!/bin/bash

cores=(1 2 4 6 8 16 32 64)
#cores=(1 2 4 6 8)
for number_of_cores in ${cores[@]}; do
        ## Give the Job a descriptive name
        #PBS -N run_kmeans

        ## Output and error files
        #PBS -o run_kmeans_naive.out
        #PBS -e run_kmeans_naive.err

        ## How many machines should we get?
        #PBS -l nodes=sandman:ppn=64

        ##How long should the job run for?
        #PBS -l walltime=00:02:00

        ## Start
        ## Run make in the src folder (modify properly)

        module load openmp
        cd /home/parallel/parlab03/parlab-ex02/kmeans_new
        export OMP_NUM_THREADS=$number_of_cores
        #export GOMP_CPU_AFFINITY=0-63
        ./kmeans_omp_reduction  -s 256 -n 1 -c 4 -l 10
done
