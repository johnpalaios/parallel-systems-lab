#!/bin/bash

cores=(1 2 4 8 16 32 64)
board_sizes=(1024 2048 4096)
for number_of_cores in ${cores[@]}; do
  for board_size in ${board_sizes[@]}; do
        #PBS -N run_omp_Game_Of_Life_${number_of_cores}_cores_${board_size}

        ## Output and error files
        #PBS -o fw_log_o.out
        #PBS -e fw_log_e.err

        #PBS -l nodes=sandman:ppn=64

        ##How long should the job run for?
        #PBS -l walltime=00:5:00

        ## Start
        ## Run make in the src folder (modify properly)

        module load openmp
        cd /home/parallel/parlab03/parlab-ex02/FW
        export OMP_NUM_THREADS=$number_of_cores
	export GOMP_CPU_AFFINITY=0-63        
        ./fw_sr $board_size 64
  done
done
