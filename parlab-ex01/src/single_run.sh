number_of_cores=2
board_size=1064
#PBS -N run_omp_Game_Of_Life_${number_of_cores}_cores_${board_size}

## Output and error files
#PBS -o run_omp_Game_Of_Life_${number_of_cores}_${board_size}.out
#PBS -e run_omp_Game_Of_Life_${number_of_cores}_${board_size}.err

## How many machines should we get? 
#PBS -l nodes=1:ppn=$number_of_cores

##How long should the job run for?
#PBS -l walltime=00:10:00

## Start 
## Run make in the src folder (modify properly)

module load openmp
cd /home/parallel/parlab03/parlab-ex01
export OMP_NUM_THREADS=$number_of_cores
echo "$OMP_NUM_THREADS"
./Game_Of_Life 1024 1000         
