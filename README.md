# parallel-systems-lab

This repository contains the reports and source code written for the lab of the [Parallel Processing Systems course](https://www.ece.ntua.gr/en/undergraduate/courses/3257) of the school of Electrical and Computer Engineering at the National Technical University of Athens.

The lab consists of 4 different exercises (currently finished the first two) :
### Exercise 1 : Familiarization with the programming environment
Its main goal was to parallelize a serial version of [Conway's Game Of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) on a shared memory architecture using OpenMP's API .

- [The Report](https://github.com/johnpalaios/parallel-systems-lab/blob/main/parlab-ex01/report_ex01.pdf)

- More info on [/parlab-ex01](https://github.com/johnpalaios/parallel-systems-lab/tree/main/parlab-ex01)


### Exercise 2 : Algorithm Parallelization and Optimization in Shared Memory Architectures
The goal was to parallelize the [K-means Clustering Algorithm](https://en.wikipedia.org/wiki/K-means_clustering)  and the [Floyd-Warshall Algorithm](https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm)  on a shared memory architecture (NUMA node) using OpenMP's API.
- For the K-means clustering algorithm, we were assigned to develop two parallel version, the one having shared cluster arrays (between the threads) and updating them with atomic operations and the other having copied clusters for each thread and later reducing them to one final array.
- Benchmarked and compared 5 different Lock implementations on the K-means Clustering algorithm, having understood the differences in their implementations. 
- For the Floyd-Warshall algorithm, the goal was to parallelize its recursive version (more cache friendly in comparison to the iterative) using OpenMP's Tasks.
- Benchmarked and compared the serial and parallel version in a NUMA node and observed the different tradeoffs of this architecture.
- Benchmarked and compared 5 Concurrent Linked List implementations and commented on their differences in performance.

- [The Report](https://github.com/johnpalaios/parallel-systems-lab/blob/main/parlab-ex02/report_ex02.pdf)

- More info on [/parlab-ex02](https://github.com/johnpalaios/parallel-systems-lab/tree/main/parlab-ex02)

### Exerise 3 : Algorithm Parallelization and Optimization on GPUs
The goal was to parallelize 4 different versions of the K-means algorithm on a GPU using Nvidia's CUDA API.
- The first version is called Naive due to non-uniform memory accesses.
- The second version is called Transpose due to transposing two of the arrays in order to perform uniform memory accesses.
- The third version is called Shared due to placing the clusters array onto the GPU's shared memory for each thread block.
- The fourth version is called Full-Offload (All-GPU) due to avoiding CPU and GPU communication between the program's loop and and instead performing the entirety of the loops on the GPU (with minimal communication between them).
- Thoroughly benchmarked the 4 versions where we observed significant performance improvement to performing the algorithm on the solely on the CPU.
- As expected (and for reasons explained in the report), the best performing version is the Full-Offload.  
- Plotted the results of the benchmarkes and explained the reasons we saw performance differences between the 4 versions through exploring the GPU's and CUDA's internals.
  
- [The Report](https://github.com/johnpalaios/parallel-systems-lab/blob/main/parlab-ex03/report_ex03.pdf)
- More info on [/parlab-ex03](https://github.com/johnpalaios/parallel-systems-lab/tree/main/parlab-ex03)

  ### Exerise 4 : Algorithm Parallelization and Optimization on distributed memory architectures.
The goal was to parallelize 2 different algorithms, the K-means and the 2-d Heat Transfer, assuming a distributed memory architecture and using MPI.
- The K-means algorithm was parallelized by assigning each MPI process different objects and communicating between them in each iteration.
- The 2 dimensional Heat Transfer problem was solved using the Jacobi method (method for solving partial differential equations) where we assigned each MPI process different blocks of the 2d global block and performed communication between them when needed.
-  Highly suggest you take a look at the source code of the jacobi mpi implementation ([Link](https://github.com/johnpalaios/parallel-systems-lab/blob/main/parlab-ex04/heat_transfer/mpi/jacobi_mpi.c)).
-  We benchmarked each version for different configurations and number of MPI processes and we plotted the results accordingly.
  
- [The Report](https://github.com/johnpalaios/parallel-systems-lab/blob/main/parlab-ex04/report_ex04.pdf)
- More info on [/parlab-ex04](https://github.com/johnpalaios/parallel-systems-lab/tree/main/parlab-ex04)




