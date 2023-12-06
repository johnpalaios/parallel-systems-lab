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



