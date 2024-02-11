#include <stdio.h>
#include <stdlib.h>

#include "kmeans.h"
#include "alloc.h"
#include "error.h"

#ifdef __CUDACC__
inline void checkCuda(cudaError_t e) {
    if (e != cudaSuccess) {
        // cudaGetErrorString() isn't always very helpful. Look up the error
        // number in the cudaError enum in driver_types.h in the CUDA includes
        // directory for a better explanation.
        error("CUDA Error %d: %s\n", e, cudaGetErrorString(e));
    }
}

inline void checkLastCudaError() {
    checkCuda(cudaGetLastError());
}
#endif

__device__ int get_tid(){
	return 0; /* TODO: Copy from full-offload */
}

/* square of Euclid distance between two multi-dimensional points using column-base format */
__host__ __device__ inline static
double euclid_dist_2_transpose(int numCoords,
                    int    numObjs,
                    int    numClusters,
                    double *objects,     // [numCoords][numObjs]
                    double *clusters,    // [numCoords][numClusters]
                    int    objectId,
                    int    clusterId)
{
    int i;
    double ans=0.0;

    /* TODO: Copy from full-offload */

    return(ans);
}

/*----< find_nearest_cluster() >---------------------------------------------*/
__global__ static
void find_nearest_cluster(int numCoords,
                          int numObjs,
                          int numClusters,
                          double *deviceobjects,           //  [numCoords][numObjs]
                          int *devicenewClusterSize,           //  [numClusters]
                          double *devicenewClusters,    //  [numCoords][numClusters]
                          double *deviceClusters,    //  [numCoords][numClusters]
                          int *deviceMembership,          //  [numObjs]
                          double *devdelta)
{
    extern __shared__ double shmem_total[]; 
    double* shmemClusters = shmem_total;
    double* delta_reduce_buff = shmem_total + numClusters*numCoords;
	/* TODO: copy me from shared version... */

	/* Get the global ID of the thread. */
    int tid = get_tid(); 

	/* TODO: copy me from shared version... */
    if (1) {

		/* TODO: copy me from shared version... */

		/* TODO: Replacing (*devdelta)+= 1.0; with reduction: 
			- each thread updates the single element of delta_reduce_buff 
			corresponding to its local id (threadIdx.x) -> 1.0 if membership changes, otherwise 0. 
			- Then, ensuring delta_reduce_buff is fully updated, its containts must be summed in delta_reduce_buff[0]
			either by one thread (lower perf) or with a tree-based reduction (similar to dot reduction example in slides) 
			- Finally, delta_reduce_buff[0] (local value in block) must be added to devdelta (global delta value), ensuring write dependencies! 
		*/    
    	/* TODO: additional steps for calculating new centroids in GPU? */
    }
    
}

__global__ static
void update_centroids(int numCoords,
                          int numClusters,
                          int *devicenewClusterSize,           //  [numClusters]
                          double *devicenewClusters,    //  [numCoords][numClusters]
                          double *deviceClusters)    //  [numCoords][numClusters])
{
    /* TODO: Copy from full-offload */
}

//
//  ----------------------------------------
//  DATA LAYOUT
//
//  objects         [numObjs][numCoords]
//  clusters        [numClusters][numCoords]
//  dimObjects      [numCoords][numObjs]
//  dimClusters     [numCoords][numClusters]
//  newClusters     [numCoords][numClusters]
//  deviceObjects   [numCoords][numObjs]
//  deviceClusters  [numCoords][numClusters]
//  ----------------------------------------
//
/* return an array of cluster centers of size [numClusters][numCoords]       */            
void kmeans_gpu(	double *objects,      /* in: [numObjs][numCoords] */
		               	int     numCoords,    /* no. features */
		               	int     numObjs,      /* no. objects */
		               	int     numClusters,  /* no. clusters */
		               	double   threshold,    /* % objects change membership */
		               	long    loop_threshold,   /* maximum number of iterations */
		               	int    *membership,   /* out: [numObjs] */
						double * clusters,   /* out: [numClusters][numCoords] */
						int blockSize)  
{
    double timing = wtime(), timing_internal, timer_min = 1e42, timer_max = 0; 
	int    loop_iterations = 0; 
    int      i, j, index, loop=0;
    double  delta = 0, *dev_delta_ptr;          /* % of objects change their clusters */
    /* TODO: Copy me from transpose version*/
    double  **dimObjects = NULL; //calloc_2d(...) -> [numCoords][numObjs]
    double  **dimClusters = NULL;  //calloc_2d(...) -> [numCoords][numClusters]
    double  **newClusters = NULL;  //calloc_2d(...) -> [numCoords][numClusters]

    printf("\n|-----------Full-offload Delta Reduction GPU Kmeans------------|\n\n");
    
    /* TODO: Copy me from transpose version*/
	for(;;);
    
    double *deviceObjects;
    double *deviceClusters, *devicenewClusters;
    int *deviceMembership;
    int *devicenewClusterSize; /* [numClusters]: no. objects assigned in each new cluster */

    /* pick first numClusters elements of objects[] as initial cluster centers*/
    for (i = 0; i < numCoords; i++) {
        for (j = 0; j < numClusters; j++) {
            dimClusters[i][j] = dimObjects[i][j];
        }
    }
	
    /* initialize membership[] */
    for (i=0; i<numObjs; i++) membership[i] = -1;
    
    timing = wtime() - timing;
    printf("t_alloc: %lf ms\n\n", 1000*timing);
    timing = wtime(); 
    const unsigned int numThreadsPerClusterBlock = (numObjs > blockSize)? blockSize: numObjs;
    const unsigned int numClusterBlocks = -1; /* TODO: Calculate Grid size, e.g. number of blocks. */

	/*	Define the shared memory needed per block. 
		- BEWARE: Also add extra shmem for delta buffer.
    	- BEWARE: We can overrun our shared memory here if there are too many
    	clusters or too many coordinates! 
    	- This can lead to occupancy problems or even inability to run. 
    	- Your exercise implementation is not requested to account for that (e.g. always assume deviceClusters fit in shmemClusters */
    const unsigned int clusterBlockSharedDataSize = -1;

    cudaDeviceProp deviceProp;
    int deviceNum;
    cudaGetDevice(&deviceNum);
    cudaGetDeviceProperties(&deviceProp, deviceNum);

    if (clusterBlockSharedDataSize > deviceProp.sharedMemPerBlock) {
        error("Your CUDA hardware has insufficient block shared memory to hold all cluster centroids\n");
    }
           
    checkCuda(cudaMalloc(&deviceObjects, numObjs*numCoords*sizeof(double)));
    checkCuda(cudaMalloc(&deviceClusters, numClusters*numCoords*sizeof(double)));
    checkCuda(cudaMalloc(&devicenewClusters, numClusters*numCoords*sizeof(double)));
    checkCuda(cudaMalloc(&devicenewClusterSize, numClusters*sizeof(int)));
    checkCuda(cudaMalloc(&deviceMembership, numObjs*sizeof(int)));
    checkCuda(cudaMalloc(&dev_delta_ptr, sizeof(double)));
    
    timing = wtime() - timing;
    printf("t_alloc_gpu: %lf ms\n\n", 1000*timing);
    timing = wtime(); 
    
    checkCuda(cudaMemcpy(deviceObjects, dimObjects[0],
              numObjs*numCoords*sizeof(double), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(deviceMembership, membership,
              numObjs*sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(deviceClusters, dimClusters[0],
                  numClusters*numCoords*sizeof(double), cudaMemcpyHostToDevice));
    checkCuda(cudaMemset(devicenewClusterSize, 0, numClusters*sizeof(int)));
    free(dimObjects[0]);        
    timing = wtime() - timing;
    printf("t_get_gpu: %lf ms\n\n", 1000*timing);
    timing = wtime();  
    
    do {
		timing_internal = wtime(); 
		checkCuda(cudaMemset(dev_delta_ptr, 0, sizeof(double)));          
		//printf("Launching find_nearest_cluster Kernel with grid_size = %d, block_size = %d, shared_mem = %d KB\n", numClusterBlocks, numThreadsPerClusterBlock, clusterBlockSharedDataSize/1000);
		/* TODO: change invocation if extra parameters needed 
		find_nearest_cluster
			<<< numClusterBlocks, numThreadsPerClusterBlock, clusterBlockSharedDataSize >>>
			(numCoords, numObjs, numClusters,
			 deviceObjects, devicenewClusterSize, devicenewClusters, deviceClusters, deviceMembership, dev_delta_ptr);
		*/

		cudaDeviceSynchronize(); checkLastCudaError();
		//printf("Kernels complete for itter %d, updating data in CPU\n", loop);

		/* TODO: Copy dev_delta_ptr to &delta
		checkCuda(cudaMemcpy(...)); */

		const unsigned int update_centroids_block_sz = (numCoords* numClusters > blockSize) ? blockSize: numCoords* numClusters;  /* TODO: can use different blocksize here if deemed better */
		const unsigned int update_centroids_dim_sz =  -1; /* TODO: calculate dim for "update_centroids" and fire it 
		update_centroids<<< update_centroids_dim_sz, update_centroids_block_sz, 0 >>>
			(numCoords, numClusters, devicenewClusterSize, devicenewClusters, deviceClusters);  */  
		cudaDeviceSynchronize(); checkLastCudaError();   
				       
		delta /= numObjs;
		//printf("delta is %f - ", delta);
		loop++; 
		//printf("completed loop %d\n", loop);
		timing_internal = wtime() - timing_internal; 
		if ( timing_internal < timer_min) timer_min = timing_internal; 
		if ( timing_internal > timer_max) timer_max = timing_internal; 
	} while (delta > threshold && loop < loop_threshold);
    
                 	
    checkCuda(cudaMemcpy(membership, deviceMembership,
                 numObjs*sizeof(int), cudaMemcpyDeviceToHost));     
    checkCuda(cudaMemcpy(dimClusters[0], deviceClusters,
                 numClusters*numCoords*sizeof(double), cudaMemcpyDeviceToHost));  
                                   
	for (i=0; i<numClusters; i++) {
	    //if (newClusterSize[i] > 0) {
		for (j=0; j<numCoords; j++) {
		    clusters[i*numCoords + j] = dimClusters[j][i];
		}
	    //}
	}
	
    timing = wtime() - timing;
    printf("nloops = %d  : total = %lf ms\n\t-> t_loop_avg = %lf ms\n\t-> t_loop_min = %lf ms\n\t-> t_loop_max = %lf ms\n\n|-------------------------------------------|\n", 
    	loop, 1000*timing, 1000*timing/loop, 1000*timer_min, 1000*timer_max);

	char outfile_name[1024] = {0}; 
	sprintf(outfile_name, "Execution_logs/silver1-V100_Sz-%lu_Coo-%d_Cl-%d.csv", numObjs*numCoords*sizeof(double)/(1024*1024), numCoords, numClusters);
	FILE* fp = fopen(outfile_name, "a+");
	if(!fp) error("Filename %s did not open succesfully, no logging performed\n", outfile_name); 
	fprintf(fp, "%s,%d,%lf,%lf,%lf\n", "All_GPU_Delta_Reduction", blockSize, timing/loop, timer_min, timer_max);
	fclose(fp); 
	
    checkCuda(cudaFree(deviceObjects));
    checkCuda(cudaFree(deviceClusters));
    checkCuda(cudaFree(devicenewClusters));
    checkCuda(cudaFree(devicenewClusterSize));
    checkCuda(cudaFree(deviceMembership));

    return;
}

