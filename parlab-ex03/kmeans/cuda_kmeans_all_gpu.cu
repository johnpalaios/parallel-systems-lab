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
	return threadIdx.x + blockIdx.x * blockDim.x; /* TODO: copy me from naive version... */
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

	/* TODO: Calculate the euclid_dist of elem=objectId of objects from elem=clusterId from clusters, but for column-base format!!! */
	for(i = 0; i < numCoords; i++) {
        ans += (objects[objectId+(numObjs*i)] - clusters[clusterId+(numClusters*i)]) * (objects[objectId+(numObjs*i)] - clusters[clusterId+(numClusters*i)]);
    }
    return(ans);

}

__global__ static
void find_nearest_cluster(int numCoords,
                          int numObjs,
                          int numClusters,
                          double *objects,           //  [numCoords][numObjs]
						  int *deviceNewClusterSize,    // [numClusters]
						  double *deviceNewClusters, 	   //  [numCoords][numClusters]               
                          double *deviceClusters,    //  [numCoords][numClusters] 
                          int *deviceMembership,          //  [numObjs]
                          double *devdelta)
{
     extern __shared__ double shmemClusters[];

	/* TODO: copy me from shared version... */
	int local_tid = threadIdx.x;
    int block_threads = blockDim.x;
    int clusters_size = numCoords*numClusters;
	
	int idx = local_tid;
    while(idx < clusters_size) {
        shmemClusters[idx] = deviceClusters[idx];
        idx += block_threads;
    }
    __syncthreads();
	
	/* Get the global ID of the thread. */
    int tid = get_tid(); 

	/* TODO: copy me from shared version... */
    if (tid < numObjs) {
    	int   index, i, j;
        double dist, min_dist;
		
        /* find the cluster id that has min distance to object */
        index = 0;
        min_dist = euclid_dist_2_transpose(numCoords, numObjs, numClusters, objects, shmemClusters, tid, 0);
        for (i=1; i<numClusters; i++) {
            dist = euclid_dist_2_transpose(numCoords, numObjs, numClusters, objects, shmemClusters, tid, i);
           /* no need square root */
            if (dist < min_dist) { /* find the min and its array index */
                min_dist = dist;
                index    = i;
            }
        }

        if (deviceMembership[tid] != index) {
            /* TODO: Maybe something is missing here... is this write safe? */
            //(*devdelta)+= 1.0;
            atomicAdd(devdelta, 1.0);
        }

        /* assign the deviceMembership to object objectId */
        deviceMembership[tid] = index;

    	/* TODO: additional steps for calculating new centroids in GPU? */
		
		// should reduce
		atomicAdd(&deviceNewClusterSize[index], 1);
		
		for(j=0; j<numCoords; j++) { 
			// also should reduce
			atomicAdd(&deviceNewClusters[j*numClusters+index], objects[j*numObjs+tid]);
		} 
		
    }
}

__global__ static
void update_centroids(int numCoords,
                          int numClusters,
                          int *devicenewClusterSize,           //  [numClusters]
                          double *devicenewClusters,    //  [numCoords][numClusters]
                          double *deviceClusters)    //  [numCoords][numClusters])
{

    /* TODO: additional steps for calculating new centroids in GPU? */
	int tid = get_tid();
	int i = tid % numClusters;
	int j = tid / numClusters;
	if(tid < numClusters*numCoords) {
		if(devicenewClusterSize[i] > 0) 
			deviceClusters[tid] = devicenewClusters[tid] / devicenewClusterSize[i];
		devicenewClusters[tid] = 0.0;
	}
	__syncthreads();
	if(tid < numClusters*numCoords && j == 0) 		
		devicenewClusterSize[i] = 0;	
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
	/*
    double  **dimObjects = NULL; //calloc_2d(...) -> [numCoords][numObjs]
    double  **dimClusters = NULL;  //calloc_2d(...) -> [numCoords][numClusters]
    double  **newClusters = NULL;  //calloc_2d(...) -> [numCoords][numClusters] */
	
	double  **dimObjects = (double**) calloc_2d(numCoords, numObjs, sizeof(double));
    double  **dimClusters = (double**) calloc_2d(numCoords, numClusters, sizeof(double));
    double  **newClusters = (double**) calloc_2d(numCoords, numClusters, sizeof(double));
	
    printf("\n|-----------Full-offload GPU Kmeans------------|\n\n");
    
    /* TODO: Copy me from transpose version*/
	for (i = 0; i < numCoords; i++) {
        for (j = 0; j < numObjs; j++) {
            dimObjects[i][j] = objects[j*numCoords+i];
        }
    }
    
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
    const unsigned int numClusterBlocks = numObjs/blockSize + 1; /* TODO: Calculate Grid size, e.g. number of blocks. */
	/*	Define the shared memory needed per block.
    	- BEWARE: We can overrun our shared memory here if there are too many
    	clusters or too many coordinates! 
    	- This can lead to occupancy problems or even inability to run. 
    	- Your exercise implementation is not requested to account for that (e.g. always assume deviceClusters fit in shmemClusters */
    const unsigned int clusterBlockSharedDataSize = numClusters*numCoords*sizeof(double); 

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
   	double gpu_time_1, gpu_start_2, gpu_time_2;
	double gpu_total_1 = 0.0, gpu_total_2 = 0.0;
 
    do {
        timing_internal = wtime(); 
        checkCuda(cudaMemset(dev_delta_ptr, 0, sizeof(double)));          
		//printf("Launching find_nearest_cluster Kernel with grid_size = %d, block_size = %d, shared_mem = %d KB\n", numClusterBlocks, numThreadsPerClusterBlock, clusterBlockSharedDataSize/1000);
        /* TODO: change invocation if extra parameters needed */ 
        find_nearest_cluster
            <<< numClusterBlocks, numThreadsPerClusterBlock, clusterBlockSharedDataSize >>>
            (numCoords, numObjs, numClusters, 
             deviceObjects, devicenewClusterSize, devicenewClusters, deviceClusters, deviceMembership, dev_delta_ptr);
        

        cudaDeviceSynchronize(); checkLastCudaError();
		//printf("Kernels complete for itter %d, updating data in CPU\n", loop);
    	gpu_time_1 = wtime() - timing_internal;
		printf("\t\tGPU part 1 : %lf ms\n", 1000*gpu_time_1);
		gpu_total_1 += gpu_time_1;

    	/* TODO: Copy dev_delta_ptr to &delta
        checkCuda(cudaMemcpy(...)); */
		checkCuda(cudaMemcpy(&delta, dev_delta_ptr, sizeof(double), cudaMemcpyDeviceToHost));

     	const unsigned int update_centroids_block_sz = (numCoords* numClusters > blockSize) ? blockSize: numCoords* numClusters;  /* TODO: can use different blocksize here if deemed better */
     	const unsigned int update_centroids_dim_sz =  numCoords*numClusters/blockSize + 1; /* TODO: calculate dim for "update_centroids" and fire it */
	 	gpu_start_2 = wtime();
     	update_centroids<<< update_centroids_dim_sz, update_centroids_block_sz, 0 >>>
            (numCoords, numClusters, devicenewClusterSize, devicenewClusters, deviceClusters);    
        cudaDeviceSynchronize(); checkLastCudaError();
        	
		gpu_time_2 = wtime() - gpu_start_2;
        printf("\t\tGPU part 2 : %lf ms\n", 1000*gpu_time_2);
        gpu_total_2 += gpu_time_2;
		   
        delta /= numObjs;
       	//printf("delta is %f - ", delta);
        loop++;
        //printf("completed loop %d\n", loop);
		timing_internal = wtime() - timing_internal; 
		if ( timing_internal < timer_min) timer_min = timing_internal; 
		if ( timing_internal > timer_max) timer_max = timing_internal; 
	} while (delta > threshold && loop < loop_threshold);
    
	printf("\tGPU 1 total : %lf ms\n\tGPU 2 total : %lf ms\n",1000*gpu_total_1, 1000*gpu_total_2);
		          	
    checkCuda(cudaMemcpy(membership, deviceMembership,
                 numObjs*sizeof(int), cudaMemcpyDeviceToHost));     
    checkCuda(cudaMemcpy(dimClusters[0], deviceClusters,
                 numClusters*numCoords*sizeof(double), cudaMemcpyDeviceToHost));  
                                   
	for (i=0; i<numClusters; i++) {
		for (j=0; j<numCoords; j++) {
		    clusters[i*numCoords + j] = dimClusters[j][i];
		}
	}
	
    timing = wtime() - timing;
    printf("nloops = %d  : total = %lf ms\n\t-> t_loop_avg = %lf ms\n\t-> t_loop_min = %lf ms\n\t-> t_loop_max = %lf ms\n\n|-------------------------------------------|\n", 
    	loop, 1000*timing, 1000*timing/loop, 1000*timer_min, 1000*timer_max);

	char outfile_name[1024] = {0}; 
	sprintf(outfile_name, "Execution_logs/silver1-V100_Sz-%lu_Coo-%d_Cl-%d.csv", numObjs*numCoords*sizeof(double)/(1024*1024), numCoords, numClusters);
	FILE* fp = fopen(outfile_name, "a+");
	if(!fp) error("Filename %s did not open succesfully, no logging performed\n", outfile_name); 
	fprintf(fp, "%s,%d,%lf,%lf,%lf\n", "All_GPU", blockSize, timing/loop, timer_min, timer_max);
	fclose(fp); 
	
    checkCuda(cudaFree(deviceObjects));
    checkCuda(cudaFree(deviceClusters));
    checkCuda(cudaFree(devicenewClusters));
    checkCuda(cudaFree(devicenewClusterSize));
    checkCuda(cudaFree(deviceMembership));

    return;
}

