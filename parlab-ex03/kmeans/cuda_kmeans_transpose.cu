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
	/* NAIVE :
	for(i = 0; i < numCoords; i++) {
    	ans += (objects[objectId*numCoords+i] - clusters[clusterId*numCoords+i]) * (objects[objectId*numCoords+i] - clusters[clusterId*numCoords+i]);
    }	
	*/

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
                          double *deviceClusters,    //  [numCoords][numClusters]
                          int *membership,          //  [numObjs]
                          double *devdelta)
{
	/* TODO: copy me from naive version... */
	/* Get the global ID of the thread. */
    int tid = get_tid();
    /*
	int local_tid = threadIdx.x;
    extern __shared__ double partial_devdelta[];
    partial_devdelta[local_tid] = 0;
	*/
    /* TODO: Maybe something is missing here... should all threads run this? */
    // threads with tid > numObjs ??
    if (tid < numObjs) {
        int   index, i;
        double dist, min_dist;

        /* find the cluster id that has min distance to object */
        index = 0;
        /* TODO: call min_dist = euclid_dist_2(...) with correct objectId/clusterId */
        // objectId is tid maybe ?
        min_dist = euclid_dist_2_transpose(numCoords, numObjs, numClusters, objects, deviceClusters, tid, 0);
        for (i=1; i<numClusters; i++) {
        	/* TODO: call dist = euclid_dist_2(...) with correct objectId/clusterId */
            // again objectId is tid ?
            dist = euclid_dist_2_transpose(numCoords, numObjs, numClusters, objects, deviceClusters, tid, i);
            /* no need square root */
            if (dist < min_dist) { /* find the min and its array index */
            	min_dist = dist;
                index    = i;
            }
        }

        if (membership[tid] != index) {
            /* TODO: Maybe something is missing here... is this write safe? */
	        // reduce cuda variable..how??
            // we will use shared memory for the reduction
            //partial_devdelta[local_tid] += 1.0;
            //(*devdelta)+= 1.0;
	    	atomicAdd(devdelta, 1.0);
        }
        /* assign the deviceMembership to object objectId */
        membership[tid] = index;
    }
	/*
    __syncthreads();
    int i = blockDim.x / 2;
    while(i != 0) {
        if(local_tid < i) {
            partial_devdelta[local_tid] += partial_devdelta[local_tid + i];
        }
        __syncthreads();
        i /= 2;
    }
    if(local_tid == 0) {
        atomicAdd(devdelta, partial_devdelta[0]);
    }	
	*/
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
    int     *newClusterSize; /* [numClusters]: no. objects assigned in each
                                new cluster */
    double  delta = 0, *dev_delta_ptr;          /* % of objects change their clusters */
    
    /* TODO: Transpose dims */
    double  **dimObjects = (double**) calloc_2d(numCoords, numObjs, sizeof(double)); //calloc_2d(...) -> [numCoords][numObjs]
    double  **dimClusters = (double**) calloc_2d(numCoords, numClusters, sizeof(double));  //calloc_2d(...) -> [numCoords][numClusters]
    double  **newClusters = (double**) calloc_2d(numCoords, numClusters, sizeof(double));  //calloc_2d(...) -> [numCoords][numClusters]
    
    double *deviceObjects;
    double *deviceClusters;
    int *deviceMembership;

    printf("\n|-----------Transpose GPU Kmeans------------|\n\n");
    
    //  TODO: Copy objects given in [numObjs][numCoords] layout to new
    //  [numCoords][numObjs] layout
    for (i = 0; i < numCoords; i++) {
        for (j = 0; j < numObjs; j++) {
            dimObjects[i][j] = objects[j*numCoords+i];
        }
    }
	
    /* pick first numClusters elements of objects[] as initial cluster centers*/
    for (i = 0; i < numCoords; i++) {
        for (j = 0; j < numClusters; j++) {
            dimClusters[i][j] = dimObjects[i][j];
        }
    }
	
    /* initialize membership[] */
    for (i=0; i<numObjs; i++) membership[i] = -1;

    /* need to initialize newClusterSize and newClusters[0] to all 0 */
    newClusterSize = (int*) calloc(numClusters, sizeof(int));
    assert(newClusterSize != NULL); 
    
    timing = wtime() - timing;
    printf("t_alloc: %lf ms\n\n", 1000*timing);
    timing = wtime(); 

    const unsigned int numThreadsPerClusterBlock = (numObjs > blockSize)? blockSize: numObjs;
    const unsigned int numClusterBlocks = numObjs/blockSize + 1; /* TODO: Calculate Grid size, e.g. number of blocks. */
    //const unsigned int clusterBlockSharedDataSize = numThreadsPerClusterBlock*sizeof(double);;
    const unsigned int clusterBlockSharedDataSize = 0;
 
    checkCuda(cudaMalloc(&deviceObjects, numObjs*numCoords*sizeof(double)));
    checkCuda(cudaMalloc(&deviceClusters, numClusters*numCoords*sizeof(double)));
    checkCuda(cudaMalloc(&deviceMembership, numObjs*sizeof(int)));
    checkCuda(cudaMalloc(&dev_delta_ptr, sizeof(double)));
    timing = wtime() - timing;
    printf("t_alloc_gpu: %lf ms\n\n", 1000*timing);
    timing = wtime(); 
    
    checkCuda(cudaMemcpy(deviceObjects, dimObjects[0],
              numObjs*numCoords*sizeof(double), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(deviceMembership, membership,
              numObjs*sizeof(int), cudaMemcpyHostToDevice));
    timing = wtime() - timing;
    printf("t_get_gpu: %lf ms\n\n", 1000*timing);
    timing = wtime();   
   	double cpu_to_gpu_time, gpu_start, gpu_time,
        gpu_to_cpu_start, gpu_to_cpu_time, cpu_start, cpu_time; 
    do {
    	timing_internal = wtime();

		/* GPU part: calculate new memberships */
		        
        /* TODO: Copy clusters to deviceClusters
        checkCuda(cudaMemcpy(...)); */
		// maybe dimClusters[0] in similarity with dimObjects[0] above ?
        checkCuda(cudaMemcpy(deviceClusters, dimClusters[0],
				  numClusters*numCoords*sizeof(double), cudaMemcpyHostToDevice));
        checkCuda(cudaMemset(dev_delta_ptr, 0, sizeof(double)));          
		
		cpu_to_gpu_time = wtime() - timing_internal;
        printf("\t\tCPU-GPU transfer : %lf ms\n", 1000*cpu_to_gpu_time);
        gpu_start = wtime();		

		//printf("Launching find_nearest_cluster Kernel with grid_size = %d, block_size = %d, shared_mem = %d KB\n", numClusterBlocks, numThreadsPerClusterBlock, clusterBlockSharedDataSize/1000);
        find_nearest_cluster
            <<< numClusterBlocks, numThreadsPerClusterBlock, clusterBlockSharedDataSize >>>
            (numCoords, numObjs, numClusters,
             deviceObjects, deviceClusters, deviceMembership, dev_delta_ptr);

        cudaDeviceSynchronize(); checkLastCudaError();
		//printf("Kernels complete for itter %d, updating data in CPU\n", loop);
		
		gpu_time = wtime() - gpu_start;
        printf("\t\tGPU part: %lf ms\n", 1000*gpu_time);
        gpu_to_cpu_start = wtime();	
		
		/* TODO: Copy deviceMembership to membership
        checkCuda(cudaMemcpy(...)); */
    	checkCuda(cudaMemcpy(membership, deviceMembership,
                  numObjs*sizeof(int), cudaMemcpyDeviceToHost));

    	/* TODO: Copy dev_delta_ptr to &delta
        checkCuda(cudaMemcpy(...)); */
		checkCuda(cudaMemcpy(&delta, dev_delta_ptr, sizeof(double), cudaMemcpyDeviceToHost));
		gpu_to_cpu_time = wtime() - gpu_to_cpu_start;
        printf("\t\tGPU-CPU transfer : %lf ms\n", 1000*gpu_to_cpu_time);
		/* CPU part: Update cluster centers*/
  		cpu_start = wtime();

        for (i=0; i<numObjs; i++) {
            /* find the array index of nestest cluster center */
            index = membership[i];
			
            /* update new cluster centers : sum of objects located within */
            newClusterSize[index]++;
            for (j=0; j<numCoords; j++)
                newClusters[j][index] += objects[i*numCoords + j];
        }
 
        /* average the sum and replace old cluster centers with newClusters */
        for (i=0; i<numClusters; i++) {
            for (j=0; j<numCoords; j++) {
                if (newClusterSize[i] > 0)
                    dimClusters[j][i] = newClusters[j][i] / newClusterSize[i];
                newClusters[j][i] = 0.0;   /* set back to 0 */
            }
            newClusterSize[i] = 0;   /* set back to 0 */
        }

        delta /= numObjs;
       	//printf("delta is %f - ", delta);
        loop++; 
        //printf("completed loop %d\n", loop);
		cpu_time = wtime() - cpu_start;
		timing_internal = wtime() - timing_internal; 
		printf("\t\tCPU part : %lf ms\n", 1000*cpu_time);
		if ( timing_internal < timer_min) timer_min = timing_internal; 
		if ( timing_internal > timer_max) timer_max = timing_internal; 
	} while (delta > threshold && loop < loop_threshold);
    
    /*TODO: Update clusters using dimClusters. Be carefull of layout!!! clusters[numClusters][numCoords] vs dimClusters[numCoords][numClusters] */ 
	//  [numCoords][numObjs] layout
    for (i = 0; i < numCoords; i++) {
        for (j = 0; j < numClusters; j++) {
			clusters[j*numCoords+i] = dimClusters[i][j];
        }
    }	
    timing = wtime() - timing;
    printf("nloops = %d  : total = %lf ms\n\t-> t_loop_avg = %lf ms\n\t-> t_loop_min = %lf ms\n\t-> t_loop_max = %lf ms\n\n|-------------------------------------------|\n", 
    	loop, 1000*timing, 1000*timing/loop, 1000*timer_min, 1000*timer_max);

	char outfile_name[1024] = {0}; 
	sprintf(outfile_name, "Execution_logs/silver1-V100_Sz-%lu_Coo-%d_Cl-%d.csv", numObjs*numCoords*sizeof(double)/(1024*1024), numCoords, numClusters);
	FILE* fp = fopen(outfile_name, "a+");
	if(!fp) error("Filename %s did not open succesfully, no logging performed\n", outfile_name); 
	fprintf(fp, "%s,%d,%lf,%lf,%lf\n", "Transpose", blockSize, timing/loop, timer_min, timer_max);
	fclose(fp); 
	
    checkCuda(cudaFree(deviceObjects));
    checkCuda(cudaFree(deviceClusters));
    checkCuda(cudaFree(deviceMembership));

    free(dimObjects[0]);
    free(dimObjects);
    free(dimClusters[0]);
    free(dimClusters);
    free(newClusters[0]);
    free(newClusters);
    free(newClusterSize);

    return;
}

