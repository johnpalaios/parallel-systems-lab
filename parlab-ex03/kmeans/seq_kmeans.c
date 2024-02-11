#include <stdio.h>
#include <stdlib.h>
#include "kmeans.h"

// square of Euclid dis://open.spotify.com/track/5nJtrGIe7ndSmRHfjuYnP0tance between two multi-dimensional points
inline static double euclid_dist_2(int    numdims,  /* no. dimensions */
                                 double * coord1,   /* [numdims] */
                                 double * coord2)   /* [numdims] */
{
    int i;
    double ans = 0.0;

    for(i=0; i<numdims; i++)
        ans += (coord1[i]-coord2[i]) * (coord1[i]-coord2[i]);

    return ans;
}

inline static int find_nearest_cluster(int     numClusters, /* no. clusters */
                                       int     numCoords,   /* no. coordinates */
                                       double * object,      /* [numCoords] */
                                       double * clusters)    /* [numClusters][numCoords] */
{
    int index, i;
    double dist, min_dist;

    // find the cluster id that has min distance to object 
    index = 0;
    min_dist = euclid_dist_2(numCoords, object, clusters);

    for(i=1; i<numClusters; i++) {
        dist = euclid_dist_2(numCoords, object, &clusters[i*numCoords]);
        // no need square root 
        if (dist < min_dist) { // find the min and its array index
            min_dist = dist;
            index    = i;
        }
    }
    return index;
}

void kmeans(double * objects,          /* in: [numObjs][numCoords] */
            int     numCoords,        /* no. coordinates */
            int     numObjs,          /* no. objects */
            int     numClusters,      /* no. clusters */
            double   threshold,        /* minimum fraction of objects that change membership */
            long    loop_threshold,   /* maximum number of iterations */
            int   * membership,       /* out: [numObjs] */
            double * clusters)         /* out: [numClusters][numCoords] */
{
    int i, j;
    int index, loop=0;
    double timing = wtime(), timing_internal, timer_min = 1e42, timer_max = 0;

    double delta;          // fraction of objects whose clusters change in each loop 
    int * newClusterSize; // [numClusters]: no. objects assigned in each new cluster 
    double * newClusters;  // [numClusters][numCoords] 

    printf("\n|-------------Sequential Kmeans-------------|\n\n");

    // initialize membership
    for (i=0; i<numObjs; i++) 
        membership[i] = -1;

    // initialize newClusterSize and newClusters to all 0 
    newClusterSize = (typeof(newClusterSize)) calloc(numClusters, sizeof(*newClusterSize));
    newClusters = (typeof(newClusters))  calloc(numClusters * numCoords, sizeof(*newClusters));
  
    timing = wtime() - timing;
    printf("t_alloc: %lf ms\n\n", 1000*timing);
    
    timing = wtime(); 
    do {
    	timing_internal = wtime(); 
        // before each loop, set cluster data to 0
        for (i=0; i<numClusters; i++) {
            for (j=0; j<numCoords; j++)
                newClusters[i*numCoords + j] = 0.0;
            newClusterSize[i] = 0;
        }

        delta = 0.0;

        for (i=0; i<numObjs; i++) {
            // find the array index of nearest cluster center 
            index = find_nearest_cluster(numClusters, numCoords, &objects[i*numCoords], clusters);

            // if membership changes, increase delta by 1 
            if (membership[i] != index)
                delta += 1.0;
            // assign the membership to object i 
            membership[i] = index;

            // update new cluster centers : sum of objects located within
            newClusterSize[index]++;
            for (j=0; j<numCoords; j++)
                newClusters[index*numCoords + j] += objects[i*numCoords + j];
        }
        // average the sum and replace old cluster centers with newClusters 
        for (i=0; i<numClusters; i++) {
            if (newClusterSize[i] > 0) {
                for (j=0; j<numCoords; j++) {
                    clusters[i*numCoords + j] = newClusters[i*numCoords + j] / newClusterSize[i];
                }
            }
        }

        // Get fraction of objects whose membership changed during this loop. This is used as a convergence criterion.
        delta /= numObjs;
		//printf("delta is %f - ", delta);
        loop++;
        //printf("completed loop %d\n", loop);
        //fflush(stdout);
        // printf("\tcompleted loop %d\n", loop);
        // for (i=0; i<numClusters; i++) {
        //     printf("\tclusters[%ld] = ",i);
        //     for (j=0; j<numCoords; j++)
        //         printf("%6.6f ", clusters[i*numCoords + j]);
        //     printf("\n");
        // }
        timing_internal = wtime() - timing_internal; 
        if ( timing_internal < timer_min) timer_min = timing_internal; 
        if ( timing_internal > timer_max) timer_max = timing_internal;         
    } while (delta > threshold && loop < loop_threshold);
    timing = wtime() - timing;
    printf("nloops = %d  : total = %lf ms\n\t-> t_loop_avg = %lf ms\n\t-> t_loop_min = %lf ms\n\t-> t_loop_max = %lf ms\n\n|-------------------------------------------|\n", 
    	loop, 1000*timing, 1000*timing/loop, 1000*timer_min, 1000*timer_max);

	char outfile_name[1024] = {0}; 
	sprintf(outfile_name, "Execution_logs/Sz-%lu_Coo-%d_Cl-%d.csv", numObjs*numCoords*sizeof(double)/(1024*1024), numCoords, numClusters);
	FILE* fp = fopen(outfile_name, "a+");
	if(!fp){ printf("Filename %s did not open succesfully, no logging performed\n", outfile_name); exit(42); }
	fprintf(fp, "%s,%d,%lf,%lf,%lf\n", "Sequential", -1, timing/loop, timer_min, timer_max);
	fclose(fp); 
	
    free(newClusters);
    free(newClusterSize);
}
