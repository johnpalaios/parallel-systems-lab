#include <stdio.h>
#include <stdlib.h>
#include "kmeans.h"

// square of Euclid distance between two multi-dimensional points
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

inline static int find_nearest_cluster(int      numClusters, /* no. clusters */
                                       int      numCoords,   /* no. coordinates */
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
            int      numCoords,        /* no. coordinates */
            int      numObjs,          /* no. objects */
            int      numClusters,      /* no. clusters */
            double   threshold,        /* minimum fraction of objects that change membership */
            long     loop_threshold,   /* maximum number of iterations */
            int    * membership,       /* out: [numObjs] */
            double * clusters)         /* out: [numClusters][numCoords] */
{
    int i, j;
    int index, loop=0;
    double timing = 0;

    double delta;          // fraction of objects whose clusters change in each loop 
    int * newClusterSize; // [numClusters]: no. objects assigned in each new cluster 
    double * newClusters;  // [numClusters][numCoords] 

    printf("Sequential Kmeans\n");

    // initialize membership
    for (i=0; i<numObjs; i++) 
        membership[i] = -1;

    // initialize newClusterSize and newClusters to all 0 
    newClusterSize = (typeof(newClusterSize)) calloc(numClusters, sizeof(*newClusterSize));
    newClusters = (typeof(newClusters))  calloc(numClusters * numCoords, sizeof(*newClusters));

    timing = wtime();   
    
    do {
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

        loop++;
        printf("\r\tcompleted loop %d", loop);
        fflush(stdout);
    } while (delta > threshold && loop < loop_threshold);
    timing = wtime() - timing;
    printf("\n        nloops = %3d   (total = %7.4fs)  (per loop = %7.4fs)\n", loop, timing, timing/loop);

    free(newClusters);
    free(newClusterSize);
}
