#include <stdio.h>
#include <stdlib.h>
#include <string.h>     /* strtok() */
#include <sys/types.h>  /* open() */
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>     /* getopt() */

int _debug;
#include "kmeans.h"

static void usage(char *argv0) {
    char *help =
        "Usage: %s [switches]\n"
        "       -c num_clusters    : number of clusters (must be > 1)\n"
        "       -s size            : size of examined dataset\n"
        "       -n num_coords      : number of coordinates\n"
        "       -t threshold       : threshold value (default : 0.001)\n"
        "       -l loop_threshold  : iterations threshold (default : 10)\n"
        "       -d                 : enable debug mode\n"
        "       -h                 : print this help information\n";
    fprintf(stderr, help, argv0);
    exit(-1);
}

int main(int argc, char **argv)
{
    long i, j, opt;
    extern char* optarg;
    extern int optind;

    long     numClusters=0, numCoords=0, numObjs=0;
    int    * membership;    // [numObjs]
    double * objects;       // [numObjs * numCoords] data  objects
    double * clusters;      // [numClusters * numCoords] cluster center
    double   dataset_size = 0, threshold;
    long     loop_threshold;
    double   io_timing_read;

    /* some default values */
    _debug         = 0;
    threshold      = 0.001;
    loop_threshold = 10;
    numClusters    = 0;

    printf("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n");

    while ( (opt = getopt(argc,argv,"n:t:l:c:s:dh")) != EOF) {
        switch (opt) {
            case 'c': numClusters = atol(optarg);
                      break;
            case 't': threshold=atof(optarg);
                      break;
            case 'l': loop_threshold=atol(optarg);
                      break;
            case 's': dataset_size=atof(optarg);
                      break;
            case 'n': numCoords=atol(optarg);
                      break;
            case 'd': _debug = 1;
                      break;
            case 'h':
            default: usage(argv[0]);
                      break;
        }
    }
    if (numClusters <= 1)
        usage(argv[0]);

    numObjs = (dataset_size*1024*1024) / (numCoords*sizeof(double));

    if (numObjs < numClusters) {
        printf("Error: number of clusters must be larger than the number of data points to be clustered.\n");
        return 1;
    }
    printf("dataset_size = %.2f MB    numObjs = %ld    numCoords = %ld    numClusters = %ld\n", dataset_size, numObjs, numCoords, numClusters);

    objects = dataset_generation(numObjs, numCoords);

    // Allocate space for clusters (coordinates of cluster centers)
    clusters = (double*)  malloc(numClusters * numCoords * sizeof(double));

    // The first numClusters elements are selected as initial centers
    for (i=0; i<numClusters; i++)
        for (j=0; j<numCoords; j++)
            clusters[i*numCoords + j] = objects[i*numCoords + j];

    // check initial cluster centers for repeatition 
    if (check_repeated_clusters(numClusters, numCoords, clusters) == 0) {
        printf("Error: some initial clusters are repeated. Please select distinct initial centers\n");
        return 1;
    }

    printf("Initial cluster centers:\n");
    for (i=0; i<numClusters; i++) {
        printf("clusters[%ld] =",i);
        for (j=0; j<numCoords; j++)
            printf(" %6.2f", clusters[i*numCoords + j]);
        printf("\n");
    }

    // membership: the cluster id for each data object
    membership = (int*) malloc(numObjs * sizeof(int));

    // start the core computation
    printf("\n");
    kmeans(objects, numCoords, numObjs, numClusters, threshold, loop_threshold, membership, clusters);
    printf("\n");

    printf("Final cluster centers:\n");
    for (i=0; i<numClusters; i++) {
        printf("clusters[%ld] = ",i);
        for (j=0; j<numCoords; j++)
            printf("%6.2f ", clusters[i*numCoords + j]);
        printf("\n");
    }

    free(objects);
    free(membership);
    free(clusters);

    return 0;
}
