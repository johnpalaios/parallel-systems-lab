#ifndef _H_KMEANS
#define _H_KMEANS

#include <assert.h>

void kmeans(double * objects, int numCoords, int numObjs, int numClusters, double threshold, long loop_threshold, int *membership, double * clusters);

void kmeans_gpu(double * objects, int numCoords, int numObjs, int numClusters, double threshold, long loop_threshold, int *membership, double * clusters, int block_size);

double * dataset_generation(int numObjs, int numCoords);

int check_repeated_clusters(int, int, double*);

double wtime(void);

extern int _debug;

#endif
