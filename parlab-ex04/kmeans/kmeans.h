#ifndef _H_KMEANS
#define _H_KMEANS

#include <assert.h>

void kmeans(double * objects, int numCoords, int numObjs, int numClusters, double threshold, long loop_threshold, int *membership, double * clusters);

double * dataset_generation(int numObjs, int numCoords, long *rank_numObjs);

int check_repeated_clusters(int, int, double*);

double wtime(void);

extern int _debug;

#endif
