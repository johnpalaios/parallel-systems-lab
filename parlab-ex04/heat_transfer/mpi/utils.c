#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"

double max(double a, double b) {
	return a>b?a:b;
}

int converge(double ** u_previous, double ** u_current, int i_min, int i_max, int j_min, int j_max) {
	int i,j;
	for (i=i_min;i<=i_max;i++)
		for (j=j_min;j<=j_max;j++)
			if (fabs(u_current[i][j]-u_previous[i][j])>e) return 0;
	return 1;
}

double ** allocate2d(int dimX, int dimY) {
	double ** array, * tmp;
	int i;
	tmp = ( double * )calloc( dimX * dimY, sizeof( double ) );
	array = ( double ** )calloc( dimX, sizeof( double * ) );
	for ( i = 0 ; i < dimX ; i++ )
		array[i] = tmp + i * dimY;
	if ( array == NULL || tmp == NULL) {
		fprintf( stderr,"Error in allocation\n" );
		exit( -1 );
	}
	return array;
}

void free2d(double ** array) {
	if (array == NULL) {
		fprintf(stderr,"Error in freeing matrix\n");
		exit(-1);
	}
	if (array[0])
		free(array[0]);
	if (array)
		free(array);
}

void init2d(double ** array, int dimX, int dimY) {
	int i,j;
	for ( i = 0 ; i < dimX ; i++ )
		for ( j = 0; j < dimY ; j++) 
			array[i][j]=(i==0 || i==dimX-1 || j==0 || j==dimY-1)?0.01*(i+1)+0.001*(j+1):0.0;
}

void zero2d(double ** array, int dimX, int dimY) {
	int i,j;
	for ( i = 0 ; i < dimX ; i++ )
		for ( j = 0; j < dimY ; j++) 
			array[i][j] = 0.0;
}

void print2d(double ** array, int dimX, int dimY) {
	int i,j;
	for (i=0;i<dimX;i++) {
		for (j=0;j<dimY;j++)
			printf("%lf ",array[i][j]);
		printf("\n");
	}
}

void fprint2d(char * s, double ** array, int dimX, int dimY) {
	int i,j;
	FILE * f=fopen(s,"w");
	for (i=0;i<dimX;i++) {
		for (j=0;j<dimY;j++)
			fprintf(f,"%lf ",array[i][j]);
		fprintf(f,"\n");
	}
	fclose(f);
}

void print_int_arr(double *arr, int size, int rank) {
    // Calculate the maximum size needed for the string
    // Assuming each integer takes up at most 10 characters (including possible sign and digits)
    int maxStringSize = size * 20;

    // Allocate memory for the string
    char *resultString = (char *)malloc(maxStringSize * sizeof(char));

    // Check if allocation was successful
    if (resultString == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return; // Return without printing if allocation fails
    }

    // Initialize the result string
    resultString[0] = '\0';
    int i;
    // Concatenate each element to the result string
    for (i = 0; i < size; i++) {
        char temp[22]; // Assuming each integer takes up at most 10 characters + 1 for null-terminator
        sprintf(temp, "%.2lf ", arr[i]);
        strcat(resultString, temp);
    }

    // Print the formatted string
    printf("Rank: %d and arr: %s\n", rank, resultString);

    // Free the dynamically allocated memory
    free(resultString);
}

