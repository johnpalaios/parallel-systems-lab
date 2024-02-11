#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "mpi.h"
#include "utils.h"
#include <string.h>

int main(int argc, char ** argv) {
    int rank,size;
    int global[2],local[2]; //global matrix dimensions and local matrix dimensions (2D-domain, 2D-subdomain)
    int global_padded[2];   //padded global matrix dimensions (if padding is not needed, global_padded=global)
    int grid[2];            //processor grid dimensions
    int i,j,t;
    int global_converged=0,converged=0; //flags for convergence, global and per process
    MPI_Datatype dummy;     //dummy datatype used to align user-defined datatypes in memory
    double omega; 			//relaxation factor - useless for Jacobi

    struct timeval tts,ttf,tcs,tcf, tconvs, tconvf;   //Timers: total-> tts,ttf, computation -> tcs,tcf
    double ttotal=0,tcomp=0, tconv = 0, total_time,comp_time,conv_time;
    
    double ** U, ** u_current, ** u_previous, ** swap; //Global matrix, local current and previous matrices, pointer to swap between current and previous
    

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    //----Read 2D-domain dimensions and process grid dimensions from stdin----//

    if (argc!=5) {
        fprintf(stderr,"Usage: mpirun .... ./exec X Y Px Py");
        exit(-1);
    }
    else {
        global[0]=atoi(argv[1]);
        global[1]=atoi(argv[2]);
        grid[0]=atoi(argv[3]);
        grid[1]=atoi(argv[4]);
    }
	//----Create 2D-cartesian communicator----//
	//----Usage of the cartesian communicator is optional----//

    MPI_Comm CART_COMM;         //CART_COMM: the new 2D-cartesian communicator
    int periods[2]={0,0};       //periods={0,0}: the 2D-grid is non-periodic
    int rank_grid[2];           //rank_grid: the position of each process on the new communicator
		
    MPI_Cart_create(MPI_COMM_WORLD,2,grid,periods,0,&CART_COMM);    //communicator creation
    MPI_Cart_coords(CART_COMM,rank,2,rank_grid);	                //rank mapping on the new communicator

    //----Compute local 2D-subdomain dimensions----//
    //----Test if the 2D-domain can be equally distributed to all processes----//
    //----If not, pad 2D-domain----//
    
    for (i=0;i<2;i++) {
        if (global[i]%grid[i]==0) {
            local[i]=global[i]/grid[i];
            global_padded[i]=global[i];
        }
        else {
            local[i]=(global[i]/grid[i])+1;
            global_padded[i]=local[i]*grid[i];
        }
    }

	//Initialization of omega
    omega=2.0/(1+sin(3.14/global[0]));

    //----Allocate global 2D-domain and initialize boundary values----//
    //----Rank 0 holds the global 2D-domain----//
    if (rank==0) {
        U=allocate2d(global_padded[0],global_padded[1]);   
        init2d(U,global[0],global[1]);
    }

    //----Allocate local 2D-subdomains u_current, u_previous----//
    //----Add a row/column on each size for ghost cells----//

    u_previous=allocate2d(local[0]+2,local[1]+2);
    u_current=allocate2d(local[0]+2,local[1]+2);   
	//printf("local[0] = %d & local[1] = %d\n", local[0], local[1]);       
    //----Distribute global 2D-domain from rank 0 to all processes----//
         
 	//----Appropriate datatypes are defined here----//
	/*****The usage of datatypes is optional*****/
    
    //----Datatype definition for the 2D-subdomain on the global matrix----//

    MPI_Datatype global_block;
    MPI_Type_vector(local[0],local[1],global_padded[1],MPI_DOUBLE,&dummy);
    MPI_Type_create_resized(dummy,0,sizeof(double),&global_block);
    MPI_Type_commit(&global_block);

    //----Datatype definition for the 2D-subdomain on the local matrix----//

    MPI_Datatype local_block;
    MPI_Type_vector(local[0],local[1],local[1]+2,MPI_DOUBLE,&dummy);
    MPI_Type_create_resized(dummy,0,sizeof(double),&local_block);
	MPI_Type_commit(&local_block);

    //----Rank 0 defines positions and counts of local blocks (2D-subdomains) on global matrix----//
    int * scatteroffset, * scattercounts;
    if (rank==0) {
        scatteroffset=(int*)malloc(size*sizeof(int));
        scattercounts=(int*)malloc(size*sizeof(int));
        for (i=0;i<grid[0];i++)
            for (j=0;j<grid[1];j++) {
                scattercounts[i*grid[1]+j]=1;
				scatteroffset[i*grid[1]+j]=(local[0]*local[1]*grid[1]*i+local[1]*j);
            }
    }

    //----Rank 0 scatters the global matrix----//

	//*************TODO*******************//



	/*Fill your code here*/
	/*
	if(rank==0) {
        for(i=0;i<10;i++)
            for(j=0;j<10;j++)
                printf("%f ", U[i*local[1]+j]);
    }
	*/
	/*Make sure u_current and u_previous are
		both initialized*/

		
    MPI_Scatterv(&(U[0][0]), scattercounts, scatteroffset, global_block, &(u_previous[1][1]), 1, local_block, 0, CART_COMM);
    //MPI_Scatterv(&(U[0][0]), scattercounts, scatteroffset, global_block, &(u_current[1][1]), 1, local_block, 0, CART_COMM);	
	
	for(i = 0; i < local[0]+2; i++) {
		for(j = 0; j < local[1]+2; j++) {
			u_current[i][j] = u_previous[i][j];	
		}
	}
	/*
	printf("Proccess %d has rank_grid = (%d, %d): \n",rank, rank_grid[0], rank_grid[1]);
    
	for(i=0;i<4;i++)
        for(j=0;j<4;j++)
            printf("%f ",u_previous[i][j]);
    printf("\n"); 
	*/
    



     //************************************//


    if (rank==0)
        free2d(U);

 
     
	//----Define datatypes or allocate buffers for message passing----//

	//*************TODO*******************//

	

	/*Fill your code here*/	
	int row_idx = rank_grid[0], col_idx = rank_grid[1];  
	int *num_of_rows = (int*) malloc(sizeof(int));
	int *num_of_cols = (int*) malloc(sizeof(int));
	*num_of_rows = local[0]; 
	*num_of_cols = local[1];
	int *row_size = (int*) malloc(sizeof(int)); 
	int *col_size = (int*) malloc(sizeof(int));
	*row_size = local[0]+2;
	*col_size = local[1]+2;
	

	// send element arrays
	double *north_arr = (double*) malloc(*num_of_cols * sizeof(double));
	double *south_arr = (double*) malloc(*num_of_cols * sizeof(double));
	double *east_arr = (double*) malloc(*num_of_rows * sizeof(double));
    double *west_arr = (double*) malloc(*num_of_rows * sizeof(double));
	
	// receive element arrays 
	double *north_arr_recv = (double*) malloc((*num_of_cols) * sizeof(double));
    double *south_arr_recv = (double*) malloc((*num_of_cols) * sizeof(double));
    double *east_arr_recv = (double*) malloc((*num_of_rows) * sizeof(double));
    double *west_arr_recv = (double*) malloc((*num_of_rows) * sizeof(double));	
	
	//************************************//


    //----Find the 4 neighbors with which a process exchanges messages----//

	//*************TODO*******************//
    int north, south, east, west;
	north = row_idx - 1;
	south = row_idx + 1;
	east = col_idx + 1;
	west = col_idx - 1;
	

	/*Fill your code here*/

	/*Make sure you handle non-existing
		neighbors appropriately*/

	//************************************//

    //---Define the iteration ranges per process-----//
	//*************TODO*******************//

    int i_min,i_max,j_min,j_max;
	
	/*Fill your code here*/
	
	// if is west boundary
	if(west < 0) {
		j_min = 2;
	} else {
		j_min = 1;
	}

	// if is east boundary
	if(east >= grid[1]) {
		j_max = *num_of_cols-1;
	} else {
		j_max = *num_of_cols;
 	}

	// if is north boundary
	if(north < 0) {
		i_min = 2;
	} else {
		i_min = 1;
	}

	// if is south boundary
	if(south >= grid[0]) {
		i_max = *num_of_rows-1;
	} else {
		i_max = *num_of_rows;
	}

	/*Three types of ranges:
		-internal processes
		-boundary processes
		-boundary processes and padded global array
	*/
	
	int *cart_coords_north = (int*) malloc(2 * sizeof(int));
    int *cart_coords_south = (int*) malloc(2 * sizeof(int));
    int *cart_coords_west = (int*) malloc(2 * sizeof(int));
    int *cart_coords_east = (int*) malloc(2 * sizeof(int));

	cart_coords_north[0] = row_idx-1;
    cart_coords_north[1] = col_idx;
    cart_coords_south[0] = row_idx+1;
    cart_coords_south[1] = col_idx;
    cart_coords_east[0] = row_idx;
    cart_coords_east[1] = col_idx+1;
    cart_coords_west[0] = row_idx;
    cart_coords_west[1] = col_idx-1;

	//************************************//

	MPI_Status *north_stats = (MPI_Status*) malloc(2 *sizeof(MPI_Status));
    MPI_Status *south_stats = (MPI_Status*) malloc(2 *sizeof(MPI_Status));
    MPI_Request *north_reqs = (MPI_Request*) malloc(2 *sizeof(MPI_Request));
    MPI_Request *south_reqs = (MPI_Request*) malloc(2 *sizeof(MPI_Request));
	MPI_Status *east_stats = (MPI_Status*) malloc(2 *sizeof(MPI_Status));
    MPI_Status *west_stats = (MPI_Status*) malloc(2 *sizeof(MPI_Status));
    MPI_Request *east_reqs = (MPI_Request*) malloc(2 *sizeof(MPI_Request));
    MPI_Request *west_reqs = (MPI_Request*) malloc(2 *sizeof(MPI_Request));	

	int *tags = (int*) malloc(4 * sizeof(int));
	for(i = 0; i < 4; i++)
		tags[i] = 0;
	int *c = (int*) malloc(1 * sizeof(int));
	
    int *cart_rank_north = (int*) malloc(1 * sizeof(int));
	int *cart_rank_south = (int*) malloc(1 * sizeof(int));
	int *cart_rank_east = (int*) malloc(1 * sizeof(int));
	int *cart_rank_west = (int*) malloc(1 * sizeof(int));
	
	*cart_rank_north = -1;
	*cart_rank_south = -1;
	*cart_rank_east = -1;
	*cart_rank_west = -1;
	
	int coord_var = grid[0];
	if(cart_coords_north[0]>=0) {
    	*cart_rank_north = coord_var*cart_coords_north[0] + cart_coords_north[1];
	}
	if(cart_coords_south[0]<grid[0]) {
    	*cart_rank_south = coord_var*cart_coords_south[0] + cart_coords_south[1];
	}
	if(cart_coords_west[1]>=0) {
  		*cart_rank_west = coord_var*cart_coords_west[0] + cart_coords_west[1];
	}
	if(cart_coords_east[1]<grid[1]) {
        *cart_rank_east = coord_var*cart_coords_east[0] + cart_coords_east[1];
    }
	//int cart_rank;
	//MPI_Cart_rank(CART_COMM, rank_grid, &cart_rank);	
	
 	//----Computational core----//   
	int old_grid = grid[0];	
	//printf("1. grid[0] : %d\n", grid[0]);
	gettimeofday(&tts, NULL);
    #ifdef TEST_CONV
    for (t=0;t<T && !global_converged;t++) {
    #endif
    #ifndef TEST_CONV
    #undef T
    #define T 256
    for (t=0;t<T;t++) {
    #endif

	 	///*************TODO*******************//
     
		/*Fill your code here*/
		
		// initialize boundary arrays
    	for(i = 1; i < (*num_of_cols)+1; i++) {
			north_arr[i-1] = u_previous[1][i];
        	south_arr[i-1] = u_previous[*num_of_rows][i];
    	}
		 	
    	for(i = 1; i < (*num_of_rows)+1; i++) {
        	west_arr[i-1] = u_previous[i][1];
        	east_arr[i-1] = u_previous[i][*num_of_cols];
		}
		
			
		// sending phase
		if(*cart_rank_north > -1) {	
			MPI_Isend(north_arr, *num_of_cols, MPI_DOUBLE, *cart_rank_north, tags[0], CART_COMM, &(north_reqs[0]));
		} 
		
		if(*cart_rank_south > -1) {	
			MPI_Isend(south_arr, *num_of_cols, MPI_DOUBLE, *cart_rank_south, tags[1], CART_COMM, &(south_reqs[0]));
		}
		
		if(*cart_rank_west > -1) {			
			MPI_Isend(west_arr, *num_of_rows, MPI_DOUBLE, *cart_rank_west, tags[2], CART_COMM, &(west_reqs[0]));
		}
		if(*cart_rank_east > -1) {	
			MPI_Isend(east_arr, *num_of_rows, MPI_DOUBLE, *cart_rank_east, tags[3], CART_COMM, &(east_reqs[0]));
    	}
		
		// receiving phase
		if(*cart_rank_north > -1) {
			MPI_Irecv(north_arr_recv, *num_of_cols, MPI_DOUBLE, *cart_rank_north, tags[0], CART_COMM, &north_reqs[1]);
		}
        if(*cart_rank_south > -1) {
            MPI_Irecv(south_arr_recv, *num_of_cols, MPI_DOUBLE, *cart_rank_south, tags[1], CART_COMM, &south_reqs[1]);
        }
        if(*cart_rank_west > -1) {
            MPI_Irecv(west_arr_recv, *num_of_rows, MPI_DOUBLE, *cart_rank_west, tags[2], CART_COMM, &west_reqs[1]);
		}
        if(*cart_rank_east > -1) {
			MPI_Irecv(east_arr_recv, *num_of_rows, MPI_DOUBLE, *cart_rank_east, tags[3], CART_COMM, &east_reqs[1]);           
		}
					
		// process synchronization
		if(*cart_rank_north > -1) {
			MPI_Waitall(2, north_reqs, north_stats);
		}
		if(*cart_rank_south > -1) {
            MPI_Waitall(2, south_reqs, south_stats);
        }
		if(*cart_rank_west > -1) {
            MPI_Waitall(2, west_reqs, west_stats);
        }
		if(*cart_rank_east > -1) {
            MPI_Waitall(2, east_reqs, east_stats);
        }
		
		/*Compute and Communicate*/
		
		/*Add appropriate timers for computation*/
				
		gettimeofday(&tcs, NULL);
		for(i = i_min; i < i_max; i++) {
			for(j = j_min; j < j_max; j++) {	
				u_current[i][j] = (u_previous[i-1][j] + u_previous[i][j-1]+u_previous[i+1][j]+u_previous[i][j+1])/4;
			}
		} 
		gettimeofday(&tcf, NULL);
		tcomp += (tcf.tv_sec-tcs.tv_sec)+(tcf.tv_usec-tcs.tv_usec)*0.000001;      
			
		#ifdef TEST_CONV
		if (t%C==0) {
			gettimeofday(&tconvs, NULL);
			converged=1;
			for(i = i_min; i < i_max; i++) {
				for(j = j_min; j < j_max; j++) {
					if(abs(u_current[i][j]-u_previous[i][j])>0.01) {
						converged=0;					
					 	i = i_max;
						j = j_max;
						break;
					}
				}
			}
			gettimeofday(&tconvf, NULL);
			tconv += (tconvf.tv_sec-tconvs.tv_sec)+(tconvf.tv_usec-tconvs.tv_usec)*0.000001;
		}	
		#endif
		for(i = i_min; i < i_max; i++) {
            for(j = j_min; j < j_max; j++) {
                u_previous[i][j] = u_current[i][j];
           }
        }
		//************************************//
		MPI_Allreduce(&converged, &global_converged, 1, MPI_DOUBLE, MPI_MIN, CART_COMM);
	}
	
    gettimeofday(&ttf,NULL);
	grid[0] = old_grid;
	//printf("2. grid[0] : %d\n", grid[0]);	
    ttotal=(ttf.tv_sec-tts.tv_sec)+(ttf.tv_usec-tts.tv_usec)*0.000001;
	//free(u_previous);
	free(row_size);
	free(col_size);
    free(cart_coords_north);
    free(cart_coords_south);
    free(cart_coords_west);
    free(cart_coords_east);
	MPI_Reduce(&ttotal,&total_time,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Reduce(&tcomp,&comp_time,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
	MPI_Reduce(&tconv,&conv_time,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

    //----Rank 0 gathers local matrices back to the global matrix----//
   
    if (rank==0) {
            U=allocate2d(global_padded[0],global_padded[1]);
    }


	//*************TODO*******************//
	/*Fill your code here*/			
	int * gatheroffset, * gathercounts;
    if (rank==0) {
        
		gatheroffset=(int*)malloc(size*sizeof(int));
        gathercounts=(int*)malloc(size*sizeof(int));
        
		for (i=0;i<grid[0];i++)
            for (j=0;j<grid[1];j++) {
                gathercounts[i*grid[1]+j]=1;
				//gatheroffset[i*grid[1]+j]=0;
                gatheroffset[i*grid[1]+j]=(local[0]*local[1]*grid[1]*i+local[1]*j);
				//printf("gatheroffset[%d] = %d\n", i*grid[1]+j, (local[0]*local[1]*grid[1]*i+local[1]*j));
            }
		gatheroffset[(grid[0]-1)*grid[1]+grid[1]-1] -= 1;
	}
	
	//MPI_Bcast(gatheroffset, size, MPI_INT, 0, CART_COMM);
    //MPI_Bcast(gathercounts, size, MPI_INT, 0, CART_COMM);	
	/*
	if(rank==-1) {
		printf("U size : %d | last offset : %d | global_block size : %d | sum : %d \n", global_padded[0]*global_padded[1], local[0]*local[1]*grid[1]*(grid[0]-1)+local[1]*(grid[1]-1), local[0]*global_padded[1]+local[1], local[0]*local[1]*grid[1]*(grid[0]-1)+local[1]*(grid[1]-1)+local[0]*global_padded[1]+local[1]);
	}
	*/	

	MPI_Gatherv(&(u_current[1][1]), 1, local_block, &(U[0][0]), gathercounts, gatheroffset, global_block, 0, CART_COMM);
	
	//free(u_current);
     //************************************//


	
  	//printf("grid[0] : %d\t", grid[0]); 

	//----Printing results----//

	//**************TODO: Change "Jacobi" to "GaussSeidelSOR" or "RedBlackSOR" for appropriate printing****************//
    if (rank==0) {
        printf("Jacobi X %d Y %d Px %d Py %d Iter %d ComputationTime %lf CommunicationTime %lf Convergence Time %lf TotalTime %lf midpoint %lf\n",global[0],global[1],grid[0],grid[1],t,comp_time, total_time-comp_time, conv_time, total_time,U[global[0]/2][global[1]/2]);
	
        #ifdef PRINT_RESULTS
        char * s=malloc(50*sizeof(char));
        sprintf(s,"resJacobiMPI_%dx%d_%dx%d",global[0],global[1],grid[0],grid[1]);
        fprint2d(s,U,global[0],global[1]);
        free(s);
        #endif

    }
    MPI_Finalize();
    return 0;
}
