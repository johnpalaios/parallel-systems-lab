CC = mpicc

CFLAGS =    

OPTIONS =   

H_FILES = utils.h

COMM_SRC = utils.c

all: jacobi_mpi

jacobi_mpi: jacobi_mpi.o utils.o 
	mpicc -lm $(CFLAGS) $(OPTIONS) -o jacobi_mpi jacobi_mpi.o utils.o

jacobi_mpi.o: jacobi_mpi.c $(COMM_SRC) $(H_FILES) 
	$(CC) $(CFLAGS) $(OPTIONS) -c $< -o $@
		
utils.o: utils.c 
	$(CC) $(CFLAGS) -c $< -o $@ 

clean:
	rm -rf jacobi_mpi *.o


