.KEEP_STATE:

MPICC = mpicc

CFLAGS = -Wall -Wextra -Wno-unused -O3

LDFLAGS = 

H_FILES = kmeans.h

COMM_SRC = file_io.c util.c

all: kmeans_mpi

kmeans_mpi: main.o file_io.o kmeans.o util.o
	$(MPICC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

main.o: main.c $(H_FILES)
	$(MPICC) $(CFLAGS) -c $< -o $@

kmeans.o: kmeans.c
	$(MPICC) $(CFLAGS) -c $< -o $@
file_io.o: file_io.c
	$(MPICC) $(CFLAGS) -c $< -o $@

util.o: util.c
	$(MPICC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf *.o kmeans_mpi
