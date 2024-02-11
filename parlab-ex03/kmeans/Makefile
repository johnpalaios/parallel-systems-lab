.KEEP_STATE:

OBJECT_DIR = ObjectDir
HELPER_DIR = helper_functions

CPP          = g++
NVCC        = /usr/local/cuda-12.1/bin/nvcc

# TODO: Enable VALIDATE when implementing new versions to check results in respect to serial execution.  
#VALIDATE_FLAG=-DVALIDATE
#VALIDATE_FLAG=#IMPORTANT: Disable in long benchmarks if correctness is established.

CFLAGS  = -O3 -I$(HELPER_DIR) $(VALIDATE_FLAG)
NVCCFLAGS   =$(CFLAGS) --ptxas-options=-v -arch=sm_60\
#-gencode=arch=compute_35,code=sm_35\
#-gencode=arch=compute_52,code=sm_52\
-gencode=arch=compute_60,code=sm_60\
#-gencode=arch=compute_61,code=sm_61\
#-gencode=arch=compute_75,code=sm_75\
#-gencode=arch=compute_80,code=sm_80\
#-gencode=arch=compute_86,code=sm_86\
#-gencode=arch=compute_87,code=sm_87\
#-gencode=arch=compute_86,code=compute_86

LDFLAGS     =
LIBS        =

SEQ_HELP_OBJ = $(OBJECT_DIR)/main_sec.o $(OBJECT_DIR)/file_io.o $(OBJECT_DIR)/util.o 
CUDA_HELP_OBJ = $(OBJECT_DIR)/main_gpu.o  $(OBJECT_DIR)/file_io.o $(OBJECT_DIR)/util.o $(OBJECT_DIR)/error.o $(OBJECT_DIR)/alloc.o $(OBJECT_DIR)/seq_kmeans.o

#all: kmeans_seq kmeans_cuda_naive kmeans_cuda_transpose kmeans_cuda_shared kmeans_cuda_all_gpu kmeans_cuda_all_gpu_delta_reduction

all : kmeans_seq kmeans_cuda_all_gpu

#---------------------------------------------------------------------

kmeans_seq: $(OBJECT_DIR)/seq_kmeans.o $(SEQ_HELP_OBJ)
	$(CPP) $(CFLAGS) $^ -o $@ $(LDFLAGS)

kmeans_cuda_naive: $(OBJECT_DIR)/cuda_kmeans_naive.o $(CUDA_HELP_OBJ)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS) 


kmeans_cuda_transpose: $(OBJECT_DIR)/cuda_kmeans_transpose.o $(CUDA_HELP_OBJ)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS) 
		
kmeans_cuda_shared: $(OBJECT_DIR)/cuda_kmeans_shared.o $(CUDA_HELP_OBJ)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS) 

kmeans_cuda_all_gpu: $(OBJECT_DIR)/cuda_kmeans_all_gpu.o $(CUDA_HELP_OBJ)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS) 

kmeans_cuda_all_gpu_delta_reduction: $(OBJECT_DIR)/cuda_kmeans_all_gpu_delta_reduction.o $(CUDA_HELP_OBJ)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS) 
		
kmeans_cuda_all_gpu_reduction: $(OBJECT_DIR)/cuda_kmeans_all_gpu_reduction.o $(CUDA_HELP_OBJ)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS) 

kmeans_cuda_gitman: $(OBJECT_DIR)/cuda_kmeans_gitman.o $(CUDA_HELP_OBJ)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS) 


$(OBJECT_DIR)/cuda_kmeans_transpose.o: cuda_kmeans_transpose.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJECT_DIR)/cuda_kmeans_naive.o: cuda_kmeans_naive.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
		
$(OBJECT_DIR)/cuda_kmeans_shared.o: cuda_kmeans_shared.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJECT_DIR)/cuda_kmeans_all_gpu.o: cuda_kmeans_all_gpu.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJECT_DIR)/cuda_kmeans_all_gpu_delta_reduction.o: cuda_kmeans_all_gpu_delta_reduction.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
	
#$(OBJECT_DIR)/cuda_kmeans_all_gpu_reduction.o: cuda_kmeans_all_gpu_reduction.cu
#	$(NVCC) $(NVCCFLAGS) -c $< -o $@
		
#$(OBJECT_DIR)/cuda_kmeans_gitman.o: cuda_kmeans_gitman.cu
#	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJECT_DIR)/main_gpu.o: main_gpu.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
				
$(OBJECT_DIR)/main_sec.o: main_sec.c
	$(CPP) $(CFLAGS) -c $< -o $@

$(OBJECT_DIR)/seq_kmeans.o: seq_kmeans.c
	$(CPP) $(CFLAGS) -c $< -o $@

$(OBJECT_DIR)/file_io.o: $(HELPER_DIR)/file_io.c
	$(CPP) $(CFLAGS) -c $< -o $@

$(OBJECT_DIR)/error.o: $(HELPER_DIR)/error.c
	$(CPP) $(CFLAGS) -c $< -o $@

$(OBJECT_DIR)/alloc.o: $(HELPER_DIR)/alloc.c
	$(CPP) $(CFLAGS) -c $< -o $@
		
$(OBJECT_DIR)/util.o: $(HELPER_DIR)/util.c
	$(CPP) $(CFLAGS) -c $< -o $@

clean:
	rm -rf *.o -rf $(OBJECT_DIR)/*.o kmeans_seq kmeans_cuda_naive kmeans_cuda_transpose kmeans_cuda_shared kmeans_cuda_all_gpu kmeans_cuda_all_gpu_delta_reduction
