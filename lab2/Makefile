
# nvcc -o heatdist heatdist.cu -lm

# CC=g++
NVCC=nvcc
# CXXFLAGS= -fopenmp -O3 -Wextra -std=c++11
# CUDAFLAGS= -std=c++11 -c -arch=sm_20
# LIBS= -lopenblas -lpthread -lcudart -lcublas
LIBS= -lm
# LIBDIRS=-L/usr/local/cuda-7.5/lib64
# INCDIRS=-I/usr/local/cuda-7.5/include
vectorprog: heatdist.cu
	$(NVCC) -o heatdist heatdist.cu $(LIBS)
    # $(NVCC) $(CUDAFLAGS)   matrix_cuda.cu 
# all: matrix_cuda.o
#         $(CC) -o test matrix_blas.cpp alg.cpp test.cpp matrix_cuda.o $(LIBDIRS) $(INCDIRS) $(LIBS) $(CXXFLAGS)
clean:
	rm -rf heatdist