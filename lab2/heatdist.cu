/* 
 * This file contains the code for doing the heat distribution problem. 
 * You do not need to modify anything except starting  gpu_heat_dist() at the bottom
 * of this file.
 * In gpu_heat_dist() you can organize your data structure and the call to your
 * kernel(s), memory allocation, data movement, etc. 
 * 
 */

#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h> 

/* To index element (i,j) of a 2D square array of dimension NxN stored as 1D */
#define index(i, j, N)  ((i)*(N)) + (j)
#define WIDTH 75

/*****************************************************************/

// Function declarations: Feel free to add any functions you want.
void  seq_heat_dist(float *, unsigned int, unsigned int);
void  gpu_heat_dist(float *, unsigned int, unsigned int);
void  gpu_optimized_heat_dist(float *, unsigned int, unsigned int);

// Kernel funtions
__global__ void updateTemperatures(float* current, float* last, int n);
__global__ void updateTemperaturesOptimized(float* global, unsigned int N, unsigned int iterations);
// __device__ void loadDataFromGloablToShared(float* global, float* local);

/*****************************************************************/
/**** Do NOT CHANGE ANYTHING in main() function ******/

int main(int argc, char * argv[])
{
  unsigned int N; /* Dimention of NxN matrix */
  int type_of_device = 0; // CPU or GPU
  int iterations = 0;
  int i;
  
  /* The 2D array of points will be treated as 1D array of NxN elements */
  float * playground; 
  
  // to measure time taken by a specific part of the code 
  double time_taken;
  clock_t start, end;
  
  if(argc != 4)
  {
    fprintf(stderr, "usage: heatdist num  iterations  who\n");
    fprintf(stderr, "num = dimension of the square matrix (50 and up)\n");
    fprintf(stderr, "iterations = number of iterations till stopping (1 and up)\n");
    fprintf(stderr, "who = 0: sequential code on CPU, 1: GPU version, 2: GPU opitmized version\n");
    exit(1);
  }
  
  type_of_device = atoi(argv[3]);
  N = (unsigned int) atoi(argv[1]);
  iterations = (unsigned int) atoi(argv[2]);
 
  
  /* Dynamically allocate NxN array of floats */
  playground = (float *)calloc(N*N, sizeof(float));
  if( !playground )
  {
   fprintf(stderr, " Cannot allocate the %u x %u array\n", N, N);
   exit(1);
  }
  
  /* Initialize it: calloc already initalized everything to 0 */
  // Edge elements  initialization
  for(i = 0; i < N; i++)
    playground[index(0,i,N)] = 100;
  for(i = 0; i < N; i++)
    playground[index(N-1,i,N)] = 150;
  

  switch(type_of_device)
  {
	case 0: printf("CPU sequential version:\n");
			start = clock();
			seq_heat_dist(playground, N, iterations);
			end = clock();
			break;
		
	case 1: printf("GPU version:\n");
			start = clock();
			gpu_heat_dist(playground, N, iterations); 
			end = clock();  
			break;
			
	case 2: printf("GPU optimized version:\n");
			start = clock();
			gpu_optimized_heat_dist(playground, N, iterations); 
			end = clock();  
			break;
			
	default: printf("Invalid device type\n");
			 exit(1);
  }
  
  // printf("Final output:\n");
  // for(int i = 0; i<N; i++)
  // {
  //   for(int j = 0; j<N; j++)
  //   {
  //     printf("%lf ", playground[index(i,j,N)]);
  //   }
  //   printf("\n");
  // }
  time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
  
  printf("Time taken = %lf\n", time_taken);
  
  free(playground);
  
  return 0;

}


/*****************  The CPU sequential version (DO NOT CHANGE THAT) **************/
void  seq_heat_dist(float * playground, unsigned int N, unsigned int iterations)
{
  // Loop indices
  int i, j, k;
  int upper = N-1;
  
  // number of bytes to be copied between array temp and array playground
  unsigned int num_bytes = 0;
  
  float * temp; 
  /* Dynamically allocate another array for temp values */
  /* Dynamically allocate NxN array of floats */
  temp = (float *)calloc(N*N, sizeof(float));
  if( !temp )
  {
   fprintf(stderr, " Cannot allocate temp %u x %u array\n", N, N);
   exit(1);
  }
  
  num_bytes = N*N*sizeof(float);
  
  /* Copy initial array in temp */
  memcpy((void *)temp, (void *) playground, num_bytes);
  
  for( k = 0; k < iterations; k++)
  {
    /* Calculate new values and store them in temp */
    for(i = 1; i < upper; i++)
      for(j = 1; j < upper; j++)
	temp[index(i,j,N)] = (playground[index(i-1,j,N)] + 
	                      playground[index(i+1,j,N)] + 
			      playground[index(i,j-1,N)] + 
			      playground[index(i,j+1,N)])/4.0;
  
			      
   			      
    /* Move new values into old values */ 
    memcpy((void *)playground, (void *) temp, num_bytes);
  }
  
}

/***************** The GPU version: Write your code here *********************/
/* This function can call one or more kernels if you want ********************/
void  gpu_heat_dist(float * playground, unsigned int N, unsigned int iterations)
{
  int k;
  int size = N * N * sizeof(float);

  float *playground_d, *playground_d_last;//, *temp;
  
  cudaError_t err = cudaSuccess;

  err = cudaMalloc((void **)&playground_d, size);
  if(err != cudaSuccess)
  {
    fprintf(stderr, "Error allocating playground_d array on device: %s\n", cudaGetErrorString(err));
    exit(1);
  }

  err = cudaMalloc((void **)&playground_d_last, size);
  if(err != cudaSuccess)
  {
    fprintf(stderr, "Error allocating temp array on device: %s\n", cudaGetErrorString(err));
    exit(1);
  }

  cudaMemcpy(playground_d, playground, size, cudaMemcpyHostToDevice);
  cudaMemcpy(playground_d_last, playground, size, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(N,N);
  dim3 blocksPerGrid(1);
  
  for( k = 0; k < iterations; k++)
  {
    
    updateTemperatures<<<blocksPerGrid,threadsPerBlock>>>(playground_d, playground_d_last, N);
    cudaMemcpy(playground_d_last, playground_d, size, cudaMemcpyDeviceToDevice);
  }
  
  cudaMemcpy(playground, playground_d, size, cudaMemcpyDeviceToHost);

  cudaFree(playground_d);
  cudaFree(playground_d_last);

}

/***************** The GPU optimized version: Write your code here *********************/
/* This function can call one or more kernels if you want ********************/
void  gpu_optimized_heat_dist(float * playground, unsigned int N, unsigned int iterations)
{
  
  int size = N * N * sizeof(float);
  float *playground_d;

  cudaError_t err = cudaSuccess;
  err = cudaMalloc((void **)&playground_d, size);
  if(err != cudaSuccess)
  {
    fprintf(stderr, "Error allocating playground_d array on device: %s\n", cudaGetErrorString(err));
    exit(1);
  }

  cudaMemcpy(playground_d, playground, size, cudaMemcpyHostToDevice);

  // int threadsPerBlock, blocksPerGrid;
  // threadsPerBlock = WIDTH * WIDTH;
  // blocksPerGrid = N*N / threadsPerBlock;

  dim3 block(WIDTH,WIDTH);
	// dim3 grid( (N*N) / (WIDTH * WIDTH));
	dim3 grid( N/WIDTH, N/WIDTH);

  updateTemperaturesOptimized<<<grid,block>>>(playground_d, N, iterations);
    
  cudaMemcpy(playground, playground_d, size, cudaMemcpyDeviceToHost);

  cudaFree(playground_d);
  
}


__global__
void updateTemperatures(float* current, float* last, int N)
{

  int index, top, bottom, left, right;
  // index = threadIdx.x + blockDim.x * blockIdx.x;
  index = threadIdx.y * blockDim.x + threadIdx.x;
  // printf("%d: %lf -> %lf\n", index, last[index], current[index]);
  if (threadIdx.y == blockDim.y-1 
      || threadIdx.y == 0 
      || threadIdx.x == 0 
      || threadIdx.x == blockDim.x-1)
  {
    return;
  }

  // int index, top, bottom, left, right;
  // // index = threadIdx.x + blockDim.x * blockIdx.x;
  // index = threadIdx.y * blockDim.x + threadIdx.x;
  top = (threadIdx.y+1) * blockDim.x + threadIdx.x;
  bottom = (threadIdx.y-1) * blockDim.x + threadIdx.x;
  left = threadIdx.y * blockDim.x + threadIdx.x-1;
  right = threadIdx.y * blockDim.x + threadIdx.x+1;


  
  current[index] = (last[top] + last[bottom] + 
                    last[left] + last[right])/4.0;
  
  // printf("%d: %lf -> %lf\n", index, last[index], current[index]);

}

__global__
void updateTemperaturesOptimized(float* playground_d, unsigned int N, unsigned int iterations)
{

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int row = by * WIDTH + ty;
  int col = bx * WIDTH + tx;
  int globalIndex = row*N + col;

  __shared__ float s_current[WIDTH+2][WIDTH+2];
  // __shared__ float s_temp[WIDTH+2][WIDTH+2];

  // if((bx == 0 && tx == 0) || (bx == N-1 && tx == WIDTH-1)  || (by == 0 && ty == 0) || (by == N-1 && ty == WIDTH-1))
  // if(row == 0 || row == N-1  || col == 0 || col == N-1)
  // {
  //   float value = playground_d[globalIndex];
  //   s_current[ty+1][tx+1] = value;
  //   // s_temp[ty+1][tx+1] = value;
  //   return;
  // }
  

  for(int k = 0; k < iterations; k++)
  {
    float value = playground_d[globalIndex];
    s_current[ty+1][tx+1] = value;
    // s_temp[ty+1][tx+1] = value;

    if (tx == 0 && bx != 0) { s_current[ty+1][tx] = playground_d[row*N + col-1];}
    if(tx == WIDTH-1 && bx != N/WIDTH-1) { s_current[ty+1][tx+1+1] = playground_d[row*N + col+1];}
    if(ty == 0 && by != 0) { s_current[ty][tx+1] = playground_d[(row-1)*N + col];}
    if(ty == WIDTH-1 && by != N/WIDTH-1) { s_current[ty+1+1][tx+1] = playground_d[(row+1)*N + col];}

    __syncthreads();

    playground_d[globalIndex] = (s_current[ty+1][tx+1+1] + 
                      s_current[ty+1][tx-1+1] + 
                      s_current[ty+1+1][tx+1] + 
                      s_current[ty-1+1][tx+1])/4.0;
    
    __syncthreads();

    // playground_d[globalIndex] = s_temp[ty+1][tx+1];
    __syncthreads();
    //  printf("%d: %lf -> %lf/%lf\n", globalIndex, s_current[tx+1][ty+1], playground_d[globalIndex], s_temp[tx+1][ty+1]);

  }

}

