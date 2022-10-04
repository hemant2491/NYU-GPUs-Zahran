#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#define RANGE 19.87

/*** TODO: insert the declaration of the kernel function below this line ***/
__global__
void vecGPU(float* ad, float* bd, float* cd, int n, int threadsPerBlock);

/**** end of the kernel declaration ***/


int main(int argc, char *argv[]){

	int n = 0; //number of elements in the arrays
	int i;  //loop index
	float *a, *b, *c; // The arrays that will be processed in the host.
	float *temp;  //array in host used in the sequential code.
	float *ad, *bd, *cd; //The arrays that will be processed in the device.
	clock_t start, end; // to meaure the time taken by a specific part of code
	
	if(argc != 2){
		printf("usage:  ./vectorprog n\n");
		printf("n = number of elements in each vector\n");
		exit(1);
		}
		
	n = atoi(argv[1]);
	printf("Each vector will have %d elements\n", n);
	
	
	//Allocating the arrays in the host
	
	if( !(a = (float *)malloc(n*sizeof(float))) )
	{
	   printf("Error allocating array a\n");
	   exit(1);
	}
	
	if( !(b = (float *)malloc(n*sizeof(float))) )
	{
	   printf("Error allocating array b\n");
	   exit(1);
	}
	
	if( !(c = (float *)malloc(n*sizeof(float))) )
	{
	   printf("Error allocating array c\n");
	   exit(1);
	}
	
	if( !(temp = (float *)malloc(n*sizeof(float))) )
	{
	   printf("Error allocating array temp\n");
	   exit(1);
	}
	
	//Fill out the arrays with random numbers between 0 and RANGE;
	srand((unsigned int)time(NULL));
	for (i = 0; i < n;  i++){
        a[i] = ((float)rand()/(float)(RAND_MAX)) * RANGE;
		b[i] = ((float)rand()/(float)(RAND_MAX)) * RANGE;
		c[i] = ((float)rand()/(float)(RAND_MAX)) * RANGE;
		temp[i] = c[i]; //temp is just another copy of C
	}
	
    //The sequential part
	start = clock();
	for(i = 0; i < n; i++)
		temp[i] += a[i] * b[i];
	end = clock();
	printf("Total time taken by the sequential part = %lf\n", (double)(end - start) / CLOCKS_PER_SEC);

    /******************  The start GPU part: Do not modify anything in main() above this line  ************/
	//The GPU part
	
	/* TODO: in this part you need to do the following:
		1. allocate ad, bd, and cd in the device
		2. send a, b, and c to the device  
		*/

	    int size = n * sizeof(float);
		cudaError_t err = cudaSuccess;

	    err = cudaMalloc((void **)&ad, size);
		if(err != cudaSuccess)
		{
			fprintf(stderr, "Error allocating array ad on device: %s\n", cudaGetErrorString(err));
			exit(1);
		}

		err = cudaMalloc((void **)&bd, size);
		if(err != cudaSuccess)
		{
			fprintf(stderr, "Error allocating array bd on device: %s\n", cudaGetErrorString(err));
			exit(1);
		}

		err = cudaMalloc((void **)&cd, size);
		if(err != cudaSuccess)
		{
			fprintf(stderr, "Error allocating array cd on device: %s\n", cudaGetErrorString(err));
			exit(1);
		}

		cudaMemcpy(ad, a, size, cudaMemcpyHostToDevice);
		cudaMemcpy(bd, b, size, cudaMemcpyHostToDevice);
		cudaMemcpy(cd, c, size, cudaMemcpyHostToDevice);
		
	/* TODO: 	
		3. write the kernel, call it: vecGPU
		4. call the kernel (the kernel itself will be written at the comment at the end of this file), 
		   you need to decide about the number of threads, blocks, etc and their geometry.
		*/
		int threadsPerBlock, blocksPerGrid;
		// 4 blocks and 500 threads per block
		blocksPerGrid = 4;
		threadsPerBlock = 500;
		// // 8 blocks and 500 threads per block
		// blocksPerGrid = 4;
		// threadsPerBlock = 500;
		// // 16 blocks and 500 blocks per block
		// blocksPerGrid = 4;
		// threadsPerBlock = 500;
		// // 4 blocks and 250 threads per block
		// blocksPerGrid = 4;
		// threadsPerBlock = 500;
		// // 8 blocks and 250 threads per block
		// blocksPerGrid = 4;
		// threadsPerBlock = 500;
		// // 16 blocks and 250 blocks per block
		// blocksPerGrid = 4;
		// threadsPerBlock = 500;

		// dim3 block(threadsPerBlock);
		// dim3 grid(blocksPerGrid);
		vecGPU<<<blocksPerGrid,threadsPerBlock>>>(ad, bd, cd, n, blocksPerGrid*threadsPerBlock);
		end = clock();
	/* TODO: 
		5. bring the cd array back from the device and store it in c array (declared earlier in main)
		6. free ad, bd, and cd
	*/
        cudaMemcpy(c, cd, size, cudaMemcpyDeviceToHost);

		cudaFree(ad);
		cudaFree(bd);
		cudaFree(cd);
	
	printf("Total time taken by the GPU part = %lf\n", (double)(end - start) / CLOCKS_PER_SEC);
	/******************  The end of the GPU part: Do not modify anything in main() below this line  ************/
	
	int wrongCount = 0;
	//checking the correctness of the GPU part
	for(i = 0; i < n; i++)
	{
		if( fabs(temp[i] - c[i]) >= 0.009) //compare up to the second degit in floating point
		{
			printf("Element %d in the result array does not match the sequential version\n", i);
			// printf("%d:\ta %.2lf\tb %.2lf\tc\tsequential %.2lf\tGPU %.2lf\n\n", i, a[i], b[i], temp[i], c[i]);
			wrongCount++;
		}
	}

	printf("Number of c values:\twrong %d\tcorrect %d\n", wrongCount, n-wrongCount);
		
	// Free the arrays in the host
	free(a); free(b); free(c); free(temp);

	return 0;
}


/**** TODO: Write the kernel itself below this line *****/

__global__
void vecGPU(float* ad, float* bd, float* cd, int n, int totalNumOfThreads)
{
    // int totalNumOfThreads = blockDim.x * gridDim.x;
	int i, j, index;
	j = threadIdx.x + blockDim.x * blockIdx.x;
	for (i = 1; i <= (n/totalNumOfThreads + 1); i++)
	{
		index = i * j;
		if(index<n)
		{
			cd[index] += ad[index] * bd[index];
		}
	}
}

