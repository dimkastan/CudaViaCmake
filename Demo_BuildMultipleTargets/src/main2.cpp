#include "cuda_runtime.h"
#include <stdio.h>
#include <cuda.h>

using namespace std;

__global__ void test(float *a, float *b, float *c, int N);

int main(){
	int isOk=0;
	cudaError err=cudaSuccess;
	int N =2;
	int numBytes  =2*sizeof(float);

	printf("main2\n");
	// host
	float a[2]={1,2};
	float b[2]={1,2};
	float c[2]={0,0};
	
	//device
	float *d_a=NULL;
	float *d_b=NULL;
	float *d_c=NULL;
	
	err = cudaMalloc((void**)&d_a,numBytes);  if (err != cudaSuccess){ printf("[%s, %d]: %s\n", __FILE__, __LINE__,cudaGetErrorString(err));  return -1;}
	err = cudaMalloc((void**)&d_b,numBytes);  if (err != cudaSuccess){ printf("[%s, %d]: %s\n", __FILE__, __LINE__,cudaGetErrorString(err));  return -1;}
	err = cudaMalloc((void**)&d_c,numBytes);  if (err != cudaSuccess){ printf("[%s, %d]: %s\n", __FILE__, __LINE__,cudaGetErrorString(err));  return -1;}
   
	
	// ----
	// send data to gpu
	// ----
	err =cudaMemcpy(d_a,a,numBytes,cudaMemcpyHostToDevice);  if (err != cudaSuccess){ printf("[%s, %d]: %s\n", __FILE__, __LINE__,cudaGetErrorString(err));  return -1;}
	err =cudaMemcpy(d_b,b,numBytes,cudaMemcpyHostToDevice);  if (err != cudaSuccess){ printf("[%s, %d]: %s\n", __FILE__, __LINE__,cudaGetErrorString(err));  return -1;}
 

	// call kernel
	test<< <2,1>> >(d_a,d_b,d_c,N); // 2 blocks, 1 thread per block
	err = cudaGetLastError();
	if (err != cudaSuccess) {printf("[%s, %d]: %s\n", __FILE__, __LINE__,cudaGetErrorString(err));  return -1;}
     
       // ----
       // read data back to host
       // ----
       err= cudaMemcpy(c, d_c, numBytes, cudaMemcpyDeviceToHost);
       if (err != cudaSuccess){ printf("[%s, %d]: %s\n", __FILE__, __LINE__,cudaGetErrorString(err));  return -1;}
 
	printf("Checking Results\n");
	// check results
	for(int i=0;i<2;i++)
	{
		if(c[i]!=(a[i]*b[i]))
		{
			isOk=1;
			printf("Error: Value at position %d Value %f does not match expected %f\n",i,c[i],a[i]*b[i]);
			
		}
		 
	}
	
	if(isOk==0)
	{
		printf("Results verified\n");
	}
	// check results
	for(int i=0;i<2;i++)
	{
		printf("c[%d]=%f (== %f * %f)\n",i,c[i],a[i],b[i]);
	}
	
	err = cudaFree(d_a); if (err != cudaSuccess){ printf("[%s, %d]: %s\n", __FILE__, __LINE__,cudaGetErrorString(err));  return -1;}
	err = cudaFree(d_b); if (err != cudaSuccess){ printf("[%s, %d]: %s\n", __FILE__, __LINE__,cudaGetErrorString(err));  return -1;}
	err = cudaFree(d_c); if (err != cudaSuccess){ printf("[%s, %d]: %s\n", __FILE__, __LINE__,cudaGetErrorString(err));  return -1;}
    
	return 0;
}
