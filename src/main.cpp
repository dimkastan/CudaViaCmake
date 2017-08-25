#include "cuda_runtime.h"
#include <stdio.h>
#include <cuda.h>

using namespace std;

__global__ void test(float *a, float *b, float *c, int N);

int main(){
int isOk=0;
	int N =2;
	int numBytes  =2*sizeof(float);
	// host
	float a[2]={1,2};
	float b[2]={1,2};
	float c[2]={0,0};
	
	//device
	float *d_a=NULL;
	float *d_b=NULL;
	float *d_c=NULL;
	
	cudaMalloc((void**)&d_a,numBytes);
	cudaMalloc((void**)&d_b,numBytes);
	cudaMalloc((void**)&d_c,numBytes);
	cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
	        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
	                 __FILE__, __LINE__, cudaGetErrorString( err ) );
	        return( -1 );
	}
	
	// ----
	// send data to gpu
	// ----
	cudaMemcpy(d_a,a,numBytes,cudaMemcpyHostToDevice);
	cudaMemcpy(d_b,b,numBytes,cudaMemcpyHostToDevice);
  	err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
	        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
	                 __FILE__, __LINE__, cudaGetErrorString( err ) );
	        return( -1 );
	}
	// call kernel
	test<< <2,1>> >(d_a,d_b,d_c,N); // 2 blocks, 1 thread per block
	err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
	        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
	                 __FILE__, __LINE__, cudaGetErrorString( err ) );
	        return( -1 );
	} 
    // ----
    // read data back to host
	// ----
    cudaMemcpy(c, d_c, numBytes, cudaMemcpyDeviceToHost);
    err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
	        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
	                 __FILE__, __LINE__, cudaGetErrorString( err ) );
	        return ( -1 );
	} 
 
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
	
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
    err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
	        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
	                 __FILE__, __LINE__, cudaGetErrorString( err ) );
	        return( -1 );
	}
	return 0;
}
