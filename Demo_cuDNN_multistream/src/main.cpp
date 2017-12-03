#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cuda.h>
#include <cudnn.h>

using namespace std;

int cudnn_function(cudnnHandle_t &cudnnHandle, cudaStream_t &cudaStream );

int main(){
	
	
	cudnnHandle_t * cudnnhandle  = new cudnnHandle_t[8];
	cudaStream_t *  cudaStream   = new cudaStream_t[8];

	for(int i=0;i<8;i++){
	cudnnCreate(&cudnnhandle[i] );
	cudaStreamCreate(&cudaStream[i] );

     }

	
	for(int i=0;i<8;i++){
		// call cudnn code
		int isOk = cudnn_function(cudnnhandle[0],cudaStream[0]);


    }

    cudaDeviceSynchronize();

   for(int i=0;i<8;i++){
		// call cudnn code
		int isOk = cudnn_function(cudnnhandle[i],cudaStream[i]);


    }

    cudaDeviceSynchronize();
	

	for(int i=0;i<8;i++){
	cudnnDestroy(cudnnhandle[i] );
	cudaStreamDestroy(cudaStream[i] );

    } 
	 
    
	return 0;
}
