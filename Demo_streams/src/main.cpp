#include "cuda_runtime.h"
#include <stdio.h>
#include <cuda.h>

// Stream demo based on NVIDIA's example https://devblogs.nvidia.com/parallelforall/gpu-pro-tip-cuda-7-streams-simplify-concurrency/ 
// Please refer to this page for more information
//
//
using namespace std;

const int N = 1000;

__global__ void kernel(float *x, int n);

int main()
{
    const int num_streams = 8;
    cudaStream_t streams[num_streams];
    float *data[num_streams];


    for (int i = 0; i < num_streams; i++) {
 
 
        cudaMalloc(&data[i], N * sizeof(float)); //blocking.
        
        // launch one worker kernel per stream
        kernel<<<1, 64, 0 >>>(data[i], N);
 
    }
    cudaDeviceSynchronize();




    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
 
        cudaMalloc(&data[i], N * sizeof(float)); //blocking.
        
        // launch one worker kernel per stream
        kernel<<<1, 64, 0, streams[i]>>>(data[i], N);

        // launch a dummy kernel on the default stream
        kernel<<<1, 1>>>(0, 0);
    }

    cudaDeviceSynchronize();

    for (int i = 0; i < num_streams; i++) {
 
 
        cudaMalloc(&data[i], N * sizeof(float)); //blocking.
        
        // launch one worker kernel per stream
        kernel<<<1, 64, 0 >>>(data[i], N);
 
    }

    cudaDeviceSynchronize();

    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
 
        cudaMalloc(&data[i], N * sizeof(float)); //blocking.
        
        // launch one worker kernel per stream
        kernel<<<1, 64, 0, streams[i]>>>(data[i], N);

        // launch a dummy kernel on the default stream
        kernel<<<1, 1>>>(0, 0);
    }

    cudaDeviceSynchronize();



    cudaDeviceReset();

    return 0;
}
