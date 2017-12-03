#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>


#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cuda.h>
#include <cudnn.h>

/*
Description:
 
 This is a self contained demo, showing the usage of cuDNN Application Programming Interface (API)
 For simplicity in [1], we define the size of a convolution filter (width, height and number of filters) as well as the size of the image (batch size, input channels, width, height)
 and next we allocate all cudnn-specific tensors and allocate cuda memory in order to perform forward passes.

Currently it is not tested, but is should be ok once you add artificial data inside weights and image.

Author: Dimitris Kastaniotis, November 2017
 
*/

 
using namespace std;
 
 
int ITERS = 1000;  // Measure time as an average of 1000 Iterations.

	
cudnnStatus_t err;
#define CHECK_CUDNN(err) if ((err) != CUDNN_STATUS_SUCCESS) {  printf("[%s: %i]: CUDNN Error: %d %s\n",__FILE__,__LINE__, err, printf("Error code: %d\n",err)); return -1; }

int cudnn_function(cudnnHandle_t &cudnnHandle, cudaStream_t &cudaStream ){
	
	int kernel_size;                         //holds the  size of the convolution kernel (only symmetric kernels are supported here)
	int NumberOfOutputFilters;               // holds the number of output channels
	int NumberOfInputFilters;                // holds the number of input channels
	int stepsize;                            // holds the step size
	int padd;                                // holds the padding size
	int dilation_h, dilation_w;              // holds the dlation size
	
	int    in_width=0;              // input image/feature map width
	int    in_height=0;             // input image/feature map height
	int    out_height=0;            // output image/feature map width
	int    out_width=0;             // output image/feature map height
	int    in_channels=0;           // input image/feature map channels
	int    out_channels=0;          // output image/feature map channels
	int    BatchSize=0;             // input-output batch size
	float *dataout=0;               // pointer to output data (CPU)
	float *datain=0;                // pointer to input data ( CPU)
	float *dout=0;                  // pointer to output data ( GPU)
	float *din=0;                   // pointer to input data ( GPU)
	float *h_convWeights=0;         // pointer to filter data ( CPU)
	
	


	// optional bias
	cudnnTensorDescriptor_t inTensor, outTensor, outTensorR, BiasTensor; // cudnn specific Tensors
	cudnnFilterDescriptor_t FilterDesc;                                  // cudnn Specific Descriptors

	cudnnConvolutionDescriptor_t convDesc;                               // cudnn Specific Descriptor
	cudnnConvolutionFwdAlgo_t convAlgo;                                  // cudnn Specific variable for algorithm type

	// a handle
	// cudnnHandle_t   cudnnHandle;                                         // pointer  to a cudnn Handle
	
	size_t workspaceSize;                                               // workspace size used by the operations



	float *convWeights  ;                                                 // pointers to filter weights (GPU)
	void *workspace = nullptr;                                            // pointers to cudnn workspaceSize
	
	
	// set to arbitrary large value
	size_t workspace_limit_bytes = 32*1024*1024;
	
	//-----------------------------------------------
	//						 print cudnn version
	//-----------------------------------------------
	
	printf("cudnn version: %u\n", cudnnGetVersion());



	//--------------------------------------------------------
	//          cudnn set stream
	//--------------------------------------------------------
	cudnnSetStream( cudnnHandle, cudaStream);
	
	//-----------------------------------------------
	// [1] Setting filter sizes. Assume that we have an input image of three channels, batch size 1, and a 128 filters
	//-----------------------------------------------
	BatchSize    =1;
    in_channels  =3;
	out_channels =128; // assume that we have 128 filters of 3 x 3 size. 
	in_height    =224;
	in_width     = 224;
	kernel_size = 3;  // assume that filter width ==height
	stepsize    = 1;
	padd        = 0;
	dilation_h  = 1;
	dilation_w  = 1;
	
	

	
	
	

	//--------------------------------------------------------------------
	//                Setup Tensors, Filter etc. Create and initialize
	//--------------------------------------------------------------------



 
	CHECK_CUDNN(cudnnCreateTensorDescriptor(&inTensor));
 
	CHECK_CUDNN(cudnnSetTensor4dDescriptor(inTensor,
		CUDNN_TENSOR_NCHW,
		CUDNN_DATA_FLOAT,
		BatchSize, in_channels,
		in_height, in_width));
	//}
	CHECK_CUDNN(cudnnCreateFilterDescriptor(&FilterDesc));
	// set dimensions
	CHECK_CUDNN(cudnnSetFilter4dDescriptor(FilterDesc,
		CUDNN_DATA_FLOAT,
		CUDNN_TENSOR_NCHW,
		out_channels, // number of filters
		in_channels,
		kernel_size,
		kernel_size));

	CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
	// set dimensions
	CHECK_CUDNN(cudnnSetConvolution2dDescriptor(convDesc,
		padd, padd,
		stepsize, stepsize,         // stride
		dilation_h, dilation_w,     // These variables define dilation. supported from Cudnn 6.0 and later
		CUDNN_CROSS_CORRELATION,    //CUDNN_CONVOLUTION
		CUDNN_DATA_FLOAT       ));  // operation mode- precision 
	
	// Find dimension of convolution output
	CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc,
		inTensor,
		FilterDesc,
		&BatchSize, &out_channels, &out_height, &out_width));

	std::cout << "output width" << out_width << "output height" << out_height << std::endl;

 

	// Output Tensor
	CHECK_CUDNN(cudnnCreateTensorDescriptor(&outTensor));
	// set dimensions [Library Version]
	CHECK_CUDNN(cudnnSetTensor4dDescriptor(outTensor,
		CUDNN_TENSOR_NCHW,
		CUDNN_DATA_FLOAT,
		BatchSize, out_channels,
		out_height, out_width));
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
		inTensor,
		FilterDesc,
		convDesc,
		outTensor,
		CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, //CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, //CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, //CUDNN_CONVOLUTION_FWD_NO_WORKSPACE, //CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
		workspace_limit_bytes,
		&convAlgo));
 
	// For that given algorithm  find the WorkSpace Size in bytes
	// In general we set the size as the maximum needed across operations
	CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
		inTensor,
		FilterDesc,
		convDesc,
		outTensor,
		convAlgo,
		&workspaceSize));
 
		std::cout << "Workspace size in bytes is " << workspaceSize << std::endl;
 
 
 // Note on memoty allocation. Recently, cudnn introduced managed memory allocation. this provides flexibility as it follows the shared memory model
 // however it is expected to have worst performance than custom mallocs

	// Allocate space for input output, convolutional filters and bias
	cudaMalloc(&din, sizeof(float) * BatchSize * in_channels* in_height *in_width); // data in
	 
	
	 cudaMalloc(&dout, sizeof(float) * BatchSize* out_channels * out_height * out_width); // data out (after conv)
	 cudaMalloc(&convWeights, sizeof(float) * BatchSize* in_channels* out_channels * kernel_size * kernel_size); // filter coeffs
 	 
	if (workspaceSize > 0)
		 (cudaMalloc(&workspace, workspaceSize)); 
  
    // Allocate space on host. cudaMallocHost takes a double pointer and allocates pinned memory.
	 cudaMallocHost(&datain, in_channels*in_width*in_height*sizeof(float));
	 cudaMallocHost(&dataout, out_channels*out_width*out_height*sizeof(float));
	 cudaMallocHost(&h_convWeights, BatchSize* in_channels* out_channels * kernel_size * kernel_size*sizeof(float));
	 
	 
	 
	 //-----------------------------------------------------------
	 // TODO: Load data to datain from DISK and then copy them to din.
	 // TODO: Load filter weights to h_convWeights from DISK and then copy them to convWeights.
	 // din is the cuda memory and what we want to do is first load data into cpu memory (datain) and then copy them to cuda (din)
	 // This memcpy MUST be performed with cudaMemcpy
	 // ADD your code here to load data from CPU to cuda
	 //
	 // FREAD == > h_convWeights and then cudaMemcpy (h_convWeights ==> convWeights)
	 // FREAD == > datain and then cudaMemcpy (datain ==> din)
	 // 
	 //
	
 
		//create timer.
		std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

          //The following operation is a BLOCKING OPERATION. therefore, you can use cpu timer. However, it is more accurate to use cuda time (event)		 
		for (int i = 0; i < ITERS; i++) {
			
			const float alpha1 = 1.0f, beta1 = 1.0f, beta0 = 0.0f;

		 
		CHECK_CUDNN(cudnnConvolutionForward(cudnnHandle,
			(void*)&alpha1,
			inTensor, (void*)din,  //data (Tensor and pointer to GPU array)
			FilterDesc, (void*)convWeights,   //Filter (Descriptor and pointer to GPU array)
			convDesc,                 //Convolution (Descriptor)
			convAlgo,
			workspace,
			workspaceSize,
			(void*)&beta0,
			outTensor, (void*)dout));
 
		}

		std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
		printf("Time: %2.3f\n",
			double(std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count())
			/ (1e6 * float(ITERS))
		);
 
 
	 
 
	return 0;
}
 