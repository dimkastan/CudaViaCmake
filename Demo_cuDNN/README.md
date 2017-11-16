# CudaViaCmake

Example project for cuDNN setup using cmake

# Build project

You can build and run the application using the following command line script:

    
	build_script.cmd



# Notes
## Build and run cuDNN sample test on windows

<pre>
TODO List: Add random image and weights and verify the result with c++ code. 
Prerequisites: 
System configuration: cudnn 7.0, cuda 8.0 on Windows 10
Cmake 3.6 or greater should be installed. 

1.	Notes for cudnn

Here I provide a brief description of cudnn.
cuDNN is an API for performing operations like convolution, pooling, soft-max etc. These operations usually follow a common procedure which is described below.

1.	Creating Tensor objects: Creating Tensor objects and then setting them appropriately. These tensor objects define the input/output sizes, order of the data and data type. These are used by the operation (e.g. convolution) in order to define the ranges etc. 
Data descriptor (input and output tensor)
CHECK_CUDA(cudnnCreateTensorDescriptor(&inTensor));
 
	CHECK_CUDA(cudnnSetTensor4dDescriptor(inTensor,
		CUDNN_TENSOR_NCHW,
		CUDNN_DATA_FLOAT,
		BatchSize, in_channels,
		in_height, in_width));

	// Output Tensor
	CHECK_CUDA(cudnnCreateTensorDescriptor(&outTensor));
	// set dimensions [Library Version]
	CHECK_CUDA(cudnnSetTensor4dDescriptor(outTensor,
		CUDNN_TENSOR_NCHW,
		CUDNN_DATA_FLOAT,
		BatchSize, out_channels,
		out_height, out_width));

Filter
	//}
	CHECK_CUDA(cudnnCreateFilterDescriptor(&FilterDesc));
	// set dimensions
	CHECK_CUDA(cudnnSetFilter4dDescriptor(FilterDesc,
		CUDNN_DATA_FLOAT,
		CUDNN_TENSOR_NCHW,
		out_channels, // number of filters
		in_channels,
		kernel_size,
		kernel_size));

2.	Allocating cuda device memory: These Tensor Objects however, are only descriptors of the input data format. Therefore, the user should also allocate the memory on CUDA device. This memory might contain data or coefficients. For example convolution operations require three tensors, input, output and filter. Therefore, three cuda buffers should be provided.
3.	Create Descriptor operation: Accordingly the user has create and define the operation.

	CHECK_CUDA(cudnnCreateConvolutionDescriptor(&convDesc));
	// set dimensions
	CHECK_CUDA(cudnnSetConvolution2dDescriptor(convDesc,
		padd, padd,
		stepsize, stepsize,         // stride
		dilation_h, dilation_w,     // These variables define dilation. supported from Cudnn 6.0 and later
		CUDNN_CROSS_CORRELATION,    //or CUDNN_CONVOLUTION
		CUDNN_DATA_FLOAT       ));  // operation mode- precision

Here it is important to understand that in cudnn we select the algorithm manually or automatically. Here I provide only the manual selection:
    CHECK_CUDA(cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
		inTensor,
		FilterDesc,
		convDesc,
		outTensor,
		CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 
		workspace_limit_bytes,
		&convAlgo));


IMPORTANT: CUDNN_CONVOLUTION_FWD_PREFER_FASTEST might return an algorithm that requires some memory (e.g to replicate data, unroll patches etc.) In this case you MUST allocate workspace which HAS TO BE provided in forward pass.
4.	Perform forward pass: This is where the computation takes place.
 
CHECK_CUDA(cudnnConvolutionForward(cudnnHandle,
			(void*)&alpha1,
			inTensor, (void*)din,  //data (Tensor and pointer to GPU array)
			FilterDesc, (void*)convWeights,   //Filter (Descriptor and pointer to GPU array)
			convDesc,                 //Convolution (Descriptor)
			convAlgo,
			workspace, //memory allocated by the user
			workspaceSize,
			(void*)&beta0,
			outTensor, (void*)dout));


</pre>


















 