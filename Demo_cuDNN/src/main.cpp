#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cuda.h>
#include <cudnn.h>

using namespace std;

int cudnn_function(void);

int main(){
	
	

	
	
	// call cudnn code
	int isOk = cudnn_function();
	 
    
	return 0;
}
