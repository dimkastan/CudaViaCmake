
__global__ void test(float *a, float *b, float *c, int N)
{
	 
if(blockIdx.x<N)
	c[blockIdx.x] = a[blockIdx.x]*b[blockIdx.x];
return;

}