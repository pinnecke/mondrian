#include <Device/Query.cuh>
#include <stdio.h>

#define REQUIRE_SUCCESS(Expression, RetVal)																			\
{																													\
	const cudaError_t error = Expression;																			\
	if (error != cudaSuccess)																						\
	{																												\
		fprintf(stderr, "CUDA ERROR: %s:%d, code: %d\n\treason: %s\n\t%s\t",										\
				__FILE__, __LINE__, error, cudaGetErrorString(error),#Expression);									\
		return RetVal;																								\
	}																												\
}

extern "C"
{

	void *CopyDataToDevice(const void *SourceHost, unsigned long long NumberOfBytes)
	{
		void *deviceDest;
		REQUIRE_SUCCESS(cudaMalloc(&deviceDest, NumberOfBytes), NULL);
		REQUIRE_SUCCESS(cudaMemcpy(deviceDest, SourceHost, NumberOfBytes, cudaMemcpyHostToDevice), NULL);
		cudaDeviceSynchronize();
		return deviceDest;
	}

	bool CopyDataFromDevice(void *DestinationHost, void *SourceDevice, unsigned long long NumberOfBytes)
	{
		REQUIRE_SUCCESS(cudaMemcpy(DestinationHost, SourceDevice, NumberOfBytes, cudaMemcpyDeviceToHost), false);
		return true;
	}

	bool FreeDataInDevice(void *DestinationDevice)
	{
		REQUIRE_SUCCESS(cudaFree(DestinationDevice), false);
		return true;
	}

	__inline__ __device__
	int warpReduceSum(unsigned long long val) {
	  for (int offset = warpSize/2; offset > 0; offset /= 2)
	    val += __shfl_down(val, offset);
	  return val;
	}

	__inline__ __device__
	int warpAllReduceSum(unsigned long long val) {
	  for (int mask = warpSize/2; mask > 0; mask /= 2)
	    val += __shfl_xor(val, mask);
	  return val;
	}

	__inline__ __device__
	int blockReduceSum(unsigned long long val) {

	  static __shared__ int shared[32]; // Shared mem for 32 partial sums
	  int lane = threadIdx.x % warpSize;
	  int wid = threadIdx.x / warpSize;

	  val = warpReduceSum(val);     // Each warp performs partial reduction

	  if (lane==0) shared[wid]=val; // Write reduced value to shared memory

	  __syncthreads();              // Wait for all partial reductions

	  //read from shared memory only if that warp existed
	  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

	  if (wid==0) val = warpReduceSum(val); //Final reduce within first warp

	  return val;
	}

	__global__ void deviceReduceKernel(unsigned long long *in, unsigned long long* out, unsigned long long N) {
		unsigned long long sum = 0;

		 printf("sum...%llu\n", sum);
		 /*
	  //reduce multiple elements per thread
	  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
	       i < N;
	       i += blockDim.x * gridDim.x) {
	    sum += in[i];
	    printf("sum...%llu\n", sum);
	  }
	  sum = blockReduceSum(sum);
	  if (threadIdx.x==0)
	    out[blockIdx.x]=sum;*/
	}

	// see
	// https://github.com/parallel-forall/code-samples/blob/master/posts/parallel_reduction_with_shfl/main.cu
	unsigned long long EvaluateSum(void *DevicePriceColumnHandle, unsigned long long NumberOfItems, bool MultipleThreads)
	{
		if (DevicePriceColumnHandle != NULL) {

			unsigned long long *in, *out;

			int threads = 512;
			int blocks = std::min((int)(NumberOfItems + threads - 1) / threads, 1024);

			cudaMalloc(&out,sizeof(unsigned long long)*1024);

			deviceReduceKernel<<<blocks, threads>>>(in, out, NumberOfItems);
		    deviceReduceKernel<<<1, 1024>>>(out, out, blocks);
		    cudaDeviceSynchronize();

		    unsigned long long sum;
		    cudaMemcpy(&sum,out,sizeof(unsigned long long),cudaMemcpyDeviceToHost);

		    printf("SUMMM: %llu\n", sum);

		    cudaFree(in);
		    cudaFree(out);
		    cudaDeviceReset();

			return sum;
		}
		return 0;
	}

}
