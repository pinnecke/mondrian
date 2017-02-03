#include <Device/Query.cuh>
#include <stdio.h>

#define cudaCheckError() {                                          \
  cudaError_t e=cudaGetLastError();                                  \
  if(e!=cudaSuccess) {                                               \
  printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
  exit(0); \
  }                                                                  \
}

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
	int warpReduceSum(size_t val) {
	  for (int offset = warpSize/2; offset > 0; offset /= 2)
	    val += __shfl_down(val, offset);
	  return val;
	}

	__inline__ __device__
	int warpAllReduceSum(size_t val) {
	  for (int mask = warpSize/2; mask > 0; mask /= 2)
	    val += __shfl_xor(val, mask);
	  return val;
	}

	__inline__ __device__
	int blockReduceSum(size_t val) {

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

	__global__ void deviceReduceKernel(size_t *in, size_t* out, size_t N) {
		size_t sum = 0;

	  //reduce multiple elements per thread
	  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
	       i < N;
	       i += blockDim.x * gridDim.x) {
	    sum += in[i];
	  }
	  sum = blockReduceSum(sum);
	  if (threadIdx.x==0)
	    out[blockIdx.x]=sum;
	}

	// see
	// https://github.com/parallel-forall/code-samples/blob/master/posts/parallel_reduction_with_shfl/main.cu
	void *EvaluateSum(void *DevicePriceColumnHandle, size_t NumberOfItems, bool MultipleThreads)
	{
		int deviceCount = 0;
		    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

		    if (error_id != cudaSuccess)
		    {
		        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
		        printf("Result = FAIL\n");
		        exit(EXIT_FAILURE);
		    }

		    // This function call returns 0 if there are no CUDA capable devices.
		    if (deviceCount == 0)
		    {
		        printf("There are no available device(s) that support CUDA\n");
		    }
		    else
		    {
		        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
		    }

		    int dev, driverVersion = 0, runtimeVersion = 0;


		        cudaSetDevice(0);



		//if (DevicePriceColumnHandle != NULL) {

		cudaCheckError();

			size_t *out;

			int threads = 512;
			int blocks = std::min((int)(NumberOfItems + threads - 1) / threads, 1024);

			REQUIRE_SUCCESS(cudaMalloc(&out, sizeof(size_t)*1024), 0);

			cudaCheckError();

			deviceReduceKernel<<<blocks, threads>>>((size_t*) DevicePriceColumnHandle, out, NumberOfItems);
		    deviceReduceKernel<<<1, 1024>>>(out, out, blocks);
		    cudaDeviceSynchronize();

		   // size_t sum;
		   // cudaMemcpy(&sum,out,sizeof(size_t),cudaMemcpyDeviceToHost);

		    //printf("Summe returned: %zu\n", sum);

		    return out;

		//	return sum;
		//}
		//return 0;
	}

	size_t GetSumValue(void *deviceResultHandle) {
		size_t sum;
		cudaMemcpy(&sum,deviceResultHandle,sizeof(size_t),cudaMemcpyDeviceToHost);
		return sum;
	}

	void cleanUp(void *DevicePriceColumnHandle, void *deviceResultHandle) {
		cudaFree(DevicePriceColumnHandle);
		cudaFree(deviceResultHandle);
		cudaDeviceReset();
	}

}
