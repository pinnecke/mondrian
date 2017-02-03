#include <Device/Query.cuh>
#include "Kernels/ParallelReduction.cu"

#define REQUIRED(Expression, RetVal)																				\
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
	DEVICE_MEM_HANDLE deviceCopyFromHostToDevice(const HOST_MEM_POINTER Source, size_t NumberOfBytes)
	{
		DEVICE_MEM_HANDLE deviceMemoryHandle;
		REQUIRED(cudaMalloc(&deviceMemoryHandle, NumberOfBytes), NULL);
		REQUIRED(cudaMemcpy(deviceMemoryHandle, Source, NumberOfBytes, cudaMemcpyHostToDevice), NULL);
		cudaDeviceSynchronize();
		return deviceMemoryHandle;
	}

	bool deviceCopyFromDeviceToHost(HOST_MEM_POINTER Destination, DEVICE_MEM_HANDLE Source, size_t NumberOfBytes)
	{
		REQUIRED(cudaMemcpy(Destination, Source, NumberOfBytes, cudaMemcpyDeviceToHost), false);
		return true;
	}


	DEVICE_MEM_HANDLE deviceQueryEvaluateSum(DEVICE_MEM_HANDLE PriceColumn, size_t NumberOfItems, bool MultipleThreads)
	{
		DEVICE_MEM_HANDLE out;

		int threads = MultipleThreads ? 512 : 1024;
		int blocks = MultipleThreads ? std::min((int)(NumberOfItems + threads - 1) / threads, 1024) : 1;

		REQUIRED(cudaMalloc(&out, sizeof(size_t)*1024), 0);

		deviceReduceKernel<<<blocks, threads>>>((size_t*) PriceColumn, (size_t*) out, NumberOfItems);
	    deviceReduceKernel<<<1, 1024>>>((size_t*) out, (size_t*) out, blocks);
	    cudaDeviceSynchronize();

	    return out;
	}

	size_t deviceQueryFetchSumValue(DEVICE_MEM_HANDLE deviceResultHandle)
	{
		size_t sum;
		REQUIRED(cudaMemcpy(&sum,deviceResultHandle,sizeof(size_t),cudaMemcpyDeviceToHost), 0);
		return sum;
	}

	void deviceCleanUp(DEVICE_MEM_HANDLE DevicePriceColumnHandle, DEVICE_MEM_HANDLE deviceResultHandle)
	{
		cudaFree(DevicePriceColumnHandle);
		cudaFree(deviceResultHandle);
		cudaDeviceReset();
	}

}
