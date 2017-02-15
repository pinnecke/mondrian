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
		cudaDeviceSynchronize();
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

	DEVICE_MEM_HANDLE *deviceQueryEvaluateManySum(DEVICE_MEM_HANDLE *PriceColumn, size_t NumberOfColumns, size_t NumberOfItems, bool MultipleThreads)
	{
		DEVICE_MEM_HANDLE *out = (DEVICE_MEM_HANDLE *) malloc(NumberOfColumns * sizeof(DEVICE_MEM_HANDLE));

		int threads = MultipleThreads ? 512 : 1024;
		int blocks = MultipleThreads ? std::min((int)(NumberOfItems + threads - 1) / threads, 1024) : 1;

		for (size_t i = 0; i < COLUMN_NUMBER_TO_COPY; ++i) {
			REQUIRED(cudaMalloc(&out[i], sizeof(size_t)*1024), 0);
		}

		for (size_t i = 0; i < COLUMN_NUMBER_TO_COPY; ++i) {
			deviceReduceKernel<<<blocks, threads>>>((size_t*) PriceColumn[i], (size_t*) out[i], NumberOfItems); // TODO
			deviceReduceKernel<<<1, 1024>>>((size_t*) out[i], (size_t*) out[i], blocks); // TODO
		}

		cudaDeviceSynchronize();

		return out;
	}

	size_t *deviceQueryFetchManySumValues(DEVICE_MEM_HANDLE *deviceResultHandle, size_t NumberOfColumns)
	{
		size_t *sum = (size_t *) malloc (sizeof(size_t) * NumberOfColumns);

		for (size_t i = 0; i < COLUMN_NUMBER_TO_COPY; ++i) {
			REQUIRED(cudaMemcpy(&sum[i],deviceResultHandle[i],sizeof(size_t),cudaMemcpyDeviceToHost), 0);
		}
		cudaDeviceSynchronize();

		return sum;
	}

	DEVICE_MEM_HANDLE deviceQueryEvaluateManySumRow(DEVICE_MEM_HANDLE PricesTable, size_t NumberOfItems, bool MultipleThreads)
	{
		DEVICE_MEM_HANDLE out;

		int threads = MultipleThreads ? 512 : 1024;
		int blocks = MultipleThreads ? std::min((int)(NumberOfItems + threads - 1) / threads, 1024) : 1;

		REQUIRED(cudaMalloc(&out, sizeof(PricesRowStoreTuple)*1024), 0);

		deviceReduceKernel2<<<blocks, threads>>>((PricesRowStoreTuple*) PricesTable, (PricesRowStoreTuple*) out, NumberOfItems);
		deviceReduceKernel2<<<1, 1024>>>((PricesRowStoreTuple*) out, (PricesRowStoreTuple*) out, blocks);
		cudaDeviceSynchronize();

		return out;
	}

	PricesRowStoreTuple *deviceQueryFetchManySumValuesRow(DEVICE_MEM_HANDLE RowHandle)
	{
		PricesRowStoreTuple *sum = (PricesRowStoreTuple *) malloc (sizeof(PricesRowStoreTuple));
		REQUIRED(cudaMemcpy(sum,RowHandle,sizeof(PricesRowStoreTuple),cudaMemcpyDeviceToHost), 0);
		return sum;
	}

	void deviceCleanUp(DEVICE_MEM_HANDLE *DevHandle, size_t NumDevHandle)
	{
		for (size_t i = 0; i < NumDevHandle; i++) {
			REQUIRED(cudaFree(DevHandle[i]), );
		}
		cudaDeviceReset();
	}

}
