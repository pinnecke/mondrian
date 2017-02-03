#include <Device/Query.cuh>
#include <stdio.h>

#define REQUIRE_SUCCESS(Expression) 																				\
{																													\
	const cudaError_t error = Expression;																			\
	if (error != cudaSuccess)																						\
	{																												\
		fprintf(stderr, "CUDA ERROR: %s:%d, code: %d, reason: %s\n",												\
				__FILE__, __LINE__, error, cudaGetErrorString(error));												\
		return false;																								\
	}																												\
}

extern "C"
{

	const int N = 16;
	const int blocksize = 16;

	__global__
	void hello(char *a, int *b)
	{
		a[threadIdx.x] += b[threadIdx.x];
	}

	bool CopyDataToDevice(void **DestinationDevice, const void *SourceHost, size_t NumberOfBytes)
	{
		REQUIRE_SUCCESS(cudaMalloc(DestinationDevice, NumberOfBytes));
		REQUIRE_SUCCESS(cudaMemcpy(DestinationDevice, SourceHost, NumberOfBytes, cudaMemcpyHostToDevice));
		return true;
	}

	bool CopyDataFromDevice(void **DestinationHost, const void *SourceDevice, size_t NumberOfBytes)
	{
		REQUIRE_SUCCESS(cudaMemcpy(DestinationHost, SourceDevice, NumberOfBytes, cudaMemcpyDeviceToDevice));
		return true;
	}

	bool FreeDataInDevice(void *DestinationDevice)
	{
		REQUIRE_SUCCESS(cudaFree(DestinationDevice));
		return true;
	}


	void do_cuda_stuff()
	{
		char a[N] = "Hello \0\0\0\0\0\0";
		int b[N] = {15, 10, 6, 0, -11, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

		char *ad;
		int *bd;
		const int csize = N*sizeof(char);
		const int isize = N*sizeof(int);

		printf("%s", a);

		cudaMalloc( (void**)&ad, csize );
		cudaMalloc( (void**)&bd, isize );
		cudaMemcpy( ad, a, csize, cudaMemcpyHostToDevice );
		cudaMemcpy( bd, b, isize, cudaMemcpyHostToDevice );

		dim3 dimBlock( blocksize, 1 );
		dim3 dimGrid( 1, 1 );
		hello<<<dimGrid, dimBlock>>>(ad, bd);
		cudaMemcpy( a, ad, csize, cudaMemcpyDeviceToHost );
		cudaFree( ad );
		cudaFree( bd );

		printf("%s\n", a);
	}

}
