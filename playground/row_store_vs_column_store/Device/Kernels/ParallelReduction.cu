#include <Shared/Types.h>

/*
 * The following reduction kernels are taken from an excellent tutorial by Mark Harris:
 * https://github.com/parallel-forall/code-samples/tree/master/posts/parallel_reduction_with_shfl
 */

__inline__ __device__ int warpReduceSum(size_t val)
{
  for (int offset = warpSize/2; offset > 0; offset /= 2)
	val += __shfl_down(val, offset);
  return val;
}

__inline__ __device__ int warpAllReduceSum(size_t val)
{
  for (int mask = warpSize/2; mask > 0; mask /= 2)
	val += __shfl_xor(val, mask);
  return val;
}

__inline__ __device__ int blockReduceSum(size_t val)
{
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

__global__ void deviceReduceKernel(size_t *in, size_t* out, size_t N)
{
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

///////////////////////////////////////////////////////////////////////////////////////////////////////

__inline__ __device__ PricesRowStoreTuple *warpReduceSum2(PricesRowStoreTuple *val)
{
  for (int offset = warpSize/2; offset > 0; offset /= 2) {
	val->f0 += __shfl_down(val->f0, offset);
	val->f1 += __shfl_down(val->f1, offset);
	val->f2 += __shfl_down(val->f2, offset);
	val->f3 += __shfl_down(val->f3, offset);
	val->f4 += __shfl_down(val->f4, offset);
	val->f5 += __shfl_down(val->f5, offset);
	val->f6 += __shfl_down(val->f6, offset);
	val->f7 += __shfl_down(val->f7, offset);
	val->f8 += __shfl_down(val->f8, offset);
	val->f9 += __shfl_down(val->f9, offset);
  }
  return val;
}

__inline__ __device__ PricesRowStoreTuple *blockReduceSum2(PricesRowStoreTuple *val)
{
  static __shared__ PricesRowStoreTuple shared[32]; // Shared mem for 32 partial sums
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceSum2(val);

  if (lane==0) shared[wid]= *val; // Write reduced value to shared memory

  __syncthreads();              // Wait for all partial reductions

  //read from shared memory only if that warp existed
  val->f0 = (threadIdx.x < blockDim.x / warpSize) ? shared[lane].f0 : 0;
  val->f1 = (threadIdx.x < blockDim.x / warpSize) ? shared[lane].f1 : 0;
  val->f2 = (threadIdx.x < blockDim.x / warpSize) ? shared[lane].f2 : 0;
  val->f3 = (threadIdx.x < blockDim.x / warpSize) ? shared[lane].f3 : 0;
  val->f4 = (threadIdx.x < blockDim.x / warpSize) ? shared[lane].f4 : 0;
  val->f5 = (threadIdx.x < blockDim.x / warpSize) ? shared[lane].f5 : 0;
  val->f6 = (threadIdx.x < blockDim.x / warpSize) ? shared[lane].f6 : 0;
  val->f7 = (threadIdx.x < blockDim.x / warpSize) ? shared[lane].f7 : 0;
  val->f8 = (threadIdx.x < blockDim.x / warpSize) ? shared[lane].f8 : 0;
  val->f9 = (threadIdx.x < blockDim.x / warpSize) ? shared[lane].f9 : 0;

  if (wid==0) {
	  val = warpReduceSum2(val); //Final reduce within first warp
  }

  return val;
}

__global__ void deviceReduceKernel2(PricesRowStoreTuple *in, PricesRowStoreTuple* out, size_t N)
{
	PricesRowStoreTuple sum;

  //reduce multiple elements per thread
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
	sum.f0 += in[i].f0;
	sum.f1 += in[i].f1;
	sum.f2 += in[i].f2;
	sum.f3 += in[i].f3;
	sum.f4 += in[i].f4;
	sum.f5 += in[i].f5;
	sum.f6 += in[i].f6;
	sum.f7 += in[i].f7;
	sum.f8 += in[i].f8;
	sum.f9 += in[i].f9;
  }
  sum = *blockReduceSum2(&sum);
  if (threadIdx.x==0)
	out[blockIdx.x]=sum;
}
