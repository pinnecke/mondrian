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
