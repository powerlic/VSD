#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#include <driver_types.h>  // cuda driver types
#include "curand_kernel.h"
#include "device_launch_parameters.h"

#pragma comment(lib,"cuda.lib")
#pragma comment(lib,"cublas.lib")
#pragma comment(lib,"cudart.lib")
#pragma comment(lib,"curand.lib")

#include <boost/thread.hpp>


#  define CUDA_SAFE_CALL_NO_SYNC( call) {                                    \
		cudaError err = call;                                                    \
		if( cudaSuccess != err) {                                                \
		fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
							    }}


#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
      } while (0)


#define CUBLAS_CHECK(condition) \
  do { \
    cublasStatus_t status = condition; \
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << " " \
      << cublasGetErrorString(status); \
      } while (0)

#define CURAND_CHECK(call)  {                                    \
		curandStatus_t err = call;                                                    \
		if( CURAND_STATUS_SUCCESS != err) {                                                \
		fprintf(stderr, "Curand error in file '%s' in line %i.\n",        \
                __FILE__, __LINE__);              \
        exit(EXIT_FAILURE);                                                  \
							    }}

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())





const char* curandGetErrorString(curandStatus_t error);
const char* cublasGetErrorString(cublasStatus_t error);





namespace VasGpuBoost
{
	const int CUDA_NUM_THREADS = 512;
	// CUDA: number of blocks for threads.
	inline int GET_BLOCKS(const int N)
	{
		return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
	}


	class GpuRandGenerator
	{
		public:
			static GpuRandGenerator* Get()
			{
				static GpuRandGenerator instance_;
				return &instance_;
			}
			
			float* GetGpuUnifromRand(int count);
		private:
			GpuRandGenerator();
			~GpuRandGenerator();
			void GenUnifrom(int count, float *d_dst);
			curandGenerator_t gen_;
			boost::mutex gen_mutex_;
			int d_size_=0;
			float *d_rand_=NULL;
			bool used_ = true;

	};

	



}

