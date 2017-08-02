#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#include <driver_types.h>  // cuda driver types
#include "curand_kernel.h"
#include "device_launch_parameters.h"


#include <cvpreload.h>

#include <boost/thread.hpp>

#pragma comment(lib,"cuda.lib")
#pragma comment(lib,"cublas.lib")
#pragma comment(lib,"cudart.lib")
#pragma comment(lib,"curand.lib")


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
      << caffe::cublasGetErrorString(status); \
    } while (0)

#define CURAND_CHECK(condition) \
  do { \
    curandStatus_t status = condition; \
    CHECK_EQ(status, CURAND_STATUS_SUCCESS) << " " \
      << caffe::curandGetErrorString(status); \
    } while (0)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())





namespace VasGpuBoost
{
	const char* cublasGetErrorString(cublasStatus_t error);
	const char* curandGetErrorString(curandStatus_t error);



	const int CUDA_NUM_THREADS = 512;
	// CUDA: number of blocks for threads.
	inline int GET_BLOCKS(const int N) 
	{
		return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
	}
	void QueryDevice();


	class ColorConvert
	{
		public:
			static ColorConvert* Get()
			{
				static ColorConvert instance_;
				return &instance_;
			}
		
			~ColorConvert();

			void ConvertYUV422pToRGB24(const uchar* yuv_data, uchar *rgb_data, const int width, const int height);

		private:
			ColorConvert();
			uchar* d_table_r_ = NULL;
			uchar* d_table_g_ = NULL;
			uchar* d_table_b_ = NULL;

			int height_ = 1920;
			int width_ = 1080;
			int size_ = 1920 * 1080;
			int yuv_mem_size_;
			int rgb_mem_size_;

			void UpdateMemSize(int width,int height);

			uchar* d_yuv_data_ = NULL;
			uchar* d_rgb_data_ = NULL; 
			boost::mutex convert_mutex_;



	};

	
}

