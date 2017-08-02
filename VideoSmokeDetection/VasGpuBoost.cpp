#include "stdafx.h"
#include "VasGpuBoost.h"
#include <math.h>

#define max(a,b) (((a) > (b)) ? (a) : (b))  
#define min(a,b) (((a) < (b)) ? (a) : (b)) 

namespace VasGpuBoost
{
	const char* curandGetErrorString(curandStatus_t error)
	{
		switch (error) {
		case CURAND_STATUS_SUCCESS:
			return "CURAND_STATUS_SUCCESS";
		case CURAND_STATUS_VERSION_MISMATCH:
			return "CURAND_STATUS_VERSION_MISMATCH";
		case CURAND_STATUS_NOT_INITIALIZED:
			return "CURAND_STATUS_NOT_INITIALIZED";
		case CURAND_STATUS_ALLOCATION_FAILED:
			return "CURAND_STATUS_ALLOCATION_FAILED";
		case CURAND_STATUS_TYPE_ERROR:
			return "CURAND_STATUS_TYPE_ERROR";
		case CURAND_STATUS_OUT_OF_RANGE:
			return "CURAND_STATUS_OUT_OF_RANGE";
		case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
			return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
		case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
			return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
		case CURAND_STATUS_LAUNCH_FAILURE:
			return "CURAND_STATUS_LAUNCH_FAILURE";
		case CURAND_STATUS_PREEXISTING_FAILURE:
			return "CURAND_STATUS_PREEXISTING_FAILURE";
		case CURAND_STATUS_INITIALIZATION_FAILED:
			return "CURAND_STATUS_INITIALIZATION_FAILED";
		case CURAND_STATUS_ARCH_MISMATCH:
			return "CURAND_STATUS_ARCH_MISMATCH";
		case CURAND_STATUS_INTERNAL_ERROR:
			return "CURAND_STATUS_INTERNAL_ERROR";
		}
		return "Unknown curand status";
	}
	const char* cublasGetErrorString(cublasStatus_t error)
	{
		switch (error) {
		case CUBLAS_STATUS_SUCCESS:
			return "CUBLAS_STATUS_SUCCESS";
		case CUBLAS_STATUS_NOT_INITIALIZED:
			return "CUBLAS_STATUS_NOT_INITIALIZED";
		case CUBLAS_STATUS_ALLOC_FAILED:
			return "CUBLAS_STATUS_ALLOC_FAILED";
		case CUBLAS_STATUS_INVALID_VALUE:
			return "CUBLAS_STATUS_INVALID_VALUE";
		case CUBLAS_STATUS_ARCH_MISMATCH:
			return "CUBLAS_STATUS_ARCH_MISMATCH";
		case CUBLAS_STATUS_MAPPING_ERROR:
			return "CUBLAS_STATUS_MAPPING_ERROR";
		case CUBLAS_STATUS_EXECUTION_FAILED:
			return "CUBLAS_STATUS_EXECUTION_FAILED";
		case CUBLAS_STATUS_INTERNAL_ERROR:
			return "CUBLAS_STATUS_INTERNAL_ERROR";
		#if CUDA_VERSION >= 6000
				case CUBLAS_STATUS_NOT_SUPPORTED:
					return "CUBLAS_STATUS_NOT_SUPPORTED";
		#endif
		#if CUDA_VERSION >= 6050
				case CUBLAS_STATUS_LICENSE_ERROR:
					return "CUBLAS_STATUS_LICENSE_ERROR";
		#endif
		}
		return "Unknown cublas status";
	}
	void QueryDevice()
	{
		cudaDeviceProp prop;
		int device;
		if (cudaSuccess != cudaGetDevice(&device))
		{
			printf("No cuda device present.\n");
			return;
		}
		CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
		LOG(INFO) << "Device id:                     " << device;
		LOG(INFO) << "Major revision number:         " << prop.major;
		LOG(INFO) << "Minor revision number:         " << prop.minor;
		LOG(INFO) << "Name:                          " << prop.name;
		LOG(INFO) << "Total global memory:           " << prop.totalGlobalMem;
		LOG(INFO) << "Total shared memory per block: " << prop.sharedMemPerBlock;
		LOG(INFO) << "Total registers per block:     " << prop.regsPerBlock;
		LOG(INFO) << "Warp size:                     " << prop.warpSize;
		LOG(INFO) << "Maximum memory pitch:          " << prop.memPitch;
		LOG(INFO) << "Maximum threads per block:     " << prop.maxThreadsPerBlock;
		LOG(INFO) << "Maximum dimension of block:    "
			<< prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", "
			<< prop.maxThreadsDim[2];
		LOG(INFO) << "Maximum dimension of grid:     "
			<< prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", "
			<< prop.maxGridSize[2];
		LOG(INFO) << "Clock rate:                    " << prop.clockRate;
		LOG(INFO) << "Total constant memory:         " << prop.totalConstMem;
		LOG(INFO) << "Texture alignment:             " << prop.textureAlignment;
		LOG(INFO) << "Concurrent copy and execution: "
			<< (prop.deviceOverlap ? "Yes" : "No");
		LOG(INFO) << "Number of multiprocessors:     " << prop.multiProcessorCount;
		LOG(INFO) << "Kernel execution timeout:      "
			<< (prop.kernelExecTimeoutEnabled ? "Yes" : "No");
		return;
	}
	
	ColorConvert::ColorConvert()
	{
		uchar *table_r = new uchar[256 * 256];
		uchar *table_g = new uchar[256 * 256 * 256];
		uchar *table_b = new uchar[256 * 256];

		for (int i = 0; i < 256; ++i)
		{
			for (int j = 0; j < 256; ++j)
			{
				int cc = i + 1.13983 * (j - 128);
				table_r[(i << 8) + j] = min(max(cc, 0), 255);
			}
		}

		for (int i = 0; i < 256; ++i)
		{
			for (int j = 0; j < 256; ++j)
			{
				int cc = i + 2.03211 * (j - 128);
				table_b[(i << 8) + j] = min(max(cc, 0), 255);
			}
		}

		for (int i = 0; i < 256; ++i)
		{
			for (int j = 0; j < 256; ++j)
				for (int k = 0; k < 256; ++k)
				{
					int cc = i - 0.39465 * (j - 128) - 0.58060 * (k - 128);
					table_g[(i << 16) + (j << 8) + k] = min(max(cc, 0), 255);
				}
		}
		CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_table_r_, 256 * 256 * sizeof(uchar)));
		CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_table_g_, 256 * 256 * 256 * sizeof(uchar)));
		CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_table_b_, 256 * 256 * sizeof(uchar)));

		CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_table_r_, table_r, 256 * 256 * sizeof(uchar), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_table_g_, table_g, 256* 256 * 256 * sizeof(uchar), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_table_b_, table_b, 256 * 256 * sizeof(uchar), cudaMemcpyHostToDevice));

		yuv_mem_size_ = size_ + 2 * ((width_ / 2 + width_ % 2) * (height_ / 2 + height_ % 2));
		rgb_mem_size_ = size_ * 3;

		CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_yuv_data_, yuv_mem_size_ * sizeof(uchar)));
		CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_rgb_data_, rgb_mem_size_ * sizeof(uchar)));
		delete table_r;
		delete table_g;
		delete table_b;

	}

	ColorConvert::~ColorConvert()
	{
		CUDA_SAFE_CALL_NO_SYNC(cudaFree(d_table_r_));
		CUDA_SAFE_CALL_NO_SYNC(cudaFree(d_table_g_));
		CUDA_SAFE_CALL_NO_SYNC(cudaFree(d_table_b_));
		CUDA_SAFE_CALL_NO_SYNC(cudaFree(d_yuv_data_));
		CUDA_SAFE_CALL_NO_SYNC(cudaFree(d_rgb_data_));	
	}

	void ColorConvert::UpdateMemSize(int width, int height)
	{
		if (width*height>size_)
		{
			CUDA_SAFE_CALL_NO_SYNC(cudaFree(d_yuv_data_));
			CUDA_SAFE_CALL_NO_SYNC(cudaFree(d_rgb_data_));

			width_ = width; 
			height_ = height;
			size_ = width*height;

			yuv_mem_size_ = size_ + 2 * ((width_ / 2 + width_ % 2) * (height_ / 2 + height_ % 2));
			rgb_mem_size_ = size_ * 3;

			CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_yuv_data_, yuv_mem_size_ * sizeof(uchar)));
			CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_rgb_data_, rgb_mem_size_ * sizeof(uchar)));
		}


	}


}