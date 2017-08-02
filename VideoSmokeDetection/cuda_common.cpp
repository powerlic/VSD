#include"stdafx.h"
#include"cuda_common.h"

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


namespace VasGpuBoost
{

	GpuRandGenerator::GpuRandGenerator()
	{
		CURAND_CHECK(curandCreateGenerator(&gen_, CURAND_RNG_PSEUDO_DEFAULT));
	}

	GpuRandGenerator::~GpuRandGenerator()
	{
		CURAND_CHECK(curandDestroyGenerator(gen_));
		if (d_rand_)
		{
			CUDA_SAFE_CALL_NO_SYNC(cudaFree(d_rand_));
		}
	}
	void GpuRandGenerator::GenUnifrom(int count, float *d_dst)
	{
		gen_mutex_.lock();
		time_t t_seed = clock();
		CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen_, (unsigned long long)t_seed));
		CURAND_CHECK(curandGenerateUniform(gen_, d_dst, count));
		gen_mutex_.unlock();
	}
	float* GpuRandGenerator::GetGpuUnifromRand(int count)
	{
		gen_mutex_.lock();
		time_t t_seed = clock();
		if (count>d_size_)
		{
			d_size_ = count;
			if (d_rand_)
			{
				CUDA_SAFE_CALL_NO_SYNC(cudaFree(d_rand_));
			}
			CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_rand_, d_size_ * sizeof(float)));
		}
		else
		{
			if (!d_rand_)
			{
				CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_rand_, d_size_ * sizeof(float)));
			}
		}
		CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen_, (unsigned long long)t_seed));
		CURAND_CHECK(curandGenerateUniform(gen_, d_rand_, count));
		gen_mutex_.unlock();
		return d_rand_;
	}
}