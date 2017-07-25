#include "BgVibeBoost.h"
#include "time.h"
__constant__ int d_off[9] = { -1, 0, 1, -1, 1, -1, 0, 1, 0 };

using namespace VasGpuBoost;

boost::mutex proces_mutex;

__global__ void ProcessFirstFrame_Kernel(const uchar *d_imageMat, const float *d_uniform_randMatList, uchar *d_samplistData, int width, int height, int numSamples)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int size = width*height;

	int x, y;
	x = i % width;
	y = i / width;
	int random;
	int row, col;
	CUDA_KERNEL_LOOP(i, size)
	{
		for (int k = 0; k < numSamples; k++)
		{
			random = 9 * d_uniform_randMatList[k*size + i];
			if (random == 9)
			{
				random = random - 1;
			}
			row = y + d_off[random];
			if (row < 0)
				row = 0;
			if (row >= height)
				row = height - 1;

			random = 9 * d_uniform_randMatList[(k + numSamples)*size + i];

			col = x + d_off[random];
			if (col < 0)
				col = 0;
			if (col >= width)
				col = width - 1;

			d_samplistData[k*size + i] = d_imageMat[col + row*width];
		}
	}

}

__global__ void TestAndUpadateSingle_kernel(const uchar* imageMat,
											const float* randMat,
											uchar* samplelist,
											uchar* foregroundMatchCount,
											uchar* mask,
											int width, int height,
											int numSamples,
											int minMatch,
											int radius,
											int subSampleFactor,
											int max_mismatch_count,
											float learn_rate)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int size = width*height;
	int x, y;
	x = i % width;
	y = i / width;
	float dist;
	int random;
	float f_random;

	int row, col;

	int matches(0), count(0);
	CUDA_KERNEL_LOOP(i, size)
	{
		while (count < numSamples)
		{
			dist = abs(samplelist[count*size + i] - imageMat[i]);
			if (dist < radius)
				matches++;

			if (matches >= minMatch)
			{
				break;
			}
			count++;
		}
		if (matches >= minMatch)
		{
			foregroundMatchCount[i] = 0;
			mask[i] = 0;
			// rand 0 
			f_random = (float)subSampleFactor*randMat[i];
			if (f_random < 1)
			{
				// rand 1
				random = numSamples*randMat[size + i];
				if (random == numSamples)
				{
					random = random - 1;
				}
				samplelist[random*size + i] = imageMat[i];
			}
			// rand 2
			f_random = (float)subSampleFactor*randMat[2*size + i];
			if (f_random < 1)
			{
				int row, col;
				// rand 3
				random = 9 * randMat[3 * size + i];
				if (random == 9)
				{
					random = random - 1;
				}
				row = y + d_off[random];
				if (row < 0)
					row = 0;
				if (row >= height)
					row = height - 1;
				// rand 4
				random = 9 * randMat[4 * size + i];
				if (random == 9)
				{
					random = random - 1;
				}
				col = x + d_off[random];
				if (col < 0)
					col = 0;
				if (col >= width)
					col = width - 1;

				// rand 5
				random = numSamples*randMat[5 * size + i];
				if (random == numSamples)
				{
					random = random - 1;
				}
				samplelist[random*size+ row*width + col] = imageMat[i];
			}
		}
		else
		{
			foregroundMatchCount[i]++;
			mask[i] = 255;
			if (foregroundMatchCount[i] > max_mismatch_count)
			{
				mask[i] = 0;
				foregroundMatchCount[i] = 0;
				for (size_t ii = 0; ii < minMatch; ii++)
				{
					f_random = subSampleFactor*randMat[ii * size + i];
					if (f_random < subSampleFactor*learn_rate)
					{
						random = numSamples*randMat[ii * size + i];
						if (random == numSamples)
						{
							random = random - 1;
						}
						samplelist[random*size + i] = imageMat[i];
					}
				}


			}

		}

	}

}
__global__ void UtifyMask_kernel(const uchar* mask_1,const uchar* mask_2,uchar* utified_mask,int width, int height)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int size = width*height;
	int x, y;
	x = i % width;
	y = i / width;
	CUDA_KERNEL_LOOP(i, size)
	{
		if (mask_1[i]>0&&mask_2[i]>0)
		{
			utified_mask[i] = 255;
		}
		else utified_mask[i] = 0;
	}
}

void BgVibeBoost::ProcessFirstFrameGPU(const Mat& gray_frame, uchar *d_sample_list)
{
	int num_sample = bg_parameter_.vibe_parameter().num_samples();
	float *d_fisrt_frame_rand= GpuRandGenerator::Get()->GetGpuUnifromRand(2 * num_sample * size_);
	proces_mutex.lock();
	int memSize = size_*sizeof(uchar);
	CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_frame_, gray_frame.data, memSize, cudaMemcpyHostToDevice));
	ProcessFirstFrame_Kernel << <GET_BLOCKS(size_), CUDA_NUM_THREADS >> >(d_frame_, d_fisrt_frame_rand, d_sample_list1_, gray_frame.cols, gray_frame.rows, num_sample);
	proces_mutex.unlock();
}

void BgVibeBoost::TestAndUpdateSingleGPU(const Mat& gray_frame)
{
	int min_match = bg_parameter_.vibe_parameter().min_match();
	int num_sample = bg_parameter_.vibe_parameter().num_samples();
	int radius = bg_parameter_.vibe_parameter().radius();
	int subsample_factor = bg_parameter_.vibe_parameter().subsample_factor();
	int max_mismatch_count = bg_parameter_.vibe_parameter().max_mismatch_count();
	float learn_rate = bg_parameter_.vibe_parameter().learn_rate();
	if (min_match>8)
	{
		cout << "Min matach is more than 8" << endl;
		return;
	}
	size_t memSize = size_*sizeof(uchar);
	float *d_rand = GpuRandGenerator::Get()->GetGpuUnifromRand(8 * size_);
	proces_mutex.lock();
	CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_frame_, gray_frame.data, memSize, cudaMemcpyHostToDevice));
	TestAndUpadateSingle_kernel << <GET_BLOCKS(size_), CUDA_NUM_THREADS >> >(d_frame_, d_rand, d_sample_list1_, foreground_match_count_mat1_, d_mask1_, gray_frame.cols, gray_frame.rows, num_sample, min_match, radius, subsample_factor, max_mismatch_count, learn_rate);
	CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(mask_.data, d_mask1_, memSize, cudaMemcpyDeviceToHost));
	proces_mutex.unlock();

}
void BgVibeBoost::TestAndUpdateDoubleGPU(const Mat& gray_frame)
{
	int min_match = bg_parameter_.vibe_parameter().min_match();
	int num_sample = bg_parameter_.vibe_parameter().num_samples();
	int radius = bg_parameter_.vibe_parameter().radius();
	int subsample_factor = bg_parameter_.vibe_parameter().subsample_factor();
	int max_mismatch_count = bg_parameter_.vibe_parameter().max_mismatch_count();
	float learn_rate = bg_parameter_.vibe_parameter().learn_rate();
	size_t memSize = size_*sizeof(uchar);
	float *d_rand = GpuRandGenerator::Get()->GetGpuUnifromRand(8 * size_);
	proces_mutex.lock();
	CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_frame_, gray_frame.data, memSize, cudaMemcpyHostToDevice));
	TestAndUpadateSingle_kernel << <GET_BLOCKS(size_), CUDA_NUM_THREADS >> >(d_frame_, d_rand, d_sample_list1_, foreground_match_count_mat1_, d_mask1_, gray_frame.cols, gray_frame.rows, num_sample, min_match, radius, subsample_factor, max_mismatch_count, learn_rate);
	d_rand = GpuRandGenerator::Get()->GetGpuUnifromRand(8 * size_);
	TestAndUpadateSingle_kernel << <GET_BLOCKS(size_), CUDA_NUM_THREADS >> >(d_frame_, d_rand, d_sample_list2_, foreground_match_count_mat2_, d_mask2_, gray_frame.cols, gray_frame.rows, num_sample, min_match, radius, subsample_factor, max_mismatch_count, learn_rate);
	UtifyMask_kernel << <GET_BLOCKS(size_), CUDA_NUM_THREADS >> >(d_mask1_, d_mask2_, d_mask_, gray_frame.cols, gray_frame.rows);
	CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(mask_.data, d_mask_, memSize, cudaMemcpyDeviceToHost));
	proces_mutex.unlock();
}