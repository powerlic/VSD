#include "cuda_runtime.h"
#include "curand.h"
#include "curand_kernel.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include "time.h"

#include"cvpreload.h"

using namespace cv;
using namespace std;

#  define CUDA_SAFE_CALL_NO_SYNC( call) {                                    \
		cudaError err = call;                                                    \
		if( cudaSuccess != err) {                                                \
		fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
			    }}


#  define CUDA_SAFE_CALL_RAND( call) {                                    \
		curandStatus_t err = call;                                                    \
		if( CURAND_STATUS_SUCCESS != err) {                                                \
		fprintf(stderr, "Curand error in file '%s' in line %i.\n",        \
                __FILE__, __LINE__);              \
        exit(EXIT_FAILURE);                                                  \
					    }}


__constant__ int c_yoff[9] = { -1, 0, 1, -1, 1, -1, 0, 1, 0 };
__constant__ int c_xoff[9] = { -1, 0, 1, -1, 1, -1, 0, 1, 0 };


 __device__ void UpdateSamplistDevice(uchar matches, uchar pixel_value, uchar min_match, uchar max_mismatch_count, uchar sub_sample_factor, int number_samplist, int x, int y, int width,int height, const float *rand_mat_list, bool update_fore, uchar *forground_match_count_mat, uchar *mask, uchar *samplist)
{
	int offset = x + y*width;
	if (matches >= min_match)
	{
		
		*(mask + offset) = 0;
		*(forground_match_count_mat + offset) = 0;

		//update sample_list
		if (*(rand_mat_list + offset)*(float)sub_sample_factor<2)
		{
			uchar NO = uchar(*(rand_mat_list + 1*width*height +offset)*(float)number_samplist);

			*(samplist + NO*width*height + offset) = pixel_value;

			int xc, yc;
			if (x == (width - 1))
			{
				xc = 0;
			}
			else xc = x + 1;
			if (y == (height - 1))
			{
				yc = 0;
			}
			else yc = y + 1;

			int xr, yr;
			if (x == 0)
			{
				xr = 1;
			}
			else xr = x - 1;
			if (y == 0)
			{
				yr = 1;
			}
			else yr = y - 1;

			
			int xx = x + c_xoff[uchar(*(rand_mat_list + xc + width*yc)*9.0)];
			int yy = x + c_xoff[uchar(*(rand_mat_list + xr + width*yr)*9.0)];

			int neigbor = xx + yy*width;

			*(samplist + NO*width*height + neigbor) = pixel_value;
		}
	}
	else
	{
		*(mask + offset) = 255;
		*(forground_match_count_mat + offset) = *(forground_match_count_mat + offset)+1;
		if (update_fore)
		{
			if (*(forground_match_count_mat + offset)>max_mismatch_count)
			{
				*(mask + offset) = 255;
				*(forground_match_count_mat + offset) = 0;

		
				for (size_t i = 0; i < min_match; i++)
				{
					int x_ = i*(width - 1) / min_match;
					uchar NO = uchar(*(rand_mat_list + x_+y*width)*(float)number_samplist);

					*(samplist + NO*width*height + offset) = pixel_value;	
				}

			}
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
											int max_mismatch_count
											)
{

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int matches(0), count(0);
	float dist;
	int random;
	float f_random;

	if (x < width && y < height)
	{
		int offset = x + y * width;
		while (count < numSamples)
		{
			dist = abs(samplelist[count*width*height + offset] - imageMat[offset]);
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
			foregroundMatchCount[offset] = 0;
			mask[offset] = 0;

			f_random = subSampleFactor*randMat[offset]; 
			random = subSampleFactor*randMat[offset];
			if (random == subSampleFactor)
			{
				random = random - 1;
			}

			if (f_random < 1)
			{
				random = numSamples*randMat[3 * width*height + offset];
				if (random == numSamples)
				{
					random = random - 1;
				}

				samplelist[random*height*width + offset] = imageMat[offset];
			}

			f_random = subSampleFactor*randMat[width*height + offset];
			random = subSampleFactor*randMat[width*height + offset];
			if (random == subSampleFactor)
			{
				random = random - 1;
			}

			if (f_random < 1)
			{
				int row, col;
				
				random = 9 * randMat[4 * width*height + offset];
				if (random == 9)
				{
					random = random - 1;
				}

				row = y + c_yoff[random];

				if (row < 0)
					row = 0;
				if (row >= height)
					row = height - 1;

				random = 9 * randMat[5 * width*height + offset];
				if (random == 9)
				{
					random = random - 1;
				}
				col = x + c_xoff[random];
				if (col < 0)
					col = 0;
				if (col >= width)
					col = width - 1;

				random = numSamples*randMat[6 * width*height + offset];
				if (random == numSamples)
				{
					random = random - 1;
				}
				samplelist[random*width*height + row*width + col] = imageMat[offset];
			}
		}
		else
		{
			foregroundMatchCount[offset]++;
			mask[offset] = 255;
			if (foregroundMatchCount[offset] > max_mismatch_count)
			{
				mask[offset] = 0;
				foregroundMatchCount[offset] = 0;
				

				for (size_t i = 0; i < minMatch; i++)
				{
					f_random = subSampleFactor*randMat[i * width*height + offset];
					random = subSampleFactor*randMat[i * width*height + offset];
					if (random == subSampleFactor)
					{
						random = random - 1;
					}
					if (f_random < subSampleFactor/2)
					{
						if (random == numSamples)
						{
							random = random - 1;
						}
						samplelist[random*width*height + offset] = imageMat[offset];
					}
				}

				
			}
		}
	}

}


__global__ void TestAndUpadateDouble_kernel(const uchar* imageMat,
	const float* randMat,
	uchar* samplelist,
	uchar* samplelist_2,
	uchar* foregroundMatchCount,
	uchar* foregroundMatchCount_2,
	uchar* mask,
	uchar* mask_2,//两个背景模型之和
	uchar* unified_mask,
	int width, int height,
	int numSamples,
	int minMatch,
	int radius,
	int subSampleFactor,
	int max_mismatch_count
	)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int matches(0),matches_2(0), count(0);
	float dist,dist_2;
	int random;
	float f_random;

	if (x < width && y < height)
	{
		int offset = x + y * width;

		while (count < numSamples)
		{
			dist = abs(samplelist[count*width*height + offset] - imageMat[offset]);
			dist_2 = abs(samplelist_2[count*width*height + offset] - imageMat[offset]);
			if (dist < radius)
				matches++;
			if (dist_2 < radius)
				matches_2++;

			if (matches>=minMatch&&matches_2>=minMatch)
			{
				break;
			}

			count++;
		}

		if (matches >= minMatch)
		{
			foregroundMatchCount[offset] = 0;
			mask[offset] = 0;


			f_random = subSampleFactor*randMat[offset]; //(1)
				random = subSampleFactor*randMat[offset]; //(2)
			if (random == subSampleFactor)
			{
				random = random - 1;
			}

			if (f_random < 1)
			{
				random = numSamples*randMat[3 * width*height + offset];
				if (random == numSamples)
				{
					random = random - 1;
				}

				samplelist[random*height*width + offset] = imageMat[offset];
			}

			// 同时也有 1 / defaultSubsamplingFactor 的概率去更新它的邻居点的模型样本值  
			//random = subSample_randMat2[offset];

			f_random = subSampleFactor*randMat[width*height + offset];
			random = subSampleFactor*randMat[width*height + offset];
			if (random == subSampleFactor)
			{
				random = random - 1;
			}

			if (f_random < 1)
			{
				int row, col;
				//random = rng.uniform(0, 9);
				//random = uniform_randMat2[offset];

				random = 9 * randMat[4 * width*height + offset];
				if (random == 9)
				{
					random = random - 1;
				}

				row = y + c_yoff[random];

				if (row < 0)
					row = 0;
				if (row >= height)
					row = height - 1;

				//random = uniform_randMat3[offset];
				random = 9 * randMat[5 * width*height + offset];
				if (random == 9)
				{
					random = random - 1;
				}
				//random = rng.uniform(0, 9);
				col = x + c_xoff[random];
				if (col < 0)
					col = 0;
				if (col >= width)
					col = width - 1;

				//random = rng.uniform(0, NUM_SAMPLES);
				//random = uniform_randMat4[offset];
				random = numSamples*randMat[6 * width*height + offset];
				if (random == numSamples)
				{
					random = random - 1;
				}
				samplelist[random*width*height + row*width + col] = imageMat[offset];
				//m_samples[random].at<uchar>(row, col) = _image.at<uchar>(i, j);
			}
		}
		else
		{
			// It is a foreground pixel  
			foregroundMatchCount[offset]++;
			// Set background pixel to 255  
			//m_mask.at<uchar>(i, j) = 255;
			mask[offset] = 255;
			//如果某个像素点连续N次被检测为前景，则认为一块静止区域被误判为运动，将其更新为背景点  
			if (foregroundMatchCount[offset] > max_mismatch_count)
			{
				mask[offset] = 0;
				foregroundMatchCount[offset] = 0;

				for (size_t i = 0; i < minMatch; i++)
				{
					f_random = subSampleFactor*randMat[i * width*height + offset];
					random = subSampleFactor*randMat[i * width*height + offset];
					if (random == subSampleFactor)
					{
						random = random - 1;
					}
					if (f_random < subSampleFactor / 2)
					{
						if (random == numSamples)
						{
							random = random - 1;
						}
						samplelist[random*width*height + offset] = imageMat[offset];
					}
				}
			}
		}
		//mask_2
		if (matches_2 >= minMatch)
		{
			foregroundMatchCount_2[offset] = 0;

			mask_2[offset] = 0;

			f_random = subSampleFactor*randMat[offset];
			random = subSampleFactor*randMat[offset];
			if (random == subSampleFactor)
			{
				random = random - 1;
			}

			if (f_random < 1)
			{
				random = numSamples*randMat[6 * width*height + offset];
				if (random == numSamples)
				{
					random = random - 1;
				}

				samplelist_2[random*height*width + offset] = imageMat[offset];
			}

			f_random = subSampleFactor*randMat[width*height + offset];
			random = subSampleFactor*randMat[width*height + offset];
			if (random == subSampleFactor)
			{
				random = random - 1;
			}

			if (f_random < 1)
			{
				int row, col;
				//random = rng.uniform(0, 9);
				//random = uniform_randMat2[offset];

				random = 9 * randMat[5 * width*height + offset];
				if (random == 9)
				{
					random = random - 1;
				}

				row = y + c_yoff[random];

				if (row < 0)
					row = 0;
				if (row >= height)
					row = height - 1;

				//random = uniform_randMat3[offset];
				random = 9 * randMat[4 * width*height + offset];
				if (random == 9)
				{
					random = random - 1;
				}
				//random = rng.uniform(0, 9);
				col = x + c_xoff[random];
				if (col < 0)
					col = 0;
				if (col >= width)
					col = width - 1;


				random = numSamples*randMat[3 * width*height + offset];
				if (random == numSamples)
				{
					random = random - 1;
				}
				samplelist_2[random*width*height + row*width + col] = imageMat[offset];
			}
		}
		else
		{
			// It is a foreground pixel  
			foregroundMatchCount_2[offset]++;
			// Set background pixel to 255  
			//m_mask.at<uchar>(i, j) = 255;
			mask_2[offset] = 255;
			//如果某个像素点连续N次被检测为前景，则认为一块静止区域被误判为运动，将其更新为背景点  
			if (foregroundMatchCount_2[offset] > max_mismatch_count)
			{
				mask_2[offset] = 0;
				foregroundMatchCount_2[offset] = 0;

				for (size_t i = 0; i < minMatch; i++)
				{
					f_random = subSampleFactor*randMat[i * width*height + offset];
					random = subSampleFactor*randMat[i * width*height + offset];
					if (random == subSampleFactor)
					{
						random = random - 1;
					}
					if (f_random < subSampleFactor / 2)
					{
						if (random == numSamples)
						{
							random = random - 1;
						}
						samplelist[random*width*height + offset] = imageMat[offset];
					}
				}

			}
		}


		if (mask[offset] && mask_2[offset])
		{
			unified_mask[offset] = 255;
		}
		else
		{
			unified_mask[offset] = 0;
		}
	}
}


extern "C" void TestAndUpadateDouble_Caller(Mat imageMat, 
									  Mat &maskMat, Mat &maskMat2, 
									  Mat &utifiedMat, Mat *samplelist, 
									  Mat *samplelist_2, int numSamples, int minMatch, int radius, int subSampleFactor, int max_mismatch_count,
									  Mat &foregroundMatchCount, Mat &foregroundMatchCount_2)
{
	
	if (minMatch>8)
	{
		cerr<< __FUNCTION__ << "Min Match is more than 8" << endl;
		return;
	}
	
	int width = imageMat.cols;
	int height = imageMat.rows;
	size_t memSize = width*height*sizeof(uchar);
	size_t memSizefloat = 8*width*height*sizeof(float);
	uchar *d_imageMat = NULL;
	uchar *d_mask = NULL;
	uchar *d_mask_2 = NULL;
	uchar *d_utified_mask = NULL;
	uchar *d_samplelist = NULL;
	uchar *d_samplelist_2 = NULL;
	uchar *d_foregroundMatchCount = NULL;
	uchar *d_foregroundMatchCount_2 = NULL;
	float *d_randMat = NULL;


	

	//memory
	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_imageMat, memSize));
	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_mask, memSize));
	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_mask_2, memSize));
	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_utified_mask, memSize));


	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_foregroundMatchCount, memSize));
	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_foregroundMatchCount_2, memSize));
	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_samplelist, numSamples*memSize));
	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_samplelist_2, numSamples*memSize));
	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_randMat, memSizefloat));



	time_t t_1, t_2,t_3;
	t_1 = clock();
	curandGenerator_t gen;

	time_t t_seed;
	t_seed = clock();
	curandStatus_t t = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	t = curandSetPseudoRandomGeneratorSeed(gen, (unsigned long long)t_seed);
	curandGenerateUniform(gen, d_randMat, width*height*8);
	curandDestroyGenerator(gen);
	t_2 = clock();
	
	uchar *samplistData = (uchar*)malloc(memSize*numSamples);
	uchar *samplistData_2 = (uchar*)malloc(memSize*numSamples);

	for (int i = 0; i < numSamples; i++)
	{
		memcpy(samplistData + i*width*height, samplelist[i].data, memSize);
		memcpy(samplistData_2 + i*width*height, samplelist_2[i].data, memSize);
	}
	//cout << " Data Preprared need " << t_2 - t_1 << endl;

	//复制内存
	CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_imageMat, imageMat.data, memSize, cudaMemcpyHostToDevice));
	//cudaMemcpy(d_mask, maskMat.data, memSize, cudaMemcpyHostToDevice);

	CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_samplelist, samplistData, numSamples*memSize, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_samplelist_2, samplistData_2, numSamples*memSize, cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_foregroundMatchCount, foregroundMatchCount.data, memSize, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_foregroundMatchCount_2, foregroundMatchCount_2.data, memSize, cudaMemcpyHostToDevice));


	dim3 threads(32, 32);
	dim3 grids((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
	
	
	TestAndUpadateDouble_kernel << <grids, threads >> >(d_imageMat,
												  d_randMat, d_samplelist, d_samplelist_2, 
												  d_foregroundMatchCount, d_foregroundMatchCount_2, 
												  d_mask, d_mask_2, d_utified_mask,
												  width, height, numSamples, minMatch, radius, subSampleFactor, max_mismatch_count
												  );

	CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(maskMat.data, d_mask, memSize, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(maskMat2.data, d_mask_2, memSize, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(utifiedMat.data, d_utified_mask, memSize, cudaMemcpyDeviceToHost));
	
	CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(samplistData, d_samplelist, numSamples*memSize, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(samplistData_2, d_samplelist_2, numSamples*memSize, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(foregroundMatchCount.data, d_foregroundMatchCount, memSize, cudaMemcpyDeviceToHost));
	cudaMemcpy(foregroundMatchCount_2.data, d_foregroundMatchCount_2, memSize, cudaMemcpyDeviceToHost);

	for (int i = 0; i < numSamples; i++)
	{
		memcpy(samplelist[i].data, samplistData + i*width*height, memSize);
		memcpy(samplelist_2[i].data, samplistData_2 + i*width*height, memSize);
	}
	t_3 = clock();
	//cout << " Bg update need    " << t_3 - t_1 << endl;
	free(samplistData);
	free(samplistData_2);

	cudaFree(d_imageMat);

	cudaFree(d_mask);
	cudaFree(d_mask_2);
	cudaFree(d_utified_mask);

	cudaFree(d_samplelist);
	cudaFree(d_samplelist_2);

	cudaFree(d_foregroundMatchCount);
	cudaFree(d_foregroundMatchCount_2);

	cudaFree(d_randMat);

}
extern "C" void TestAndUpadateSingle_Caller(Mat imageMat,
									        Mat &maskMat, Mat *samplelist,
									        int numSamples, int minMatch, int radius, int subSampleFactor, int max_mismatch_count,
									        Mat &foregroundMatchCount)
{

	if (minMatch>8)
	{
		cerr <<__FUNCTION__<<" Min Match is more than 8" << endl;
		return;
	}

	int width = imageMat.cols;
	int height = imageMat.rows;
	size_t memSize = width*height*sizeof(uchar);
	size_t memSizefloat = 8 * width*height*sizeof(float);
	uchar *d_imageMat = NULL;
	uchar *d_mask = NULL;
	uchar *d_samplelist = NULL;
	uchar *d_foregroundMatchCount = NULL;
	float *d_randMat = NULL;

	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_imageMat, memSize));
	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_mask, memSize));


	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_foregroundMatchCount, memSize));
	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_samplelist, numSamples*memSize));
	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_randMat, memSizefloat));

	curandGenerator_t gen;

	time_t t_seed;
	t_seed = clock();
	CUDA_SAFE_CALL_RAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
	CUDA_SAFE_CALL_RAND(curandSetPseudoRandomGeneratorSeed(gen, (unsigned long long)t_seed));
	CUDA_SAFE_CALL_RAND(curandGenerateUniform(gen, d_randMat, width*height * 8));
	CUDA_SAFE_CALL_RAND(curandDestroyGenerator(gen));

	uchar *samplistData = (uchar*)malloc(memSize*numSamples);

	for (int i = 0; i < numSamples; i++)
	{
		memcpy(samplistData + i*width*height, samplelist[i].data, memSize);
	}

	CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_imageMat, imageMat.data, memSize, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_samplelist, samplistData, numSamples*memSize, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_foregroundMatchCount, foregroundMatchCount.data, memSize, cudaMemcpyHostToDevice));


	dim3 threads(32, 32);
	dim3 grids((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

	TestAndUpadateSingle_kernel << <grids, threads >> >(d_imageMat, d_randMat, d_samplelist, d_foregroundMatchCount,
		                                                d_mask, width, height, numSamples, minMatch, radius, subSampleFactor, max_mismatch_count);


	CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(maskMat.data, d_mask, memSize, cudaMemcpyDeviceToHost));

	CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(samplistData, d_samplelist, numSamples*memSize, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(foregroundMatchCount.data, d_foregroundMatchCount, memSize, cudaMemcpyDeviceToHost));

	for (int i = 0; i < numSamples; i++)
	{
		memcpy(samplelist[i].data, samplistData + i*width*height, memSize);
	}
	free(samplistData);

	cudaFree(d_imageMat);
	cudaFree(d_mask);
	cudaFree(d_samplelist);
	cudaFree(d_foregroundMatchCount);

	cudaFree(d_randMat);


}

__global__ void ProcessingFirstFrame_Kernel(const uchar *d_imageMat, const float *d_uniform_randMatList, uchar *d_samplistData, int width, int height,int numSamples)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int random;
	int row, col;
	if (x < width && y < height)
	{
		int offset = x + y * width;
		for (int k = 0; k < numSamples; k++)
		{
			
			random = 9 * d_uniform_randMatList[k*width*height + offset];
			if (random == 9)
			{
				random = random - 1;
			}

			row = y + c_yoff[random];
			if (row < 0)
				row = 0;
			if (row >= height)
				row = height - 1;

			random = 9 * d_uniform_randMatList[(k + numSamples)*width*height + offset];

			col = x + c_xoff[random];
			if (col < 0)
				col = 0;
			if (col >= width)
				col = width - 1;

			d_samplistData[k*width*height + offset] = d_imageMat[col + row*width];
		}

	}

}


extern "C" void ProcessingFirstFrame_Caller(Mat imageMat, Mat *samplelist,int numSamples)
{
	int width, int height;
	width = imageMat.cols;
	height = imageMat.rows;
	size_t memSize = width*height*sizeof(uchar);
	size_t memSizefloat = width*height*sizeof(float);
	float *d_uniform_randMatList = NULL;

	curandGenerator_t gen;
	time_t t_seed;
	t_seed = clock();
	CUDA_SAFE_CALL_RAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
	CUDA_SAFE_CALL_RAND(curandSetPseudoRandomGeneratorSeed(gen, (unsigned long long)t_seed));

	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_uniform_randMatList, 2 * memSizefloat*numSamples));
	
	CUDA_SAFE_CALL_RAND(curandGenerateUniform(gen, d_uniform_randMatList, 2 * width*height*numSamples));
	CUDA_SAFE_CALL_RAND(curandDestroyGenerator(gen));

	uchar *d_imageMat = NULL;
	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_imageMat, memSize));
	CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_imageMat, imageMat.data, memSize, cudaMemcpyHostToDevice));

	uchar *samplistData = (uchar*)malloc(memSize*numSamples);

	uchar *d_samplistData = NULL;
	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_samplistData, memSize*numSamples));

	
	dim3 threads(32, 32);
	dim3 grids((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
	
	ProcessingFirstFrame_Kernel << <grids, threads >> >(d_imageMat, d_uniform_randMatList, d_samplistData, width, height, numSamples);

	CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(samplistData, d_samplistData, numSamples*memSize, cudaMemcpyDeviceToHost));
	for (int i = 0; i < numSamples; i++)
	{
		memcpy(samplelist[i].data, samplistData + i*width*height, memSize);
	}
	

	cudaFree(d_imageMat);
	cudaFree(d_uniform_randMatList);
	cudaFree(d_samplistData);
	free(samplistData); 
}


