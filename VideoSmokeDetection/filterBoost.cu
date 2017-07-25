#include "cuda_runtime.h"
#include "curand.h"
#include "curand_kernel.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "time.h"

using namespace cv;
using namespace std;


#  define CUDA_SAFE_CALL_NO_SYNC( call) {                                    \
		cudaError err = call;                                                    \
		if( cudaSuccess != err) {                                                \
		fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
	    }}



__device__ uchar ColorFilterPixel(const uchar r, const uchar g, const uchar b,
								  const uint32_t rgb_th1_upper, 
								  const uint32_t rgb_th2_lower,
								  const uint32_t rgb_th2_upper, 
								  const uint32_t ycbcr_th1_upper,
								  const uint32_t ycbcr_th2_lower,
								  const uint32_t ycbcr_th2_upper
								  )
{
	int max_rgb, min_rgb;
	max_rgb = r>g ? r : g;
	max_rgb = max_rgb>b ? max_rgb : b;

	min_rgb = r<g ? r : g;
	min_rgb = min_rgb<b ? min_rgb : b;
	int mean = (r + g + b) / 3;

	uchar y = 0.257*r + 0.504*g + 0.098*b + 16;
	uchar cb = -0.148*r - 0.291*g + 0.439*b + 128;
	uchar cr = 0.439*r - 0.368*g - 0.071*b + 128;

	int sum = (int)((cb - 128)*(cb - 128) + (cr - 128)*(cr - 128));

	if (abs(max_rgb - min_rgb)<rgb_th1_upper
		&&mean <= rgb_th2_upper
		&&mean >= rgb_th2_lower
		&&sum  <  ycbcr_th1_upper*ycbcr_th1_upper
		&& y >= ycbcr_th2_lower
		&& y <= ycbcr_th2_upper)
	{
		return 255;
	}
	else return 0;

}


//__device__ uchar ColorFilterPixel(const uchar r, const uchar g, const uchar b,const Threshold *ycbcr_th, const Threshold *rgb_th)
//{
//
//	int max_rgb, min_rgb;
//	max_rgb = r>g ? r : g;
//	max_rgb = max_rgb>b ? max_rgb : b;
//
//	min_rgb = r<g ? r : g;
//	min_rgb = min_rgb<b ? min_rgb : b;
//	int mean = (r + g + b) / 3;
//
//	uchar y = 0.257*r + 0.504*g + 0.098*b + 16;
//	uchar cb = -0.148*r - 0.291*g + 0.439*b + 128;
//	uchar cr = 0.439*r - 0.368*g - 0.071*b + 128;
//
//	int sum = (int)((cb - 128)*(cb - 128) + (cr - 128)*(cr - 128));
//
//	if (abs(max_rgb - min_rgb)<rgb_th->th1_upper&&mean <= rgb_th->th2_upper&&mean >= rgb_th->th2_lower&&sum<ycbcr_th->th1_upper*ycbcr_th->th1_upper&& y >= ycbcr_th->th2_lower&&y <= ycbcr_th->th2_upper)
//	{
//		return 255;
//	}
//	else return 0;
//
//}



__global__ void ColorFilter_Kenrel( uchar *d_R, uchar *d_G, uchar *d_B,
									uchar *inMask, 
									const uint32_t rgb_th1_upper,
									const uint32_t rgb_th2_lower,
									const uint32_t rgb_th2_upper,
									const uint32_t ycbcr_th1_upper,
									const uint32_t ycbcr_th2_lower,
									const uint32_t ycbcr_th2_upper,
									int width, int height,
									uchar *outMask, 
									uchar *colorMask)
{

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < width && y < height)
	{


		int offset = x + y * width;

		uchar r, g, b;
		r = d_R[offset];
		g = d_G[offset];
		b = d_B[offset];

		int max_rgb, min_rgb;
		max_rgb = r>g ? r : g;
		max_rgb = max_rgb>b ? max_rgb : b;

		min_rgb = r<g ? r : g;
		min_rgb = min_rgb<b ? min_rgb : b;
		int mean = (r + g + b) / 3;

		uchar y = 0.257*(float)r + 0.504*(float)g + 0.098*(float)b + 16;
		uchar cb = -0.148*(float)r - 0.291*(float)g + 0.439*(float)b + 128;
		uchar cr = 0.439*(float)r - 0.368*(float)g - 0.071*(float)b + 128;

		int sum = (int)((cb - 128)*(cb - 128) + (cr - 128)*(cr - 128));

		if (abs(max_rgb - min_rgb)<rgb_th1_upper
			&&mean <= rgb_th2_upper
			&&mean >= rgb_th2_lower
			&&sum<ycbcr_th1_upper*ycbcr_th1_upper
			&& y >= ycbcr_th2_lower
			&& y <= ycbcr_th2_upper)
		{
			colorMask[offset] = 255;
			if (inMask[offset] > 0)
			{
				outMask[offset] = 255;
			}
			else outMask[offset] = 0;

		}
		else
		{
			colorMask[offset] = 0;
			outMask[offset] = 0;
		}
	}

}

//__global__ void ColorFilter_Kenrel(uchar *d_R,  uchar *d_G,  uchar *d_B,
//								   uchar *inMask, Threshold *ycbcr_th, Threshold *rgb_th,
//								   int width,int height,
//								   uchar *outMask, uchar *colorMask)
//{
//	int x = threadIdx.x + blockIdx.x * blockDim.x;
//	int y = threadIdx.y + blockIdx.y * blockDim.y;
//
//	if (x < width && y < height)
//	{
//		
//
//		int offset = x + y * width;
//		
//		uchar r, g, b;
//		r = d_R[offset];
//		g = d_G[offset];
//		b = d_B[offset];
//
//		int max_rgb, min_rgb;
//		max_rgb = r>g ? r : g;
//		max_rgb = max_rgb>b ? max_rgb : b;
//
//		min_rgb = r<g ? r : g;
//		min_rgb = min_rgb<b ? min_rgb : b;
//		int mean = (r + g + b) / 3;
//
//		uchar y = 0.257*(float)r + 0.504*(float)g + 0.098*(float)b + 16;
//		uchar cb = -0.148*(float)r - 0.291*(float)g + 0.439*(float)b + 128;
//		uchar cr = 0.439*(float)r - 0.368*(float)g - 0.071*(float)b + 128;
//
//		int sum = (int)((cb - 128)*(cb - 128) + (cr - 128)*(cr - 128));
//
//		if (abs(max_rgb - min_rgb)<rgb_th->th1_upper&&mean <= rgb_th->th2_upper&&mean >= rgb_th->th2_lower&&sum<ycbcr_th->th1_upper*ycbcr_th->th1_upper&& y >= ycbcr_th->th2_lower&&y <= ycbcr_th->th2_upper)
//		{
//			colorMask[offset] = 255;
//			if (inMask[offset] > 0)
//			{
//				outMask[offset] = 255;
//			}
//			else outMask[offset] = 0;
//
//		}
//		else
//		{
//			colorMask[offset] = 0;
//			outMask[offset] = 0;
//		}
//	}
//}



//__global__ void Filter_Kenrel(uchar *d_R, uchar *d_G, uchar *d_B, uchar *acc_threashold,
//						      uchar *inMask, bool beUseColor, bool beUseAcc, Threshold *ycbcr_th, Threshold *rgb_th,
//	                          int width, int height,
//	                          uchar *outMask)
//{
//	int x = threadIdx.x + blockIdx.x * blockDim.x;
//	int y = threadIdx.y + blockIdx.y * blockDim.y;
//
//	if (x < width && y < height)
//	{
//
//		int offset = x + y * width;
//		uchar r, g, b;
//		r = d_R[offset];
//		g = d_G[offset];
//		b = d_B[offset];
//
//		int max_rgb, min_rgb;
//		max_rgb = r>g ? r : g;
//		max_rgb = max_rgb>b ? max_rgb : b;
//
//		min_rgb = r<g ? r : g;
//		min_rgb = min_rgb<b ? min_rgb : b;
//		int mean = (r + g + b) / 3;
//
//		uchar y = 0.257*(float)r + 0.504*(float)g + 0.098*(float)b + 16;
//		uchar cb = -0.148*(float)r - 0.291*(float)g + 0.439*(float)b + 128;
//		uchar cr = 0.439*(float)r - 0.368*(float)g - 0.071*(float)b + 128;
//
//		int sum = (int)((cb - 128)*(cb - 128) + (cr - 128)*(cr - 128));
//
//		if (beUseColor)
//		{
//			if (inMask[offset]>0 && abs(max_rgb - min_rgb)<rgb_th->th1_upper&&mean <= rgb_th->th2_upper&&mean >= rgb_th->th2_lower&&sum<ycbcr_th->th1_upper*ycbcr_th->th1_upper&& y >= ycbcr_th->th2_lower&&y <= ycbcr_th->th2_upper)
//			{
//				outMask[offset] = 255;
//			}
//			else outMask[offset] = 0;
//		}
//		if (beUseAcc)
//		{
//			if (acc_threashold[offset] > 0)
//			{
//				outMask[offset] = 0;
//			}
//		}
//		
//	}
//}

extern "C" void ColorFilter_Caller( const Mat &colorMat, const Mat &inMask,
									const uint32_t rgb_th1_upper,
									const uint32_t rgb_th2_lower,
									const uint32_t rgb_th2_upper,
									const uint32_t ycbcr_th1_upper,
									const uint32_t ycbcr_th2_lower,
									const uint32_t ycbcr_th2_upper,
								    const int width, const int height,
									Mat &outMask, Mat &colorMask)
{

	size_t memSize = width*height*sizeof(uchar);
	Mat rgb_mat_list[3];
	split(colorMat, rgb_mat_list);

	if (outMask.empty())
	{
		outMask = Mat::zeros(cv::Size(colorMat.cols, colorMat.rows), CV_8UC1);
	}
	if (colorMask.empty())
	{
		colorMask = Mat::zeros(cv::Size(colorMat.cols, colorMat.rows), CV_8UC1);
	}

	uchar *d_RMat = NULL;
	uchar *d_GMat = NULL;
	uchar *d_BMat = NULL;

	uchar *d_InMask = NULL;
	uchar *d_OutMask = NULL;
	uchar *d_ColorMask = NULL;

	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_RMat, memSize));
	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_GMat, memSize));
	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_BMat, memSize));

	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_InMask, memSize));
	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_OutMask, memSize));
	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_ColorMask, memSize));


	CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_RMat, rgb_mat_list[0].data, memSize, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_GMat, rgb_mat_list[1].data, memSize, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_BMat, rgb_mat_list[2].data, memSize, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_InMask, inMask.data, memSize, cudaMemcpyHostToDevice));


	dim3 threads(32, 32);
	dim3 grids((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);


	ColorFilter_Kenrel << <grids, threads >> >(d_RMat, d_GMat, d_BMat, d_InMask, 
												rgb_th1_upper, rgb_th2_lower, rgb_th2_upper,
												ycbcr_th1_upper, ycbcr_th2_lower, ycbcr_th2_upper,
												width, height, d_OutMask, d_ColorMask);

	CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(outMask.data, d_OutMask, memSize, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(colorMask.data, d_ColorMask, memSize, cudaMemcpyDeviceToHost));


	cudaFree(d_RMat);
	cudaFree(d_GMat);
	cudaFree(d_BMat);
	cudaFree(d_InMask);
	cudaFree(d_OutMask);
	cudaFree(d_ColorMask);

}

//extern "C" void ColorFilter_Caller(const Mat &colorMat, const Mat &inMask, const Threshold *ycbcr_th, const Threshold *rgb_th, int width, int height, Mat &outMask, Mat &colorMask)
//{
//	size_t memSize = width*height*sizeof(uchar);
//	Mat rgb_mat_list[3];
//	split(colorMat, rgb_mat_list);
//
//	if (outMask.empty())
//	{
//		outMask = Mat::zeros(cv::Size(colorMat.cols, colorMat.rows), CV_8UC1);
//	}
//	if (colorMask.empty())
//	{
//		colorMask = Mat::zeros(cv::Size(colorMat.cols, colorMat.rows), CV_8UC1);
//	}
//
//	uchar *d_RMat = NULL;
//	uchar *d_GMat = NULL;
//	uchar *d_BMat = NULL;
//
//	uchar *d_InMask = NULL;
//	uchar *d_OutMask = NULL;
//	uchar *d_ColorMask = NULL;
//
//	Threshold *d_ycbcr_th;
//	Threshold *d_rgb_th;
//
//	cudaMalloc((void**)&d_RMat, memSize);
//	cudaMalloc((void**)&d_GMat, memSize);
//	cudaMalloc((void**)&d_BMat, memSize);
//
//	cudaMalloc((void**)&d_InMask, memSize);
//	cudaMalloc((void**)&d_OutMask, memSize);
//	cudaMalloc((void**)&d_ColorMask, memSize);
//
//	cudaMalloc((void**)&d_ycbcr_th, sizeof(Threshold));
//	cudaMalloc((void**)&d_rgb_th, sizeof(Threshold));
//
//	cudaError err;
//	err=cudaMemcpy(d_RMat, rgb_mat_list[0].data, memSize, cudaMemcpyHostToDevice);
//	err = cudaMemcpy(d_GMat, rgb_mat_list[1].data, memSize, cudaMemcpyHostToDevice);
//	err = cudaMemcpy(d_BMat, rgb_mat_list[2].data, memSize, cudaMemcpyHostToDevice);
//
//
//	err = cudaMemcpy(d_InMask, inMask.data, memSize, cudaMemcpyHostToDevice);
//
//
//	err = cudaMemcpy(d_ycbcr_th, ycbcr_th, sizeof(Threshold), cudaMemcpyHostToDevice);
//	err = cudaMemcpy(d_rgb_th, rgb_th, sizeof(Threshold), cudaMemcpyHostToDevice);
//	
//
//	dim3 threads(32, 32);
//	dim3 grids((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
//
//	ColorFilter_Kenrel <<<grids, threads >>>(d_RMat, d_GMat, d_BMat, d_InMask, d_ycbcr_th, d_rgb_th, width, height, d_OutMask, d_ColorMask);
//
//	err = cudaMemcpy(outMask.data, d_OutMask, memSize, cudaMemcpyDeviceToHost);
//	err = cudaMemcpy(colorMask.data, d_ColorMask, memSize, cudaMemcpyDeviceToHost);
//
//
//	cudaFree(d_RMat);
//	cudaFree(d_GMat);
//	cudaFree(d_BMat);
//	cudaFree(d_InMask);
//	cudaFree(d_OutMask);
//	cudaFree(d_ColorMask);
//	cudaFree(d_ycbcr_th);
//	cudaFree(d_rgb_th);
//
//}
