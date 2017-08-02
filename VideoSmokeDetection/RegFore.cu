#include "cuda_runtime.h"
#include "curand.h"
#include "curand_kernel.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <iostream>
#include "cvpreload.h"
using namespace cv;
using namespace std;

__global__ void RegFore_Kenrel(uchar *fore, uchar *color_frame,int width, int height, uchar *reg_frame)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;


	if (x < width && y < height)
	{
		int offset = x + y * width;
		int r_offset = x * 3 + y * 3 * width;
		int g_offset = x * 3 + y * 3 * width + 1;
		int b_offset = x * 3 + y * 3 * width + 2;
		if (*(fore+offset)>0)
		{
			*(reg_frame + r_offset) = *(color_frame + r_offset);
			*(reg_frame + g_offset) = *(color_frame + g_offset);
			*(reg_frame + b_offset) = *(color_frame + b_offset);
		}
		else
		{
			*(reg_frame + r_offset) = 0;
			*(reg_frame + g_offset) = 0;
			*(reg_frame + b_offset) = 0;
		}

	}
}


extern "C" void RegFrame_Caller(const Mat &colorMat, const Mat &foreMask, int width, int height, Mat &RegMat)
{
	Mat resize_fore_mask;
	if (foreMask.size().width != width || foreMask.size().height != height)
	{
		resize(foreMask, resize_fore_mask, cvSize(width, height));
	}
	else resize_fore_mask = foreMask;

	if (RegMat.empty())
	{
		RegMat = Mat::zeros(cvSize(width, height), CV_8UC3);
	}

	size_t memSize = width*height*sizeof(uchar);

	uchar *d_fore_mat = NULL;
	uchar *d_color_mat = NULL;
	uchar *d_reg_mat = NULL;

	cudaMalloc((void**)&d_fore_mat, memSize);
	cudaMalloc((void**)&d_color_mat, 3*memSize);
	cudaMalloc((void**)&d_reg_mat, 3 * memSize);

	cudaError err;
	err = cudaMemcpy(d_fore_mat, resize_fore_mask.data, memSize, cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_color_mat, colorMat.data, 3*memSize, cudaMemcpyHostToDevice);

	dim3 threads(32, 32);
	dim3 grids((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

	RegFore_Kenrel << <grids, threads >> >(d_fore_mat, d_color_mat, width, height, d_reg_mat);

	err = cudaMemcpy(RegMat.data, d_reg_mat, 3*memSize, cudaMemcpyDeviceToHost);

	cudaFree(d_fore_mat);
	cudaFree(d_color_mat);
	cudaFree(d_reg_mat);

}