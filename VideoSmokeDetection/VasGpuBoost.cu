#pragma once
#include"VasGpuBoost.h"
namespace VasGpuBoost
{
	__global__ void YuvToRgb(const uchar* yuv_data, uchar *rgb_data, const int width, const int height, const uchar *table_r, const uchar *table_g, const uchar *table_b)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int size = width*height;

		int x, y;
		x = i % width;
		y = i / width;
		int uv_size = (width / 2 + width % 2)*(height / 2 + height % 2);
		int yuv_width = width / 2 + width % 2;
		CUDA_KERNEL_LOOP(i, size)
		{
			int u_offset = size + (y / 2)*yuv_width + x / 2;
			int v_offset = size + uv_size + (y / 2)*yuv_width + x / 2;

			int Y = *(yuv_data + i);
			int U = *(yuv_data + u_offset);
			int V = *(yuv_data + v_offset);


			*(rgb_data + 3 * i) = table_r[(Y << 8) + V];
			*(rgb_data + 3 * i + 1) = table_g[(Y << 16) + (U << 8) + V];
			*(rgb_data + 3 * i + 2) = table_b[(Y << 8) + U];

		
		}
	}

	void ColorConvert::ConvertYUV422pToRGB24(const uchar* yuv_data, uchar *rgb_data, const int width, const int height)
	{
		
		if (width*height>size_)
		{
			UpdateMemSize(width, height);
		}
		boost::mutex::scoped_lock lock(convert_mutex_);
		CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_yuv_data_, yuv_data, yuv_mem_size_ * sizeof(uchar), cudaMemcpyHostToDevice));
		YuvToRgb << <GET_BLOCKS(size_), CUDA_NUM_THREADS >> >(d_yuv_data_, d_rgb_data_, width, height, d_table_r_, d_table_g_, d_table_b_);
		CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(rgb_data, d_rgb_data_, rgb_mem_size_ * sizeof(uchar), cudaMemcpyDeviceToHost));
		lock.unlock();


	}
}

