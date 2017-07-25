#include <stdio.h>
#include<iostream>


#include "cvpreload.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>

#define ImgWidth 800;
#define ImgHeight 600
#define ImgListSize 20
#define Var_Threadhold 10
#define Var_Threadhold_Gray 5
#define Mean_Threadhold 130

using namespace cv;


__global__ void MeanAndVal_kernel(const uchar* dImagelistData, const uchar* dMask, const int width, const int height,const uchar img_list_size, const uchar var_threshold,
	const uchar var_gray_threashold, const uchar meanThreashold,
	uchar *d_varData, uchar1 *d_varImgData, uchar *d_meanImgData, uchar *dOutMask)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int pixelLoc;
	int pixelImglistLoc;
	int imgNum;
	float tempVal;
	float meanVal;
	float secMomData;
	float varVal;
	if (x < width && y < height)
	{
		meanVal = 0;
		secMomData = 0;
		varVal = 0;
		pixelLoc = x + y*width;
		if (dMask[pixelLoc]>0)
		{
			for (imgNum = 0; imgNum < ImgListSize; imgNum++)
			{
				const uchar *pData = (const uchar*)(dImagelistData + imgNum*width*height);
				tempVal = (float)pData[pixelLoc];
				meanVal += tempVal / ImgListSize;
				secMomData += tempVal*tempVal / ImgListSize;
			}
			varVal = secMomData - meanVal*meanVal;
			varVal = sqrt(varVal);
		}
		
		
		d_varData[pixelLoc] = 8 * varVal;

		if (varVal>=Var_Threadhold)
		{
			d_varImgData[pixelLoc] = 255;
		}
		else d_varImgData[pixelLoc] = 0;
		if (meanVal>=Mean_Threadhold)
		{
			d_meanImgData[pixelLoc]= 255;
		}
		else d_meanImgData[pixelLoc] = 0;

		if (varVal>1 && dMask[pixelLoc]>0)
		{
			if (varVal>Var_Threadhold_Gray)
			{
				if (varVal>=Var_Threadhold&&meanVal>=Mean_Threadhold)
				{
					dOutMask[pixelLoc] = 255;
				}
			}
			
		}
		else dOutMask[pixelLoc] = 0;
	}
}

extern "C" void MeanAndVar_caller(const uchar *imglist, const uchar *moveObjMask, int width, int height,
	IplImage *pMeanImg, IplImage *pVarDataImg, IplImage *pVarImgData, IplImage *pMaskImgData,float &varPointsRatio, float &meanPointsRatio)
{
	
	size_t memSize = width*height*sizeof(uchar);

	time_t t1,t2;
	//  uchar1 *d_meanData, uchar1 *d_varData,
	//  uchar1 *d_varDataImg, uchar1 *d_varImgData, uchar1 *d_meanImgData, uchar1 *d_maskImgData
	uchar1 *d_imglist = NULL;
	uchar1 *d_moveObjMask = NULL;
	//uchar1 *d_meanData = NULL;
	uchar1 *d_varData = NULL;
	//uchar1 *d_varDataImg = NULL;
	uchar1 *d_varImgData = NULL;
	uchar1 *d_meanImgData = NULL;
	uchar1 *d_maskImgData = NULL;

	cudaMalloc((void**)&d_imglist, memSize*ImgListSize);
	cudaMalloc((void**)&d_moveObjMask, memSize);

	cudaMalloc((void**)&d_varData, memSize);
	cudaMalloc((void**)&d_varImgData, memSize);
	cudaMalloc((void**)&d_meanImgData, memSize);
	cudaMalloc((void**)&d_maskImgData, memSize);

	dim3 threads(32, 32);
	dim3 grids((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
	t1 = clock();
	cudaMemcpy(d_imglist, imglist, ImgListSize*memSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_moveObjMask, moveObjMask, memSize, cudaMemcpyHostToDevice);

	MeanAndVal_kernel <<<grids, threads>>>(d_imglist, d_moveObjMask, width, height,
		 d_varData, d_varImgData, d_meanImgData, d_maskImgData);
	cudaThreadSynchronize();


	Mat varData = Mat(Size(width, height), CV_8UC1, Scalar(0));
	Mat varImgData = Mat(Size(width, height), CV_8UC1, Scalar(0));
	Mat meanImgData = Mat(Size(width, height), CV_8UC1, Scalar(0));
	Mat maskImgData = Mat(Size(width, height), CV_8UC1, Scalar(0));

	cudaMemcpy(varData.data, d_varData, memSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(varImgData.data, d_varImgData, memSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(meanImgData.data, d_meanImgData, memSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(maskImgData.data, d_maskImgData, memSize, cudaMemcpyDeviceToHost);

	IplImage *img1 = cvCreateImage(meanImgData.size(), IPL_DEPTH_8U, 1);
	IplImage imgTmp1 = varImgData;
	img1 = cvCloneImage(&imgTmp1);
	cvCopy(img1, pVarImgData);



	IplImage *img2 = cvCreateImage(meanImgData.size(), IPL_DEPTH_8U, 1);
	IplImage imgTmp2 = varData;
	img2 = cvCloneImage(&imgTmp2);
	cvCopy(img2, pVarDataImg);

	IplImage *img3 = cvCreateImage(meanImgData.size(), IPL_DEPTH_8U, 1);
	IplImage imgTmp3 = meanImgData;
	img3 = cvCloneImage(&imgTmp3);
	cvCopy(img3, pMeanImg);


	IplImage *img4 = cvCreateImage(meanImgData.size(), IPL_DEPTH_8U, 1);
	IplImage imgTmp4 = maskImgData;
	img4 = cvCloneImage(&imgTmp4);
	cvCopy(img4, pMaskImgData);

	//test
	//imshow("varImg ", varImgData);
	//imshow("varData ", varData);
	//imshow("meanImg ", meanImgData);
	//imshow("maskImgData ", maskImgData);
	cudaFree(d_imglist);
	cudaFree(d_moveObjMask);

	cudaFree(d_varData);
	cudaFree(d_varImgData);
	cudaFree(d_meanImgData);
	cudaFree(d_maskImgData);

	t2 = clock();

	varPointsRatio = (float)g_cuda_varPoints / (width*height);
	meanPointsRatio = (float)g_cuda_meanPoints / (width*height);

	std::cout << "mean and var time GPU " << t2 - t1 << std::endl;

	cvReleaseImage(&img1);
	cvReleaseImage(&img2);
	cvReleaseImage(&img3);
	cvReleaseImage(&img4);


}

