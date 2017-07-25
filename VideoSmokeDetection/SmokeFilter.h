#pragma once
#include"cvpreload.h"
#include"LicImageOperation.h"
#include"VasData.h"
using namespace std;
using namespace cv;



extern "C" void ColorFilter_Caller(const Mat &colorMat, const Mat &inMask, const Threshold *ycbcr_th, const Threshold *rgb_th, int width, int height, Mat &outMask, Mat &colorMask);


class SmokeFilter
{
public:
	SmokeFilter();
	~SmokeFilter();
	SmokeFilter(const char *file_name, const char *section_name);
	SmokeFilter(CvSize filter_size,
				const uint8_t &filter_mode,
				const int &contour_area_threahold,
				const int &contour_perimeter_threahold,
				const float &area_perimeter_ratio_threahold,
				const uint8_t &frame_list_size,
				const uint8_t &acc_threahold,
				const Threshold &th_rgb,
				const Threshold &th_ycbcr);


	/*-----------------------------------Interface---------------------------------*/
	void Filtrate(const Mat &frame, const Mat &fore_mask, Mat &out_mask, vector<vector<Point>> &contours, vector<CvRect> &rects);


	CvSize FilterSize(){ return filter_size_; }
	void SetFilterSize(const CvSize &value){ filter_size_ = value; Reset(); }

	uint8_t FrameListSize(){ return frame_list_size_; }
	void SetFrameListSize(uint8_t size)
	{
		if (frame_list_size_ != size)
		{
			Reset();
			frame_list_size_ = size;
		}
	}
	void Reset()
	{
		gray_frame_list_.clear();
		gray_diff_frame_list_.clear();
	}

	//ACC
	void SetAccThreahold(int value){ acc_threahold_ = value; }
	int AccThreahold(){ return acc_threahold_; }


	inline void SetContourThreahold(int area, int perimeter, float ratio)
	{   
		contour_area_threahold_ = area; 
		contour_perimeter_threahold_ = perimeter;
	    area_perimeter_ratio_threahold_ = ratio;
	}
	inline int ContourAreaThreahold(){ return contour_area_threahold_; }
	inline int ContourPerimeterThreahold(){ return contour_perimeter_threahold_; }
	inline float AreaPerimeterRatioThreahold(){ return area_perimeter_ratio_threahold_; }

	void SetSmokeFilterMode(SmokeFilterMode mode){ smoke_filter_mode_ = mode; }
	SmokeFilterMode GetSmokeFilterMode(){ return smoke_filter_mode_; }
	/*===================================Interface==================================*/

private:
	/*
	Parameter
	*/
	CvSize filter_size_ = cvSize(420, 320);
	uint8_t frame_list_size_=10;
	int contour_area_threahold_=500;
	int	contour_perimeter_threahold_=10;
	float area_perimeter_ratio_threahold_=0.001;
	uint8_t acc_threahold_=50;
	float acc_move_ratio_threahold_ = 0.5;
	Threshold th_rgb_;
	Threshold th_ycbcr_;


	bool UpdateFilter(const Mat &frame, const Mat &fore_mask);

	void FrameFiltrate(const Mat &frame, const Mat &fore_mask, Mat &out_mask, vector<vector<Point>> &contours, vector<CvRect> &rects);
	void FrameFiltrateGPU(const Mat &frame, const Mat &fore_mask, Mat &out_mask, vector<vector<Point>> &contours, vector<CvRect> &rects);
	void ContourFiltrate(const Mat &frame, const Mat &fore_mask, Mat &out_mask, vector<vector<Point>> &contours, vector<CvRect> &rects);
	void PureContourFiltrate(const Mat &frame, const Mat &fore_mask, Mat &out_mask, vector<vector<Point>> &contours, vector<CvRect> &rects);
	void FiltrateContourCombined(const Mat &frame, const Mat &fore_mask, Mat &out_mask, vector<vector<Point>> &contours, vector<CvRect> &rects);
	void FiltrateContourColor(const Mat &frame, const Mat &fore_mask, Mat &out_mask, vector<vector<Point>> &contours, vector<CvRect> &rects);
	void FiltrateContourAcc(const Mat &frame, const Mat &fore_mask, Mat &out_mask, vector<vector<Point>> &contours, vector<CvRect> &rects);

	/*-----------------------------------GeneralSet---------------------------------*/
	Mat filter_frame_;
	Mat filter_fore_mask_;
	Mat gray_filter_frame_;
	
	SmokeFilterMode smoke_filter_mode_ = PURE_CONTOUR_AREA_MODE;
	/*===================================GeneralSet=================================*/


	/*-----------------------------------ContourFilter------------------------------*/
	
	void InitContourThreahold(const char *file_name,const char *section_name);
	/*===================================ContourFilter===============================*/


	/*-----------------------------------ACCFilter---------------------------------*/
	list<Mat> gray_frame_list_;
	list<Mat> gray_diff_frame_list_;
	Mat diff_acc_mask_;
	Mat diff_acc_frame_;

	
	void InitAccThreahold(const char *file_name, const char *section_name);
	void AccumulateDiffImage(const Mat &gray_frame);
	
	/*===================================ACCFilter==================================*/

	/*----------------------------------ColorFilter---------------------------------*/
	Mat color_mask_;
	float rbg_fit_move_ratio_ = 0.5;
	void InitColorFilter(const char *file_name, const char *section_name);
	inline bool PixelColorFit(const uchar &r, const uchar &g, const uchar &b);
	void ColorPixelFilter(const Mat &color_image, const Mat &in_mask, Mat&out_mask);
	/*=================================ColorFilter==================================*/
};

