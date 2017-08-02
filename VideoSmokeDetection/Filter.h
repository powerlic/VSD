#pragma once
#include"cvpreload.h"
#include"LicImageOperation.h"
#include"VasProto.prototxt.pb.h"
#include"VasIO.h"

using namespace std;
using namespace cv;

extern "C" void ColorFilter_Caller( const Mat &colorMat, const Mat &inMask,
									const uint32_t rgb_th1_upper,
									const uint32_t rgb_th2_lower,
									const uint32_t rgb_th2_upper,
									const uint32_t ycbcr_th1_upper,
									const uint32_t ycbcr_th2_lower,
									const uint32_t ycbcr_th2_upper,
									const int width, const int height,
									Mat &outMask, Mat &colorMask);

class Filter
{
public:
	Filter();
	Filter(const vas::FilterParameter &param);
	Filter(const char* file_nmae);

	void CheckParameter(const vas::FilterParameter &param);

	virtual ~Filter();
	
	/*-----------------------------------Interface---------------------------------*/
	/*@brief: filterate a frame to extract contours and rects using assigned method, default method is PURE_CONTOUR_AREA, which only filterate "small" fore areas.
	*note: the out_mask is same size with frame
	*frame[in] frame before filteration
	*fore_mask[in] mask for filteration
	*out_mask[out] output mask
	*contours[out]
	*rects[out]
	*
	*/

	void Filtrate(const Mat &frame, const Mat &fore_mask, Mat &out_mask, vector<vector<Point>> &contours, vector<CvRect> &rects);


protected:

	void Update(const Mat &frame, const Mat &fore_mask);

	vas::FilterParameter filter_parameter_;

	void PureContourFiltrate(const Mat &frame, const Mat &fore_mask, const CvSize &dst_size, Mat &out_mask, vector<vector<Point>> &contours, vector<CvRect> &rects);
	void ContourColorFiltrate(const Mat &frame, const Mat fore_mask, const CvSize &dst_size, Mat &out_mask, vector<vector<Point>> &contours, vector<CvRect> &rects);

	void FrameColorFiltrateCPU(const Mat &frame, const Mat fore_mask, const CvSize &dst_size,  Mat &out_mask, vector<vector<Point>> &contours, vector<CvRect> &rects);
	void FrameColorFiltrateGPU(const Mat &frame, const Mat fore_mask, const CvSize &dst_size, Mat &out_mask, vector<vector<Point>> &contours, vector<CvRect> &rects);

	bool PixelColorFit(const uchar &r, const uchar &g, const uchar &b);

	Mat color_mask_;
	Mat filter_frame_;
	Mat filter_fore_mask_;

	boost::mutex filter_mutex;

};

