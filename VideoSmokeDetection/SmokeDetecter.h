#pragma once
//#include "VASClassifier.h"
#include "Vibe.h"
#include "LicImageOperation.h"
#include "concurrent_queue.h"
#include "IClassification.h"


struct Contour
{
	CvPoint center;
	int area;
	vector<Point> contour;
	
};



extern "C" void ColorFilter_Caller(const Mat &colorMat, const Mat &inMask, const Threahold *ycbcr_th, const Threahold *rgb_th, int width, int height, Mat &outMask);
extern "C" void Filter_Caller(const Mat &colorMat, const Mat &accThresholdMat, const Mat &inMask, const Threahold *ycbcr_th, const Threahold *rgb_th, int width, int height, Mat &outMask, Mat &accMask, bool useColorFilter, bool useAccFilter);

class SmokeDetecter
{
public:
	/*
	SmokeDetecter(CvSize set_reg_size,
		CvSize set_bg_size,
		int set_frame_list_size,
		int set_contour_area_threadhold,
		int set_contour_perimeter_threadhold,
		int set_sensitivity_degree,
		string set_model_version,
		shared_ptr<VASClassifierPool> ptr_vas_classifier_pool);*/
	SmokeDetecter(CvSize set_reg_size,
		          CvSize set_bg_size,
		          int set_frame_list_size,
		          int set_contour_area_threadhold,
		          int set_contour_perimeter_threadhold,
		          int set_sensitivity_degree,
		          string set_service_id,
		          string set_model_version);
	~SmokeDetecter();
	void ProcessFrame(const Mat &frame_mat, vector<CvRect> &rect_list);
	void SetFrame(const Mat &set_frame);
	void SetCaffeRegFunction(std::function<vector<Prediction>(const Mat&, const int&)> set_fun);

private:
	
	void InitSavePath(const char *file_name);
	bool is_save_image;
	string full_save_path_str;

	//vibe
	shared_ptr<ViBeExtrator> ptr_vibe_extrator;
	int frame_delay;

	string service_id;

	BackgroundSubtractorMOG2 mog;
	Mat foreground;
	Mat background;

	CvSize reg_size;
	CvSize bg_size;

	Mat frame;
	Mat gray_frame;
	Mat mask;
	
	Mat acc_mask;
	Mat acc_mat;
	Mat acc_threashold;
	Mat first_last_diff;
	list<Mat> mat_list;
	list<vector<Contour>> contours_record;
	list<Mat> diff_mat_list;
	list<Mat> mask_list;
	//concurrent_queue<Mat> mat_list;
	uint8_t frame_list_size;
	uint8_t acc_mask_delay;
	uint8_t acc_count;
	void AccumulateDiffImage();
	void CalculateAccMask();
	void FilterByAccMask();
	void UpdateSmokeAreaRecord(vector<CvRect> rect_list, vector<vector<Point>> contours);

	void UpdateMaskList();
	bool BeMaskListRectConstantVariation(const CvRect &rect, const float &ratio);
	bool BeMaskListForeBackgroundUnique(const CvRect &rect, const float &ratio);

	int aplha;
	int ycbcr_th3, ycbcr_th4, rgb_th1, rgb_th2, rgb_th3;
	Mat color_mask;
	void ColorCriteriaRGBandYCbCr();
	void FilterByColor();
	inline bool PixelColorFit(uchar r, uchar g, uchar b);

	float CalculateForeAndBackHisDis(const Mat &gray_mat,const CvRect &rect, const vector<Point> &contour);


	vector<CvRect> not_roi_rect_list;
	vector<CvRect> roi_rect_list;

	int contour_area_threadhold;
	int contour_perimeter_threadhold;

	//灵敏度
	uint8_t sensitivity_degree;

	//深度学习分类器
	//shared_ptr<VASClassifier> ptr_vas_classifier;
	string model_version;
	float confidence_probability;

	void FindCorners(const Mat &image, const Mat &gray_image, Mat &show_image);
	

	//保存识别照片
	void SaveRegImage(const Mat &reg_mat);
	std::function<vector<Prediction>(const Mat&, const int&)> caffe_reg_fun;

};

