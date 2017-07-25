#pragma once
#include "BgVibe.h"
#include "BgGaussian.h"
#include "SmokeFilter.h"
#include "LicImageOperation.h"
#include "concurrent_queue.h"
#include "IClassification.h"


class SmokeDetecterPro
{
public:
	SmokeDetecterPro();
	SmokeDetecterPro(const char*file_name,const string &set_service_id);
	~SmokeDetecterPro();
	/*-----------------------------------Set Get Method-------------------------------*/
	const string &DetectorParameterString();



	void SetServiceId(const string &str){ service_id_ = str; }
	const string &ServiceId()
	{
		return service_id_;
	}

	void SetRegSize(const CvSize value)
	{
		reg_size_ = value;
	}
	CvSize RegSize(){ return reg_size_; }

	void SetSavePath(const string value)
	{
		save_path_ = value;
	}
	const string SavePath()
	{
		return save_path_;
	}

	void SetSkipRectList(const vector<CvRect> &value)
	{
		skip_rect_list_ = value;
	}

	const vector<CvRect>& SkipRectList()
	{
		return skip_rect_list_;
	}

	void SetSaveRegImage(bool value)
	{
		save_reg_image_ = value;
	}
	bool SaveRegImage(){ return save_reg_image_; }

	//BgModel
	void SetBgSize(const CvSize &value)
	{
		bg_size_ = value;
		ptr_vibe_->SetBgSize(value);
		ptr_bgGaussian_->SetBgSize(value);
	}
	const CvSize& BgSize()
	{
		return bg_size_;
	}

	void SetBgMethod(BgMethod method)
	{
		bg_method_ = method;
	}
	BgMethod GetBgMethod(){ return bg_method_; }


	void SetCaffeRegFunction(std::function<vector<Prediction>(const Mat&, const int&)> set_fun);

	void ProcessFrame(const Mat &frame, vector<CvRect> &rects, vector<String> &labels, vector<float> &ratios);
	void ProcessFrame(const Mat &frame, const String &label, vector<CvRect> &rects, vector<float> &ratios);
	void ProcessFrame(const Mat &frame, vector<CvRect> &rects, vector<float> &ratios);

protected:
	bool UpdateDecter(const Mat &frame);
	uint8_t video_reg_frame_interval_ = 1;
	string service_id_;
	CvSize reg_size_=cvSize(800,576);
	bool save_reg_image_ = false;
	string save_path_;

	Mat raw_frame_;
	Mat reg_frame_;
	Mat show_frame_;
	Mat reg_gray_frame_;

	void InitGeneralSet(const char *file_name);

	vector<CvRect> skip_rect_list_;

	//Background Model
	Mat fore_mask_;
	CvSize bg_size_;
	BgMethod bg_method_;
	shared_ptr<BgVibe> ptr_vibe_;
	shared_ptr<BgGaussian> ptr_bgGaussian_;

	bool GetForeMask();
	void InitBgModel(const char *file_name);

	//Filter
	bool use_filter_ = true;
	shared_ptr<SmokeFilter> ptr_smoke_filter;
	void InitSmokeFilter(const char*file_name);

	//CaffeModelInfo
	string model_version_;
	float confience_probability_;
	int caffe_input_size_;
	void InitCaffeModelInfo(const char *file_name);
	std::function<vector<Prediction>(const Mat&, const int&)> caffe_reg_fun;
};


