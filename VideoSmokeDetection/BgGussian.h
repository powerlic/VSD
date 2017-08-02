#pragma once
#include"cvpreload.h"
using namespace std;
using namespace cv;

enum BgGussianStatus
{
	SETUP,
	UPDATING
};
class BgGussian
{
public:
	BgGussian();
	~BgGussian();
	BgGussian(const char *file_name, const char *section_name);
	BgGussianStatus BgStatus(){ return bg_status_; }
	Mat& Mask(){ return mask_; }
	void Update(const Mat& frame);

	void SetNumHistory(int value)
	{
		num_history_ = value;
	}
	int NumHistory(){ return num_history_; }

	void SetVarThreahold(int value)
	{
		var_threahold_ = value;
	}
	int VarThreahold(){ return var_threahold_; }

	void SetBgCountThreahold(int value)
	{
		bg_count_threahold_ = value;
	}
	int BgCountThreahold(){ return bg_count_threahold_; }

	void SetLearnRate(float value)
	{
		learn_rate_ = value;
	}
	float LearnRate(){ return learn_rate_; }

	void SetBgSize(CvSize size)
	{
		bg_size_ = size;
	}
	CvSize BgSize(){ return bg_size_; }

	void SetUseShadowDetection(bool value)
	{
		use_shadow_detection_ = value;
	}
	bool UseShadowDetection(){ return use_shadow_detection_; }

private:
	uint32_t frame_count_;
	int num_history_;
	int var_threahold_;
	bool use_shadow_detection_;
	int bg_count_threahold_;
	float learn_rate_;
	CvSize bg_size_;
	BackgroundSubtractorMOG2 mog_;
	BgGussianStatus bg_status_;
	Mat mask_;
	int morphologyEx_times_;
	int dilate_times_;
	bool BeImageStable();
};

