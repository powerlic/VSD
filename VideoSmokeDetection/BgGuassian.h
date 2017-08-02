#pragma once
#include "Background.h"
#include"cvpreload.h"


using namespace std;
using namespace cv;


class BgGuassian:public Background
{
public:
	BgGuassian();
	~BgGuassian();
	BgGuassian(const char *file_name);
	/*dispareted interface
	*/
	BgGuassian(const char *ini_file_name,const char *section_name);
	BgGuassian(const vas::BgParameter &param);
	BgGuassian(uint32_t num_history, uint32_t var_threshold,
				bool use_shadow_detection, CvSize bg_size,
				float learn_rate, uint32_t op_morphology_open_times, uint32_t op_dilate_times,
				float stable_threahold, bool use_gpu = false);

	void Init();
	void Update(const Mat &frame);
	

	bool SetNumHistory(uint32_t value)
	{
		if (bg_parameter_.has_guassian_parameter())
		{
			bg_parameter_.mutable_guassian_parameter()->set_num_history(value);
			return true;
		}
		return false;
	}
	uint32_t NumHistory(){ return bg_parameter_.guassian_parameter().num_history(); }


	bool SetVarThreahold(uint32_t value)
	{
		if (bg_parameter_.has_guassian_parameter())
		{
			bg_parameter_.mutable_guassian_parameter()->set_var_threshold(value);
			return true;
		}
		return false;
	}
	uint32_t VarThreahold(){ return bg_parameter_.guassian_parameter().var_threshold(); }

	bool SetLearnRate(float value)
	{
		if (bg_parameter_.has_guassian_parameter())
		{
			bg_parameter_.mutable_guassian_parameter()->set_learn_rate(value);
			return true;
		}
		return false;
	}
	float LearnRate(){ return bg_parameter_.guassian_parameter().learn_rate(); }

	bool SetUseShadowDetection(bool value)
	{
		if (bg_parameter_.has_guassian_parameter())
		{
			bg_parameter_.mutable_guassian_parameter()->set_shadow_detection(value);
			return true;
		}
		return false;
	}
	bool ShadowDetection(){ return bg_parameter_.guassian_parameter().shadow_detection(); }

	void SetBgSize(const CvSize& value);

private:
	BackgroundSubtractorMOG2 mog_;
	void UpdateCpu(const Mat &frame);
	void UpdateGpu(const Mat &frame);
	void CheckParameter(const vas::BgParameter &param);
};


