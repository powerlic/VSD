#pragma once
#include "Background.h"
#include "cvpreload.h"
#include "cuda_common.h"


using namespace std;
using namespace cv;

#define MaxMinMatch 7
#define RandNum 7

class BgVibeBoost :public Background
{
public:
	BgVibeBoost();
	~BgVibeBoost();
	BgVibeBoost(const char *file_name);
	BgVibeBoost(const vas::BgParameter &other);
	BgVibeBoost(const CvSize &bg_size,
			    const uint32_t morphology_open_times, const uint32_t dilate_times,
			    const uint32_t num_samples, const uint32_t min_match,
				const uint32_t radius,
				const uint32_t sub_sample_factor,
				const bool use_gpu,
				const float stable_threahold,
				const uint32_t bg2_delay,
				const uint32_t max_mismatch_count,
				const bool double_bg);
	void Init();
	void Update(const Mat &gray_frame);

	inline int NumSamples(){ return bg_parameter_.vibe_parameter().num_samples(); }
	void SetNumSamples(uint32_t value)
	{
		if (value != bg_parameter_.vibe_parameter().num_samples())
		{
			bg_parameter_.mutable_vibe_parameter()->set_num_samples(value);
			bg_parameter_.set_bg_status(vas::BG_UNINIALIZED);
		}
	}
	inline uint32_t MinMatch(){ return bg_parameter_.vibe_parameter().min_match(); }
	void SetMinMatch(uint32_t value) { bg_parameter_.mutable_vibe_parameter()->set_min_match(value); }

	inline uint32_t Radius(){ return bg_parameter_.vibe_parameter().radius(); };
	void SetRadius(uint32_t value) { bg_parameter_.mutable_vibe_parameter()->set_radius(value); }

	inline uint32_t SubSampleFactor(){ return bg_parameter_.vibe_parameter().subsample_factor(); }
	void SetSubSampleFactor(uint32_t value) { bg_parameter_.mutable_vibe_parameter()->set_subsample_factor(value); }

	inline uint32_t Bg2Delay(){ return bg_parameter_.vibe_parameter().bg2_delay(); }
	void SetBg2Delay(uint32_t value) { bg_parameter_.mutable_vibe_parameter()->set_bg2_delay(value); }

	void SetBgSize(const CvSize& value);

	uint32_t MaxMismatchCount(){ return  bg_parameter_.vibe_parameter().max_mismatch_count(); }
	void SetMaxMismatchCount(const uint32_t &value){ bg_parameter_.mutable_vibe_parameter()->set_max_mismatch_count(value); }

	Mat &mask1();
	Mat &mask2();


private:
	uchar *sample_list1_=NULL;
	uchar *sample_list2_ = NULL;	
	uchar * foreground_match_count_mat1_ = NULL;
	uchar * foreground_match_count_mat2_ = NULL;

	bool bg_setup1_ = false;
	bool bg_setup2_ = false;

	Mat mask1_;
	Mat mask2_;

	//GPU 
	uchar *d_mask1_ = NULL;
	uchar *d_mask2_ = NULL;
	uchar *d_mask_ = NULL;
	uchar *d_sample_list1_ = NULL;
	uchar *d_sample_list2_ = NULL;
	uchar *d_foreground_match_count_mat1_ = NULL;
	uchar *d_foreground_match_count_mat2_ = NULL;
	uchar *d_frame_ = NULL;


	void InitSingle();
	void UpdateSingle(const Mat &frame);
	
	void InitDouble();
	void UpdateDouble(const Mat &frame);

	void Reset();
	void ProcessFirstFrameCPU(const Mat& gray_frame, uchar *sample_list);
	void ProcessFirstFrameGPU(const Mat& gray_frame, uchar *d_sample_list);


	void TestAndUpdateSingleCPU(const Mat& gray_frame);
	void TestAndUpdateSingleGPU(const Mat& gray_frame);

	void TestAndUpdateDoubleCPU(const Mat& gray_frame);
	void TestAndUpdateDoubleGPU(const Mat& gray_frame);

	
	inline void UpdateSamplist(uchar matches, uchar pixel_value, int x, int y, uchar *foreground_match_count_mat, uchar *mask, uchar *sample_list);

	void CheckParameter(const vas::BgParameter &param);

	int size_;
	int off_[9];
};

