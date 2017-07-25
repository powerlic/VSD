#pragma once
#include "Background.h"
#include"cvpreload.h"



using namespace std;
using namespace cv;

extern "C" void ProcessingFirstFrame_Caller(Mat imageMat, Mat *samplelist, int numSamples);
extern "C" void TestAndUpadateDouble_Caller(Mat imageMat, Mat &maskMat, Mat &maskMat2, Mat &utifiedMat, Mat *samplelist, Mat *samplelist_2, int numSamples, int minMatch, int radius, int subSampleFactor, int max_mismatch_count, Mat &foregroundMatchCount, Mat &foregroundMatchCount_2);
extern "C" void TestAndUpadateSingle_Caller(Mat imageMat, Mat &maskMat, Mat *samplelist, int numSamples, int minMatch, int radius, int subSampleFactor, int max_mismatch_count, Mat &foregroundMatchCount);
class BgVibe :public Background
{
public:
	BgVibe();
	~BgVibe();
	BgVibe(const char *ini_file_name, const char *section_name);
	BgVibe(const char *file_name);
	BgVibe(const vas::BgParameter &other);
	BgVibe( const CvSize &bg_size,
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
	void SetBg2Delay(uint32_t value) { bg_parameter_.mutable_vibe_parameter()->set_bg2_delay(value);}

	void SetBgSize(const CvSize& value);

	uint32_t MaxMismatchCount(){ return  bg_parameter_.vibe_parameter().max_mismatch_count(); }
	void SetMaxMismatchCount(const uint32_t &value){ bg_parameter_.mutable_vibe_parameter()->set_max_mismatch_count(value); }

private:

	Mat *sample_list1_;
	Mat *sample_list2_;
	Mat foreground_match_count_mat1_;
	Mat foreground_match_count_mat2_;

	bool bg_setup1_;
	bool bg_setup2_;

	Mat mask1_;
	Mat mask2_;


	void UpdateSingle(const Mat &frame);
	void InitSingle();

	void InitDouble();
	void UpdateDouble(const Mat &frame);

	void Reset();
	void ProcessFirstFrameCPU(const Mat& gray_frame, Mat *sample_list);
	void TestAndUpdateCPU(const Mat& gray_frame);
	void TestAndUpdateSingleCPU(const Mat& gray_frame);
	inline void UpdateSamplist(uchar matches, uchar pixel_value, int x, int y, Mat &foreground_match_count_mat,Mat &mask, Mat *sample_list);

	int off_[9];

	void CheckParameter(const vas::BgParameter &param);
};

