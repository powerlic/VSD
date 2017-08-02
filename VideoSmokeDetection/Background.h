#pragma once
#include"VasProto.prototxt.pb.h"
#include"cvpreload.h"
using namespace std;
using namespace cv;

class Background
{
public:
	Background();
	virtual ~Background()=0;

	virtual void Init() = 0;
	virtual void Update(const Mat &frame) = 0;

	Mat& Mask(){ return mask_; }
	
	inline CvSize& BgSize(){ return cvSize(bg_parameter_.bg_width(), bg_parameter_.bg_height()); }
	virtual void SetBgSize(const CvSize& value)=0;
	inline const vas::BgStatus& Status(){ return bg_parameter_.bg_status(); }

	float StableThreahold() { return bg_parameter_.stable_threshold(); }
	void  SetStableThreahold(float value){ bg_parameter_.set_stable_threshold(value); }
	bool  Stable(const Mat &mask);

	inline uint32_t MorphologyOpenTimes(){ return bg_parameter_.bg_operation().morphology_open_times(); }
	inline uint32_t DilateTimes(){ return bg_parameter_.bg_operation().dilate_times(); }

	void SetMorphologyOpenTimes(const uint32_t &value){ bg_parameter_.mutable_bg_operation()->set_morphology_open_times(value); }
	void SetDilateTimes(const uint32_t &value){ bg_parameter_.mutable_bg_operation()->set_dilate_times(value); }

	inline const vas::BgMethod& BgMethod()
	{
		return bg_parameter_.bg_method();
	}

	inline const vas::BgStatus &BgStatus()
	{
		return bg_parameter_.bg_status();
	}
	void SetStatus(vas::BgStatus value)
	{
		bg_parameter_.set_bg_status(value);
	}

protected:
	Mat mask_;
	uint32_t frame_count_;
	bool stable_;	
	vas::BgParameter bg_parameter_;
	void MaskOperation(Mat &mask);

};

