#include "stdafx.h"
#include "BgGuassian.h"
#include "VasIO.h"


BgGuassian::BgGuassian()
{
	bg_parameter_ = vas::BgParameter();
	bg_parameter_.mutable_bg_operation();
	bg_parameter_.mutable_guassian_parameter();

	CheckParameter(bg_parameter_);

	bg_parameter_.set_bg_status(vas::BG_UNINIALIZED);
}
BgGuassian::BgGuassian(	uint32_t num_history, uint32_t var_threshold, 
						bool use_shadow_detection, CvSize bg_size, 
						float learn_rate, uint32_t op_morphology_open_times, uint32_t op_dilate_times,
						float stable_threahold, bool use_gpu)
{
	bg_parameter_ = vas::BgParameter();

	if (use_gpu) bg_parameter_.set_bg_method(vas::GUASSIAN_GPU);
	else bg_parameter_.set_bg_method(vas::GUASSIAN_CPU);

	bg_parameter_.set_stable_threshold(stable_threahold);
	bg_parameter_.set_bg_height(bg_size.height);
	bg_parameter_.set_bg_width(bg_size.width);

	bg_parameter_.mutable_bg_operation()->set_dilate_times(op_dilate_times);
	bg_parameter_.mutable_bg_operation()->set_morphology_open_times(op_morphology_open_times);

	bg_parameter_.mutable_guassian_parameter()->set_learn_rate(learn_rate);
	bg_parameter_.mutable_guassian_parameter()->set_num_history(num_history);
	bg_parameter_.mutable_guassian_parameter()->set_shadow_detection(use_shadow_detection);
	bg_parameter_.mutable_guassian_parameter()->set_var_threshold(var_threshold);

	CheckParameter(bg_parameter_);

	bg_parameter_.set_bg_status(vas::BG_UNINIALIZED);
}

BgGuassian::BgGuassian(const char *file_name)
{
	bool success=vas::ReadProtoFromTextFile(file_name,&bg_parameter_);
	if (success)
	{
		CheckParameter(bg_parameter_);
		bg_parameter_.set_bg_status(vas::BG_UNINIALIZED);
	}

}
BgGuassian::BgGuassian(const char *ini_file_name, const char *section_name)
{
	fstream bgFile;
	bgFile.open(ini_file_name, ios::in);
	if (!bgFile)
	{
		cout << "[SmokeDetecterPro.cpp] MOG: Background Model File " << ini_file_name << " Does not exist " << endl;
		return;
	}
	uint32_t num_history = ZIni::readInt(section_name, "num_history", 10, ini_file_name);
	uint32_t var_threshold = ZIni::readInt(section_name, "var_threahold", 16, ini_file_name);
	bool shadow_detection = ZIni::readInt(section_name, "use_shadow_detection", 1, ini_file_name);
	bool use_gpu = ZIni::readInt(section_name, "use_gpu", 0, ini_file_name);
	CvSize bg_size;
	bg_size.width = ZIni::readInt(section_name, "bg_width", 800, ini_file_name);
	bg_size.height = ZIni::readInt(section_name, "bg_height", 576, ini_file_name);

	uint32_t learn_rate = ZIni::readDouble(section_name, "learn_rate", 0.001, ini_file_name);
	uint32_t morphologyEx_times = ZIni::readInt(section_name, "morphologyEx_times", 1, ini_file_name);
	uint32_t dilate_times = ZIni::readInt(section_name, "dilate_times", 1, ini_file_name);

	float stable_threahold = ZIni::readDouble(section_name, "stable_threahold", 0.5, ini_file_name);

	bg_parameter_ = vas::BgParameter();

	if (use_gpu) bg_parameter_.set_bg_method(vas::GUASSIAN_GPU);
	else bg_parameter_.set_bg_method(vas::GUASSIAN_CPU);

	bg_parameter_.set_stable_threshold(stable_threahold);
	bg_parameter_.set_bg_height(bg_size.height);
	bg_parameter_.set_bg_width(bg_size.width);

	bg_parameter_.mutable_bg_operation()->set_dilate_times(dilate_times);
	bg_parameter_.mutable_bg_operation()->set_morphology_open_times(morphologyEx_times);

	bg_parameter_.mutable_guassian_parameter()->set_learn_rate(learn_rate);
	bg_parameter_.mutable_guassian_parameter()->set_num_history(num_history);
	bg_parameter_.mutable_guassian_parameter()->set_shadow_detection(shadow_detection);
	bg_parameter_.mutable_guassian_parameter()->set_var_threshold(var_threshold);

	CheckParameter(bg_parameter_);

	bg_parameter_.set_bg_status(vas::BG_UNINIALIZED);

	bgFile.close();
}
BgGuassian::BgGuassian(const vas::BgParameter &param)
{
	bg_parameter_.CopyFrom(param);
	CheckParameter(bg_parameter_);
	bg_parameter_.set_bg_status(vas::BG_UNINIALIZED);
	
}

void BgGuassian::CheckParameter(const vas::BgParameter &param)
{
	CHECK(bg_parameter_.bg_method() == vas::GUASSIAN_CPU || bg_parameter_.bg_method() == vas::GUASSIAN_GPU) << " BgMethod is not Guassian!" << endl;
	//CHECK(bg_parameter_.has_bg_height() && bg_parameter_.has_bg_width()) << " Bg size is not set!" << endl;
	//CHECK(param.has_bg_height() && param.has_bg_width()) << " BgVibe bg size is not set!" << endl;
	CHECK_GE(param.bg_height(), 300) << "Bg bg height must be greater than 300" << endl;
	CHECK_GE(param.bg_width(), 400) << "Bg bg width must be greater than 400" << endl;
	//CHECK(bg_parameter_.has_stable_threshold()) << " Bg stable threshold is not set!" << endl;
	CHECK_GE(param.stable_threshold(), 0.2) << "BgVibe stable threshold must be greater than or equal to 0.2" << endl;
	//CHECK(bg_parameter_.has_bg_operation()) << " Bg operation is not defined!" << endl;
	//CHECK(bg_parameter_.has_guassian_parameter()) << " Bg Guassian parameter is not defined!" << endl;

}

BgGuassian::~BgGuassian()
{
	bg_parameter_.Clear();
}

void BgGuassian::Init()
{
	mog_ = BackgroundSubtractorMOG2(bg_parameter_.guassian_parameter().num_history(),
									bg_parameter_.guassian_parameter().var_threshold(), 
									bg_parameter_.guassian_parameter().shadow_detection());
	mask_ = Mat::zeros(bg_parameter_.bg_width(), bg_parameter_.bg_height(), CV_8UC1);
	bg_parameter_.set_bg_status(vas::BG_SETUP);
	frame_count_ = 0;
}
void  BgGuassian::SetBgSize(const CvSize& value)
{
	if (value.height != bg_parameter_.bg_height() || value.width != bg_parameter_.bg_width())
	{
		bg_parameter_.set_bg_width(value.width);
		bg_parameter_.set_bg_height(value.height);
		bg_parameter_.set_bg_status(vas::BG_UNINIALIZED);
	}
}

void BgGuassian::UpdateCpu(const Mat &frame)
{
	CvSize bg_size = cvSize(bg_parameter_.bg_width(), bg_parameter_.bg_height());
	if (bg_parameter_.bg_status() == vas::BG_UNINIALIZED)
	{
		Init();
	}
	frame_count_++;
	CHECK(!frame.empty()) << "Frame for update bg model is emtpy" << endl;
	Mat gray_frame;
	if (frame.channels() == 3)
	{
		cvtColor(frame, gray_frame, CV_RGB2GRAY);
	}
	else gray_frame = frame;

	Mat resize_frame;
	resize(gray_frame, resize_frame, bg_size);

	mog_(resize_frame, mask_, bg_parameter_.guassian_parameter().learn_rate());

	if (frame_count_>bg_parameter_.guassian_parameter().num_history())
	{
		if (Stable(mask_))
		{
			bg_parameter_.set_bg_status(vas::BG_UPDATING);
			MaskOperation(mask_);
		}
		else
		{
			memset(mask_.data, 0, bg_size.width*bg_size.height*sizeof(uchar));
		}
	}
	else bg_parameter_.set_bg_status(vas::BG_SETUP);
}
void BgGuassian::UpdateGpu(const Mat &frame)
{
	//not implemented
}
void BgGuassian::Update(const Mat &frame)
{
	if (bg_parameter_.bg_method()==vas::GUASSIAN_CPU)
	{
		UpdateCpu(frame);
	}
	else if (bg_parameter_.bg_method() == vas::GUASSIAN_GPU)
	{
		UpdateCpu(frame);
	}
}