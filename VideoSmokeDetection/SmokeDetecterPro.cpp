#include "stdafx.h"
#include "SmokeDetecterPro.h"


SmokeDetecterPro::SmokeDetecterPro()
{

}


SmokeDetecterPro::~SmokeDetecterPro()
{

}
SmokeDetecterPro::SmokeDetecterPro(const char*file_name, const string &set_service_id)
{
	service_id_ = set_service_id;
	InitGeneralSet(file_name);
	InitBgModel(file_name);
	InitCaffeModelInfo(file_name);
	InitSmokeFilter(file_name);
}

const string &SmokeDetecterPro::DetectorParameterString()
{
	ostringstream  para_str;

	para_str << "#GeneralSet" << endl;
	para_str << "video_reg_frame_interval=" << video_reg_frame_interval_ << endl;
	para_str << "reg_width=" << reg_size_.width << endl;
	para_str << "reg_height=" << reg_size_.height << endl;

	para_str << "#BgModel" << endl;
	if (bg_method_ == VIBE)
	{
		para_str << "bg_method=VIBE" << endl;
	}
	else if (bg_method_ == GUSSIAN)
	{
		para_str << "bg_method=GUSSIAN" << endl;
	}
	para_str << "bg_width=" << bg_size_.width << endl;
	para_str << "bg_height=" << bg_size_.height << endl;
	para_str << "morphologyEx_times=" << ptr_vibe_->MorphologyOpenTimes() << endl;
	para_str << "dilate_times=" << ptr_vibe_->DilateTimes() << endl;
	para_str << "##Vibe" << endl;
	para_str << "num_samples=" << endl;
	para_str << "min_match=" << endl;
	para_str << "radius=" << endl;
	para_str << "subsample_factor=" << endl;
	para_str << "bg_count_thredhold=" << endl;
	para_str << "move_area_ratio_thread=" << endl;
	para_str << "bg2_delay=" << endl;
	para_str << "vibe_use_gpu=" << endl;
	para_str << "##Gussian" << endl;
	para_str << "num_history=" << ptr_bgGaussian_->NumHistory() << endl;
	para_str << "var_threahold=" << ptr_bgGaussian_->VarThreahold() << endl;
	para_str << "use_shadow_detection=" << ptr_bgGaussian_->UseShadowDetection() << endl;
	return para_str.str();
}

void SmokeDetecterPro::InitGeneralSet(const char *file_name)
{
	video_reg_frame_interval_ = ZIni::readInt(service_id_.c_str(), "video_reg_frame_interval", 1, file_name);
	reg_size_.width = ZIni::readInt(service_id_.c_str(), "reg_width", 800, file_name);
	reg_size_.height = ZIni::readInt(service_id_.c_str(), "reg_height", 576, file_name);
}

void SmokeDetecterPro::InitBgModel(const char *file_name)
{
	int bg_method = ZIni::readInt(service_id_.c_str(), "bg_method", 0, file_name);
	if (bg_method == 0) bg_method_ = VIBE;
	else if (bg_method == 1) bg_method_ = GUSSIAN;

	bg_size_.width = ZIni::readInt(service_id_.c_str(), "bg_width", 0, file_name);
	bg_size_.height = ZIni::readInt(service_id_.c_str(), "bg_height", 0, file_name);

	ptr_vibe_ = shared_ptr<BgVibe>(new BgVibe(file_name, service_id_.c_str()));
	ptr_bgGaussian_ = shared_ptr<BgGaussian>(new BgGaussian(file_name, service_id_.c_str()));
}

void SmokeDetecterPro::InitSmokeFilter(const char*file_name)
{
	ptr_smoke_filter = shared_ptr<SmokeFilter>(new SmokeFilter(file_name, service_id_.c_str()));
}


void SmokeDetecterPro::InitCaffeModelInfo(const char *file_name)
{
	model_version_ = ZIni::readString(service_id_.c_str(), "model_version", "NULL", file_name);
	confience_probability_ = ZIni::readDouble(service_id_.c_str(), "confience_probability", 0.5, file_name);
	caffe_input_size_ = ZIni::readInt(service_id_.c_str(), "caffe_input_size", 224, file_name);
}
bool SmokeDetecterPro::UpdateDecter(const Mat &frame)
{

	if (!lio::CheckFrame(frame,1))
	{
		return false;
	}

	frame.copyTo(raw_frame_);
	Mat gray_frame;
	Mat color_filtered_mask;
	if (frame.channels() == 3)
	{
		cvtColor(frame, gray_frame, CV_RGB2GRAY);
	}
	else gray_frame = Mat(frame);

	if (frame.cols != reg_size_.width || frame.rows != reg_size_.height)
	{
		resize(raw_frame_, reg_frame_, reg_size_);
		reg_frame_.copyTo(show_frame_);
		resize(gray_frame, reg_gray_frame_, reg_size_);
	}
	else
	{
		raw_frame_.copyTo(reg_frame_);
		raw_frame_.copyTo(show_frame_);
		gray_frame.copyTo(reg_gray_frame_);
	}

	return true;
}

void SmokeDetecterPro::ProcessFrame(const Mat &frame, vector<CvRect> &rects, vector<float> &ratios)
{

	UpdateDecter(frame);

	bool bg_ready = GetForeMask();

	rects.clear();
	ratios.clear();
	Mat filtered_mask;
	if (bg_ready)
	{
		
		vector<vector<Point>> contours;
		vector<CvRect> t_rects;
		
		ptr_smoke_filter->Filtrate(reg_frame_, fore_mask_, filtered_mask, contours, t_rects);

		lio::MergeRects(t_rects, filtered_mask.size().width, filtered_mask.size().height);

		for (size_t i = 0; i < t_rects.size(); i++)
		{
			CvRect reg_frame_rect,reg_frame_square,resize_reg_frame_square;
			lio::MapRect(t_rects[i], reg_frame_rect, filtered_mask.size(), reg_frame_.size());
			lio::RectToSquare(reg_frame_rect, reg_frame_square, reg_frame_.size());
			if (reg_frame_square.width<224)
			{
				lio::ScaleRect(reg_frame_square, resize_reg_frame_square, reg_frame_.size(), cvSize(224, 224));
			}
			else resize_reg_frame_square = reg_frame_square;
			Mat reg_mat = reg_frame_(resize_reg_frame_square);

			vector<Prediction> predictions;
			predictions=caffe_reg_fun(reg_mat,2);
			const Prediction &predict = predictions[0];
			if (predict.first.compare("smoke") == 0 && predict.second>0.3)
			{
				rects.push_back(t_rects[i]);
				ratios.push_back(predict.second);
				rectangle(reg_frame_, resize_reg_frame_square, CV_RGB(255, 0, 0), 1);
			}

		}
	}
	
	if (!filtered_mask.empty())
	{
		imshow(service_id_ + "color_flitered_mask", filtered_mask);
	}
	if (!fore_mask_.empty())
	{

		imshow(service_id_ + "fore_mask", fore_mask_);
	}
	if (!fore_mask_.empty())
	{
		Mat resize_reg_frame;
		resize(reg_frame_, resize_reg_frame, fore_mask_.size());
		imshow("frame", resize_reg_frame);

	}
	
}



bool SmokeDetecterPro::GetForeMask()
{
	if (bg_method_ == VIBE)
	{
		ptr_vibe_->Update(reg_gray_frame_);
	}
	else if (bg_method_ == GUSSIAN)
	{
		ptr_bgGaussian_->Update(reg_gray_frame_);
	}

	if ((bg_method_ == VIBE&&ptr_vibe_->Status() == UPDATING))
	{
		fore_mask_ = ptr_vibe_->Mask();
		return true;
	}
	else if (bg_method_ == GUSSIAN&&!ptr_bgGaussian_->Mask().empty())
	{
		fore_mask_ = ptr_bgGaussian_->Mask();
		return true;
	}
	else return false;
}

void SmokeDetecterPro::SetCaffeRegFunction(std::function<vector<Prediction>(const Mat&, const int&)> set_fun)
{
	caffe_reg_fun = set_fun;
}


