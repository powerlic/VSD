#include "stdafx.h"
#include "VasDetectMaster.h"


/*-------------------------------Detecter-------------------------------------*/


Detecter::Detecter()
{
	detect_param_ = vas::DetectParameter();
	detect_param_.mutable_bg_parameter();
	detect_param_.mutable_filter_parameter();
	detect_param_.mutable_caffe_parameter();

	CheckParameter(detect_param_);

}

Detecter::Detecter(const char *file_name)
{
	bool success = vas::ReadProtoFromTextFile(file_name, &detect_param_);
	if (success)
	{
		service_id_ = detect_param_.service_id();
		CheckParameter(detect_param_);
	}
	
}

Detecter::Detecter(const vas::DetectParameter &param)
{
	detect_param_.CopyFrom(param);
	service_id_ = param.service_id();

	CheckParameter(detect_param_);
}

Detecter::~Detecter()
{
	raw_frame_.release();
	reg_frame_.release();
	show_frame_.release();


	list<Mat>::iterator itor;
	itor = smoke_hot_frame_diff_list_.begin();
	while (itor != smoke_hot_frame_diff_list_.end())
	{
		(*itor).release();
		itor++;
	}
	smoke_hot_frame_diff_list_.clear();
	smoke_hot_frame_.release();


	list<Mat>::iterator fire_itor;
	fire_itor = fire_hot_frame_diff_list_.begin();
	while (fire_itor != fire_hot_frame_diff_list_.end())
	{
		(*fire_itor).release();
		fire_itor++;
	}
	fire_hot_frame_diff_list_.clear();
	fire_hot_frame_.release();

}

void Detecter::CheckParameter(const vas::DetectParameter &param)
{
	CHECK(param.has_service_id())<<" No service id "<<endl;
	CHECK_GE(param.reg_height(), 576) << " Reg height must be greater or equal to than 576!" << endl;
	CHECK_GE(param.reg_width(), 800) << " Reg height must be greater than or equal to 800!" << endl;
	CHECK_GE(param.reg_interval(), 20) << "Reg time interval must be greater than or equal to 20 " << endl;
	CHECK(param.has_caffe_parameter()) << " Detecter must set caffe model" << endl;
	CHECK_GE(param.num_history(),20) << " Num of history must be greater than or equal to 20" << endl;
	CHECK_GE(param.smoke_detect_sensitity(), 0) << "Smoke detect sensitity must be greater than or equal to 0 " << endl;
	CHECK_LE(param.smoke_detect_sensitity(), 1) << "Smoke detect sensitity must be less than or equal to 1 " << endl;
	CHECK_GE(param.fire_detect_sensitity(), 0) << "Fire detect sensitity must be greater than or equal to 0 " << endl;
	CHECK_LE(param.fire_detect_sensitity(), 1) << "Fire detect sensitity must be less than or equal to 1 " << endl;
}






/*-------------------------------DetectMaster-------------------------------------*/
VasDetectMaster::VasDetectMaster()
{

}
VasDetectMaster::VasDetectMaster(std::function<vector<Prediction>(const Mat&, const int&)> caffe_reg_fun,
	                             std::function<bool(const char *service_id, Mat &frame, int64_t &pts, int64_t &No)> get_frame_fun)
{
	caffe_reg_fun_ = caffe_reg_fun;
	get_frame_fun_ = get_frame_fun; 
}

VasDetectMaster::~VasDetectMaster()
{

}

void VasDetectMaster::CreateDetecterContext(shared_ptr<Detecter> ptr_detecter)
{
	CHECK(ptr_detecter->detect_param_.IsInitialized()) << ptr_detecter->service_id_ << " detect_parameter is not inited!" << endl;


	if (ptr_detecter->detect_param_.save_video())
	{
		SetupVideoSaveFile(ptr_detecter);
		//ptr_detecter->video_writer_ = VideoWriter("VideoTest.avi", CV_FOURCC('M', 'J', 'P', 'G'), 25.0, Size(ptr_detecter->detect_param_.reg_width(), ptr_detecter->detect_param_.reg_height()));
	}

	if (ptr_detecter->detect_param_.bg_parameter().bg_method() == vas::GUASSIAN_CPU || ptr_detecter->detect_param_.bg_parameter().bg_method() == vas::GUASSIAN_GPU)
	{
		ptr_detecter->ptr_bg_ = shared_ptr<BgGuassian>(new BgGuassian(ptr_detecter->detect_param_.bg_parameter()));
	}
	else if (ptr_detecter->detect_param_.bg_parameter().bg_method() == vas::VIBE_CPU || ptr_detecter->detect_param_.bg_parameter().bg_method() == vas::VIBE_GPU)
	{
		//ptr_detecter->ptr_bg_ = shared_ptr<BgVibe>(new BgVibe(ptr_detecter->detect_param_.bg_parameter()));
		ptr_detecter->ptr_bg_ = shared_ptr<BgVibeBoost>(new BgVibeBoost(ptr_detecter->detect_param_.bg_parameter()));
	}

	ptr_detecter->ptr_filter_ = shared_ptr<Filter>(new Filter(ptr_detecter->detect_param_.filter_parameter()));

	ptr_detecter->detect_param_.set_detect_status(vas::DETECT_ON);

	CHECK(caffe_reg_fun_) << "caffe reg fun is not set!" << endl;
	CHECK(get_frame_fun_) << "get frame fun is not set!" << endl;
	ptr_detecter->smoke_detect_include_ = false;
	ptr_detecter->fire_detect_include_ = false;
	for (size_t i = 0; i < ptr_detecter->detect_param_.reg_type().size(); i++)
	{
		if (ptr_detecter->detect_param_.reg_type(i).compare("smoke")==0)
		{
			ptr_detecter->smoke_detect_include_ = true;
			ptr_detecter->smoke_threshold_ = 255.0*(1-ptr_detecter->detect_param_.smoke_detect_sensitity());
		}
		else if (ptr_detecter->detect_param_.reg_type(i).compare("fire") == 0)
		{
			ptr_detecter->fire_detect_include_ = true;
			ptr_detecter->fire_threshold_ = 255.0*(1 - ptr_detecter->detect_param_.fire_detect_sensitity());
		}
	}
	

	if (ptr_detecter->detect_param_.show_result_frame())
	{
		namedWindow(ptr_detecter->service_id_, 1);
	}
}
void VasDetectMaster::SetupVideoSaveFile(shared_ptr<Detecter> ptr_detecter)
{
	string name_str = lio::GetDateTimeString();
	string file_path;
	file_path = "./regvideo";
	fstream reg_file;
	reg_file.open(file_path, ios::in);
	if (!reg_file)
	{
		_mkdir(file_path.c_str());
	}
	file_path = "./regvideo/" + lio::GetDateString();
	fstream _file;
	_file.open(file_path, ios::in);
	if (!_file)
	{
		_mkdir(file_path.c_str());
	}
	ptr_detecter->video_name_ = "./regvideo/" + lio::GetDateString() + "/" + ptr_detecter->service_id_ + "_" + lio::GetDateTimeString() + "_reg.avi";
	ptr_detecter->video_writer_ = VideoWriter(ptr_detecter->video_name_, CV_FOURCC('F', 'L', 'V', '1'), 25.0, Size(ptr_detecter->detect_param_.reg_width(), ptr_detecter->detect_param_.reg_height()));
}
void VasDetectMaster::GetMoveRectFrame(const Mat &frame, const vector<CvRect> &rects, Mat &out_frame)
{
	out_frame = Mat::zeros(frame.size(), CV_8UC3);
	for (size_t i = 0; i < rects.size(); i++)
	{
		Mat t_mat = frame(rects[i]);
		Mat t_dst_mat = out_frame(rects[i]);
		t_mat.copyTo(t_dst_mat);
	}

}
void VasDetectMaster::DestroyDetecterContext(shared_ptr<Detecter> ptr_detecter)
{
	ptr_detecter->raw_frame_.release();
	ptr_detecter->reg_frame_.release();
	ptr_detecter->show_frame_.release();

	list<Mat>::iterator itor;
	itor = ptr_detecter->smoke_hot_frame_diff_list_.begin();
	while (itor != ptr_detecter->smoke_hot_frame_diff_list_.end())
	{
		(*itor).release();
		itor++;
	}
	ptr_detecter->smoke_hot_frame_diff_list_.clear();
	ptr_detecter->smoke_hot_frame_.release();


	list<Mat>::iterator fire_itor;
	fire_itor = ptr_detecter->fire_hot_frame_diff_list_.begin();
	while (fire_itor != ptr_detecter->fire_hot_frame_diff_list_.end())
	{
		(*fire_itor).release();
		fire_itor++;
	}
	ptr_detecter->fire_hot_frame_diff_list_.clear();
	ptr_detecter->fire_hot_frame_.release();



	ptr_detecter->ptr_bg_.reset();
	ptr_detecter->ptr_filter_.reset();

	ptr_detecter->detect_param_.set_detect_status(vas::DETECT_OFF);

	if (ptr_detecter->detect_param_.save_video())
	{
		ptr_detecter->video_writer_.release();
	}

	if (ptr_detecter->detect_param_.show_result_frame())
	{
		destroyWindow(ptr_detecter->service_id_);
	}
}

bool VasDetectMaster::UpdateFrame(shared_ptr<Detecter> ptr_detecter, const Mat &frame, const int64_t &pts, const int64_t &No)
{
	if (pts<0 || !frame.data)
	{
		return false;
	}
	ptr_detecter->pts_ = pts;
	ptr_detecter->No_ = No;
	frame.copyTo(ptr_detecter->raw_frame_);

	CvSize reg_size = cvSize(ptr_detecter->detect_param_.reg_width(), ptr_detecter->detect_param_.reg_height());
	
	if (ptr_detecter->raw_frame_.cols != reg_size.width || ptr_detecter->raw_frame_.rows != reg_size.height)
	{
		resize(ptr_detecter->raw_frame_, ptr_detecter->reg_frame_, reg_size);
		ptr_detecter->reg_frame_.copyTo(ptr_detecter->show_frame_);
	}
	else
	{
		ptr_detecter->raw_frame_.copyTo(ptr_detecter->reg_frame_);
		ptr_detecter->reg_frame_.copyTo(ptr_detecter->show_frame_);
	}
}

shared_ptr<Detecter> VasDetectMaster::GetDetecterPtr(const char *service_id)
{
	boost::mutex::scoped_lock lock(mutex_);
	vector<shared_ptr<Detecter>>::iterator iter;
	for (iter = detecter_list_.begin(); iter != detecter_list_.end(); iter++)
	{
		string t_service_id = (*iter)->service_id_;
		if (t_service_id.compare(String(service_id)) == 0)
		{
			lock.unlock();
			return *iter;
		}
	}
	lock.unlock();
	return NULL;
}
shared_ptr<Detecter> VasDetectMaster::GetDetecterPtr(const string &service_id)
{
	boost::mutex::scoped_lock lock(mutex_);
	vector<shared_ptr<Detecter>>::iterator iter;
	for (iter = detecter_list_.begin(); iter != detecter_list_.end(); iter++)
	{
		string t_service_id = (*iter)->service_id_;
		if (t_service_id.compare(String(service_id)) == 0)
		{
			lock.unlock();
			return *iter;
		}
	}
	lock.unlock();
	return NULL;
}

bool VasDetectMaster::GetForeMask(shared_ptr<Detecter> ptr_detecter, Mat &mask)
{
	ptr_detecter->ptr_bg_->Update(ptr_detecter->reg_frame_);
	if (ptr_detecter->ptr_bg_->BgStatus() == vas::BG_UPDATING)
	{
		mask = ptr_detecter->ptr_bg_->Mask();
		return true;
	}
	else return false;
}
void VasDetectMaster::DetectFrameMerged(shared_ptr<Detecter> ptr_detecter, const Mat &frame, const int64_t &pts, const int64_t &No)
{
	UpdateFrame(ptr_detecter, frame, pts, No);


	if (!ptr_detecter->is_send_start_success_msg_)
	{
		ptr_detecter->is_send_start_success_msg_ = true;
		service_command_reply_signal_(ptr_detecter->service_id_.c_str(), vas::SERVICE_START_SUCCESS, "Service start success!");
	}


	Mat fore_mask, filtered_mask;
	bool bg_ready = GetForeMask(ptr_detecter, fore_mask);



	Mat smoke_suspect_frame,fire_suspect_frame;

	uint32_t caffe_input_size = ptr_detecter->detect_param_.caffe_parameter().input_size();
	CvSize caffe_size = cvSize(caffe_input_size, caffe_input_size);
	CvSize reg_size = cvSize(ptr_detecter->detect_param_.reg_width(), ptr_detecter->detect_param_.reg_height());


	vector<CvRect> utified_reg_rects;
	vector<float> utified_reg_scores;
	vector<string> utifiled_reg_labels;

	if (ptr_detecter->smoke_detect_include_)
	{
		smoke_suspect_frame = Mat::zeros(reg_size, CV_8UC1);
	}
	if (ptr_detecter->fire_detect_include_)
	{
		fire_suspect_frame = Mat::zeros(reg_size, CV_8UC1);
	}

	float caffe_threshold = ptr_detecter->detect_param_.caffe_parameter().confience_score();

	map<string, vector<CvRect>> label_reg_rects_map;
	map<string, vector<float>> label_score_map;
	map<string, vector<CvRect>> label_raw_rects_map;
	float max_smoke_score = 0;
	if (bg_ready&&!frame.empty())
	{
		vector<vector<Point>> t_contours;
		vector<CvRect> t_rects;
		ptr_detecter->ptr_filter_->Filtrate(ptr_detecter->reg_frame_, fore_mask, filtered_mask, t_contours, t_rects);

		if (ptr_detecter->detect_param_.show_bg())
		{
			imshow(ptr_detecter->service_id_ + "_filtered_mask", filtered_mask);
		}

		vector<CvRect> merged_rect_list;
		vector<set<int>> merged_rects_number_set_list;

		lio::MergeRects(t_rects, merged_rect_list, merged_rects_number_set_list, ptr_detecter->reg_frame_.size().width, ptr_detecter->reg_frame_.size().height);

		for (size_t i = 0; i < merged_rect_list.size(); i++)
		{
			CvRect reg_frame_square, resize_reg_frame_square;
			lio::CorrectRect(merged_rect_list[i], ptr_detecter->reg_frame_.size().width, ptr_detecter->reg_frame_.size().height);
			lio::RectToSquare(merged_rect_list[i], reg_frame_square, ptr_detecter->reg_frame_.size());
			if (reg_frame_square.width < caffe_input_size)
			{
				lio::ScaleRect(reg_frame_square, resize_reg_frame_square, reg_size, caffe_size);
			}
			else
			{
				resize_reg_frame_square = reg_frame_square;
			}
			Mat reg_mat = ptr_detecter->reg_frame_(resize_reg_frame_square);

			vector<Prediction> predictions;
			predictions = caffe_reg_fun_(reg_mat, 2);
			const Prediction &predict = predictions[0];

			if (ptr_detecter->detect_param_.save_reg_frame())
			{
				lio::SaveRegResult(ptr_detecter->service_id_.c_str(), reg_mat, predict.first);
			}

			if (predict.first.compare("smoke") == 0 && predict.second > caffe_threshold&&ptr_detecter->smoke_detect_include_)
			{
				for (set<int> ::iterator iter = merged_rects_number_set_list[i].begin(); iter != merged_rects_number_set_list[i].end(); iter++)
				{
					drawContours(smoke_suspect_frame, t_contours, *iter, Scalar(5), CV_FILLED);
				}
				if (predict.second >max_smoke_score)
				{
					max_smoke_score = predict.second;
				}
			}
			else if (predict.first.compare("fire") == 0 && predict.second > caffe_threshold&&ptr_detecter->fire_detect_include_)
			{
				drawContours(fire_suspect_frame, t_contours, i, Scalar(5), CV_FILLED);
				if (predict.second >max_smoke_score)
				{
					max_smoke_score = predict.second;
				}
			}
			else if (CheckRegType(predict.first, ptr_detecter->detect_param_) && predict.second > caffe_threshold)
			{

				label_reg_rects_map[predict.first].push_back(resize_reg_frame_square);
				label_score_map[predict.first].push_back(predict.second);
				label_raw_rects_map[predict.first].push_back(merged_rect_list[i]);
			}
		}
	}

	//Other Label Reg Result
	map<string, vector<CvRect>>::iterator   it;
	map<string, vector<float>>::iterator   score_it = label_score_map.begin();
	for (it = label_raw_rects_map.begin(); it != label_raw_rects_map.end(); it++)
	{
		const string &label = it->first;
		const vector<CvRect> &rects = it->second;
		const vector<float> &socres = score_it->second;
		vector<CvRect> d_rects;
		lio::NonMaximumSuppress2(rects, socres, d_rects, 0.01);
		for (size_t i = 0; i < d_rects.size(); i++)
		{
			//lio::WriteText(ptr_detecter->show_frame_, cvPoint(d_rects[i].x, d_rects[i].y), label, 255, 0, 0);
			lio::WtrieLabel(ptr_detecter->show_frame_, d_rects[i], label, 0, 255, 0);
			rectangle(ptr_detecter->show_frame_, d_rects[i], CV_RGB(0, 255, 0), 1);

			utified_reg_rects.push_back(d_rects[i]);
			utifiled_reg_labels.push_back(label);
			utified_reg_scores.push_back(socres[0]);
		}
		score_it++;
	}

	//Smoke Reg Result
	if (ptr_detecter->smoke_detect_include_)
	{
		vector<CvRect> smoke_rects;
		vector<vector<Point>> smoke_contours;
		Mat smoke_threshold_frame;
		UpdateHotFrame(ptr_detecter, smoke_suspect_frame, ptr_detecter->smoke_threshold_, ptr_detecter->smoke_hot_frame_diff_list_, ptr_detecter->smoke_hot_frame_, smoke_threshold_frame);
		if (!smoke_threshold_frame.empty())
		{
			lio::FindContoursRects(smoke_threshold_frame, 500, smoke_rects, smoke_contours);
			MergeSmokeRects(smoke_rects, reg_size.width, reg_size.height, caffe_input_size);
			for (size_t i = 0; i < smoke_rects.size(); i++)
			{
				utified_reg_rects.push_back(smoke_rects[i]);
				utifiled_reg_labels.push_back("smoke");
				utified_reg_scores.push_back(max_smoke_score);
				lio::WtrieLabel(ptr_detecter->show_frame_, smoke_rects[i], "smoke", 255, 0, 0);
				rectangle(ptr_detecter->show_frame_, smoke_rects[i], CV_RGB(255, 0, 0), 1);
			}
		}
		if (ptr_detecter->detect_param_.show_smoke_hot_frame() && !ptr_detecter->smoke_hot_frame_.empty())
		{
			imshow(ptr_detecter->service_id_ + "_smoke_hot", ptr_detecter->smoke_hot_frame_);
		}
	}
	//fire Reg Result
	if (ptr_detecter->fire_detect_include_)
	{
		vector<CvRect> fire_rects;
		vector<vector<Point>> fire_contours;
		Mat fire_threshold_frame;
		UpdateHotFrame(ptr_detecter, fire_suspect_frame, ptr_detecter->fire_threshold_, ptr_detecter->fire_hot_frame_diff_list_, ptr_detecter->fire_hot_frame_, fire_threshold_frame);
		if (!fire_threshold_frame.empty())
		{
			lio::FindContoursRects(fire_threshold_frame, 500, fire_rects, fire_contours);
			MergeSmokeRects(fire_rects, reg_size.width, reg_size.height, caffe_input_size);
			for (size_t i = 0; i < fire_rects.size(); i++)
			{
				utified_reg_rects.push_back(fire_rects[i]);
				utifiled_reg_labels.push_back("fire");
				utified_reg_scores.push_back(max_smoke_score);
				lio::WtrieLabel(ptr_detecter->show_frame_, fire_rects[i], "fire", 0, 0, 255);
				rectangle(ptr_detecter->show_frame_, fire_rects[i], CV_RGB(0, 0, 255), 1);
			}
		}
		if (ptr_detecter->detect_param_.show_fire_hot_frame() && !ptr_detecter->fire_hot_frame_.empty())
		{
			imshow(ptr_detecter->service_id_ + "_fire_hot", ptr_detecter->fire_hot_frame_);
		}
	}


	if (utified_reg_rects.size()>0)
	{
		reg_signal_(ptr_detecter->service_id_.c_str(), utified_reg_rects, reg_size.width, reg_size.height, utified_reg_scores, utifiled_reg_labels);

		if (ptr_detecter->detect_param_.save_reg_frame())
		{
			//lio::SaveRegResult(ptr_detecter->service_id_.c_str(), reg_mat, predict.first);
			lio::SaveRegFrame(ptr_detecter->service_id_.c_str(), ptr_detecter->show_frame_);
		}
	}
}

void VasDetectMaster::DetectFrame(shared_ptr<Detecter> ptr_detecter, const Mat &frame, const int64_t &pts, const int64_t &No)
{
	UpdateFrame(ptr_detecter, frame, pts, No);


	if (!ptr_detecter->is_send_start_success_msg_)
	{
		ptr_detecter->is_send_start_success_msg_ = true;
		service_command_reply_signal_(ptr_detecter->service_id_.c_str(), vas::SERVICE_START_SUCCESS, "Service start success!");
	}


	Mat fore_mask, filtered_mask;
	bool bg_ready = GetForeMask(ptr_detecter, fore_mask);

	

	Mat smoke_suspect_frame,fire_suspect_frame;

	uint32_t caffe_input_size = ptr_detecter->detect_param_.caffe_parameter().input_size();
	CvSize caffe_size = cvSize(caffe_input_size, caffe_input_size);
	CvSize reg_size = cvSize(ptr_detecter->detect_param_.reg_width(), ptr_detecter->detect_param_.reg_height());


	vector<CvRect> utified_reg_rects;
	vector<float> utified_reg_scores;
	vector<string> utifiled_reg_labels;

	if (ptr_detecter->smoke_detect_include_)
	{
		smoke_suspect_frame = Mat::zeros(reg_size, CV_8UC1);
	}
	if (ptr_detecter->fire_detect_include_)
	{
		fire_suspect_frame = Mat::zeros(reg_size, CV_8UC1);
	}

	float caffe_threshold = ptr_detecter->detect_param_.caffe_parameter().confience_score();

	map<string, vector<CvRect>> label_reg_rects_map;
	map<string, vector<float>> label_score_map;
	map<string, vector<CvRect>> label_raw_rects_map;
	float max_smoke_score = 0;
	if (bg_ready&&!frame.empty())
	{
		vector<vector<Point>> t_contours;
		vector<CvRect> t_rects;
		ptr_detecter->ptr_filter_->Filtrate(ptr_detecter->reg_frame_, fore_mask, filtered_mask, t_contours, t_rects);

		Mat only_move_frame;
		if (ptr_detecter->detect_param_.move_area_for_reg_only())
		{
			GetMoveRectFrame(ptr_detecter->reg_frame_,  t_rects, only_move_frame);
		}
		if (ptr_detecter->detect_param_.show_bg())
		{
			imshow(ptr_detecter->service_id_ + "_filtered_mask", filtered_mask);
		}
		if (ptr_detecter->detect_param_.show_move_area_for_reg_only() && ptr_detecter->detect_param_.move_area_for_reg_only())
		{
			imshow(ptr_detecter->service_id_ + "_move_only_frame", only_move_frame);
		}

		for (size_t i = 0; i < t_rects.size(); i++)
		{
			CvRect reg_frame_square, resize_reg_frame_square;
			lio::CorrectRect(t_rects[i], ptr_detecter->reg_frame_.size().width, ptr_detecter->reg_frame_.size().height);
			lio::RectToSquare(t_rects[i], reg_frame_square, ptr_detecter->reg_frame_.size());
			if (reg_frame_square.width < caffe_input_size)
			{
				lio::ScaleRect(reg_frame_square, resize_reg_frame_square, reg_size, caffe_size);
			}
			else
			{
				resize_reg_frame_square = reg_frame_square;
			}
			Mat reg_mat;
			if (ptr_detecter->detect_param_.move_area_for_reg_only())
			{
				reg_mat = only_move_frame(resize_reg_frame_square);
			}
			else reg_mat = ptr_detecter->reg_frame_(resize_reg_frame_square);

			vector<Prediction> predictions;
			predictions = caffe_reg_fun_(reg_mat, 2);
			const Prediction &predict = predictions[0];

			
			if (ptr_detecter->detect_param_.save_reg_frame())
			{
				lio::SaveRegResult(ptr_detecter->service_id_.c_str(), reg_mat, predict.first);
			}

			if (predict.first.compare("smoke") == 0 && predict.second > caffe_threshold&&ptr_detecter->smoke_detect_include_)
			{
				drawContours(smoke_suspect_frame, t_contours, i, Scalar(5), CV_FILLED);
				if (predict.second >max_smoke_score)
				{
					max_smoke_score = predict.second;
				}
			}
			else if (predict.first.compare("fire") == 0 && predict.second > caffe_threshold&&ptr_detecter->fire_detect_include_)
			{
				drawContours(fire_suspect_frame, t_contours, i, Scalar(5), CV_FILLED);
				if (predict.second >max_smoke_score)
				{
					max_smoke_score = predict.second;
				}
			}
			else if (CheckRegType(predict.first, ptr_detecter->detect_param_) && predict.second > caffe_threshold)
			{

				label_reg_rects_map[predict.first].push_back(resize_reg_frame_square);
				label_score_map[predict.first].push_back(predict.second);
				label_raw_rects_map[predict.first].push_back(t_rects[i]);
			}
		}
	}

	//Other Label Reg Result
	map<string, vector<CvRect>>::iterator   it;
	map<string, vector<float>>::iterator   score_it = label_score_map.begin();
	for (it = label_raw_rects_map.begin(); it != label_raw_rects_map.end(); it++)
	{
		const string &label = it->first;
		const vector<CvRect> &rects = it->second;
		const vector<float> &socres = score_it->second;
		vector<CvRect> d_rects;
		lio::NonMaximumSuppress2(rects, socres, d_rects, 0.01);
		for (size_t i = 0; i < d_rects.size(); i++)
		{
			lio::WtrieLabel(ptr_detecter->show_frame_, d_rects[i], label, 0, 255, 0);
			rectangle(ptr_detecter->show_frame_, d_rects[i], CV_RGB(0, 255, 0), 1);

			utified_reg_rects.push_back(d_rects[i]);
			utifiled_reg_labels.push_back(label);
			utified_reg_scores.push_back(socres[0]);
		}
		score_it++;
	}


	//Smoke Reg Result
	if (ptr_detecter->smoke_detect_include_)
	{
		vector<CvRect> smoke_rects;
		vector<vector<Point>> smoke_contours;
		Mat smoke_threshold_frame;
		UpdateHotFrame(ptr_detecter, smoke_suspect_frame, ptr_detecter->smoke_threshold_, ptr_detecter->smoke_hot_frame_diff_list_, ptr_detecter->smoke_hot_frame_, smoke_threshold_frame);
		if (!smoke_threshold_frame.empty())
		{
			lio::FindContoursRects(smoke_threshold_frame, 500, smoke_rects, smoke_contours);
			MergeSmokeRects(smoke_rects, reg_size.width, reg_size.height, caffe_input_size);
			for (size_t i = 0; i < smoke_rects.size(); i++)
			{
				utified_reg_rects.push_back(smoke_rects[i]);
				utifiled_reg_labels.push_back("smoke");
				utified_reg_scores.push_back(max_smoke_score);
				lio::WtrieLabel(ptr_detecter->show_frame_, smoke_rects[i], "smoke", 255, 0, 0);
				rectangle(ptr_detecter->show_frame_, smoke_rects[i], CV_RGB(255, 0, 0), 1);
			}
		}
		if (ptr_detecter->detect_param_.show_smoke_hot_frame() && !ptr_detecter->smoke_hot_frame_.empty())
		{
			imshow(ptr_detecter->service_id_ + "_smoke_hot", ptr_detecter->smoke_hot_frame_);
		}
	}
	//fire Reg Result
	if (ptr_detecter->fire_detect_include_)
	{
		vector<CvRect> fire_rects;
		vector<vector<Point>> fire_contours;
		Mat fire_threshold_frame;
		UpdateHotFrame(ptr_detecter, fire_suspect_frame, ptr_detecter->fire_threshold_, ptr_detecter->fire_hot_frame_diff_list_, ptr_detecter->fire_hot_frame_, fire_threshold_frame);
		if (!fire_threshold_frame.empty())
		{
			lio::FindContoursRects(fire_threshold_frame, 500, fire_rects, fire_contours);
			MergeSmokeRects(fire_rects, reg_size.width, reg_size.height, caffe_input_size);
			for (size_t i = 0; i < fire_rects.size(); i++)
			{
				utified_reg_rects.push_back(fire_rects[i]);
				utifiled_reg_labels.push_back("fire");
				utified_reg_scores.push_back(max_smoke_score);
				lio::WtrieLabel(ptr_detecter->show_frame_, fire_rects[i], "fire", 0, 0, 255);
				rectangle(ptr_detecter->show_frame_, fire_rects[i], CV_RGB(0, 0, 255), 1);
			}
		}
		if (ptr_detecter->detect_param_.show_fire_hot_frame() && !ptr_detecter->fire_hot_frame_.empty())
		{
			imshow(ptr_detecter->service_id_ + "_fire_hot", ptr_detecter->fire_hot_frame_);
		}

	}

	if (utified_reg_rects.size()>0)
	{
		reg_signal_(ptr_detecter->service_id_.c_str(), utified_reg_rects, reg_size.width, reg_size.height, utified_reg_scores, utifiled_reg_labels);

		if (ptr_detecter->detect_param_.save_reg_frame())
		{
			//lio::SaveRegResult(ptr_detecter->service_id_.c_str(), reg_mat, predict.first);
			lio::SaveRegFrame(ptr_detecter->service_id_.c_str(), ptr_detecter->show_frame_);
		}
	}

	


}
void VasDetectMaster::UpdateHotFrame(shared_ptr<Detecter> ptr_detecter, const Mat &suspected_frame, const uint32_t detect_threshold, list<Mat> &hot_frame_diff_list_, Mat &hot_frame, Mat &hot_threshold_frame)
{
	if (suspected_frame.empty())
	{
		return;
	}
	CvSize reg_size = cvSize(ptr_detecter->detect_param_.reg_width(), ptr_detecter->detect_param_.reg_height());
	if (hot_frame.empty())
	{
		hot_frame = Mat::zeros(reg_size, CV_8UC1);
	}
	if (hot_frame_diff_list_.size()<ptr_detecter->detect_param_.num_history())
	{
		hot_frame_diff_list_.push_back(suspected_frame);
		add(hot_frame, suspected_frame, hot_frame);
	}
	else
	{
		Mat first_frame = hot_frame_diff_list_.front();
		hot_frame_diff_list_.pop_front();
		hot_frame_diff_list_.push_back(suspected_frame);
		subtract(hot_frame, first_frame, hot_frame);
		add(hot_frame, suspected_frame, hot_frame);
		threshold(hot_frame, hot_threshold_frame, (double)detect_threshold, 255, CV_THRESH_BINARY);
	}

}

void VasDetectMaster::VideoDetect(shared_ptr<Detecter> ptr_detecter)
{
	Mat frame;
	int64_t pts;
	int64_t No;
	
	while (true)
	{
		if (ptr_detecter->detect_param_.detect_status()==vas::DETECT_OFF)
		{
			break;
		}
		CreateDetecterContext(ptr_detecter);
		LOG(INFO) << ptr_detecter->service_id_ << " create detect process success!" << endl;
		while (ptr_detecter->detect_param_.detect_status() != vas::DETECT_OFF)
		{
			if (ptr_detecter->detect_param_.detect_status()==vas::DETECT_PAUSE)
			{
				lio::BoostSleep(500);
				continue;
			}
			bool got_frame=get_frame_fun_(ptr_detecter->service_id_.c_str(), frame, pts, No);
			if (got_frame)
			{
				if (ptr_detecter->detect_param_.merge_fore_rects())
				{
					DetectFrameMerged(ptr_detecter, frame, pts, No);
				}
				else
				{
					DetectFrame(ptr_detecter, frame, pts, No);
				}
				if (ptr_detecter->detect_param_.save_video())
				{
					if (ptr_detecter->video_writer_.isOpened())
					{
						ptr_detecter->video_writer_ << ptr_detecter->show_frame_;
					}
				}
				if (ptr_detecter->detect_param_.show_result_frame())
				{
					imshow(ptr_detecter->service_id_, ptr_detecter->show_frame_);
					waitKey(1);
				}
				
			}
			
			lio::BoostSleep(ptr_detecter->detect_param_.reg_interval());

			
		}
		DestroyDetecterContext(ptr_detecter);
		lio::BoostSleep(1000);
	}
	
}
bool VasDetectMaster::GetRegResult(const char* service_id, Mat &show_frame, int64_t &pts)
{
	shared_ptr<Detecter> ptr_detecter = GetDetecterPtr(service_id);
	if (ptr_detecter)
	{
		ptr_detecter->show_frame_.copyTo(show_frame);
		pts = ptr_detecter->pts_;
		return true;
	}
	else return false;
}
bool VasDetectMaster::CheckIdUnique(const char* service_id)
{
	vector<shared_ptr<Detecter>>::iterator iter;
	boost::mutex::scoped_lock lock(mutex_);
	for (iter = detecter_list_.begin(); iter != detecter_list_.end(); iter++)
	{
		string t_service_id = (*iter)->service_id_;
		if (t_service_id.compare(String(service_id)) == 0)
		{
			LOG(INFO) << " Already has service named as " << service_id << endl;
			lock.unlock();
			return true;
		}
	}
	lock.unlock();
	return false;
}
bool VasDetectMaster::CheckIdUnique(const string &service_id)
{
	vector<shared_ptr<Detecter>>::iterator iter;
	for (iter = detecter_list_.begin(); iter != detecter_list_.end(); iter++)
	{
		string t_service_id = (*iter)->service_id_;
		if (t_service_id.compare(String(service_id)) == 0)
		{
			LOG(INFO) << " Already has service named as " << service_id << endl;
			return false;
		}
	}
	return true;
}

bool VasDetectMaster::AddDetecter(const char *proto_file_name)
{
	boost::mutex::scoped_lock lock(mutex_);
	vas::DetectParameter param;
	CHECK(vas::ReadProtoFromTextFile(proto_file_name, &param)) << "read " << proto_file_name << " failed " << endl;
	
	if (CheckIdUnique(param.service_id()))
	{
		lock.unlock();
		return false;
	}
	
	shared_ptr<Detecter> ptr_detecter = shared_ptr<Detecter>(new Detecter(proto_file_name));
	detecter_list_.push_back(ptr_detecter);
	detect_thread_list_.push_back(boost::thread(&VasDetectMaster::VideoDetect, this, detecter_list_.back()));
	lio::BoostSleep(500);
	lock.unlock();
	return true;
}

bool VasDetectMaster::AddDetecter(const string &service_id)
{
	if (CheckIdUnique(service_id))
	{
		return false;
	}
	boost::mutex::scoped_lock lock(mutex_);
	shared_ptr<Detecter> ptr_detecter = shared_ptr<Detecter>(new Detecter());
	ptr_detecter->detect_param_.set_service_id(service_id.c_str());
	ptr_detecter->service_id_ = service_id;

	detecter_list_.push_back(ptr_detecter);
	detect_thread_list_.push_back(boost::thread(&VasDetectMaster::VideoDetect, this, detecter_list_.back()));
	lio::BoostSleep(500);
	lock.unlock();
	return true;
}

bool VasDetectMaster::AddDetecter(const vas::DetectParameter &param)
{
	if (!CheckIdUnique(param.service_id()))
	{
		return false;
	}
	boost::mutex::scoped_lock lock(mutex_);
	shared_ptr<Detecter> ptr_detecter = shared_ptr<Detecter>(new Detecter(param));
	detecter_list_.push_back(ptr_detecter);
	detect_thread_list_.push_back(boost::thread(&VasDetectMaster::VideoDetect, this, detecter_list_.back()));
	lio::BoostSleep(500);
	lock.unlock();
	return true;
}

bool VasDetectMaster::DeleteDetecter(const char *service_id)
{
	boost::mutex::scoped_lock lock(mutex_);
	vector<shared_ptr<Detecter>>::iterator iter;
	for (iter = detecter_list_.begin(); iter != detecter_list_.end(); iter++)
	{
		string t_service_id = (*iter)->service_id_;
		if (t_service_id.compare(String(service_id)) == 0)
		{
			(*iter)->detect_param_.set_detect_status(vas::DETECT_OFF);
			lio::BoostSleep(500);
			(*iter).reset();
			detecter_list_.erase(iter);
			lock.unlock();
			return true;
		}
	}

	lock.unlock();
	LOG(INFO) << " No service named " << service_id << " exist!" << endl;
	return false;
}
bool VasDetectMaster::DeleteDetecter(const string &service_id)
{
	return DeleteDetecter(service_id.c_str());
}

void VasDetectMaster::Pause(const char* service_id)
{
	boost::mutex::scoped_lock lock(mutex_);
	shared_ptr<Detecter> ptr_detecter = GetDetecterPtr(service_id);
	if (ptr_detecter)
	{
		ptr_detecter->detect_param_.set_detect_status(vas::DETECT_PAUSE);
		lio::BoostSleep(500);
		LOG(INFO) << ptr_detecter->service_id_ << " pause!" << endl;
	}
	lock.unlock();
}
void VasDetectMaster::Pause(const string& service_id)
{
	Pause(service_id.c_str());
}

void VasDetectMaster::Resume(const char* service_id)
{
	boost::mutex::scoped_lock lock(mutex_);
	shared_ptr<Detecter> ptr_detecter = GetDetecterPtr(service_id);
	if (ptr_detecter)
	{
		ptr_detecter->detect_param_.set_detect_status(vas::DETECT_ON);
		lio::BoostSleep(500);
		LOG(INFO) << ptr_detecter->service_id_ << " resume!" << endl;
	}
	lock.unlock();
}
void VasDetectMaster::Resume(const string& service_id)
{
	Resume(service_id.c_str());
}
void VasDetectMaster::Resume()
{
	boost::mutex::scoped_lock lock(mutex_);
	vector<shared_ptr<Detecter>>::iterator iter;
	for (iter = detecter_list_.begin(); iter != detecter_list_.end(); iter++)
	{
		(*iter)->detect_param_.set_detect_status(vas::DETECT_ON);
		lio::BoostSleep(10);
		LOG(INFO) << (*iter)->service_id_ << " resume!" << endl;
	}
	lock.unlock();
}

void VasDetectMaster::SetGetFrameFunction(std::function<bool(const char *service_id, Mat &frame, int64_t &pts, int64_t &No)> set_fun)
{
	get_frame_fun_ = set_fun;
}
void VasDetectMaster::SetCaffeRegFunction(std::function<vector<Prediction>(const Mat&, const int&)> set_fun)
{
	caffe_reg_fun_ = set_fun;
}

bool VasDetectMaster::CheckRegType(const string &label, const vas::DetectParameter &param)
{
	for (size_t i = 0; i < param.reg_type().size(); i++)
	{
		if (label.compare(param.reg_type(i)) == 0 && label.compare("smoke") != 0)
			return true;
	}
	return false;
}

void VasDetectMaster::MergeSmokeRects(vector<CvRect> &rects,int dst_frame_width, int dst_frame_height,int caffe_input_size)
{
	vector<CvRect> t_rects;
	for (size_t i = 0; i < rects.size(); i++)
	{
		CvRect adjust_smoke_rect, show_smoke_rect;
		lio::CorrectRect(rects[i], dst_frame_width, dst_frame_height);
		lio::RectToSquare(rects[i], adjust_smoke_rect, cvSize(dst_frame_width, dst_frame_height));
		if (adjust_smoke_rect.width < caffe_input_size)
		{
			lio::ScaleRect(adjust_smoke_rect, show_smoke_rect, cvSize(dst_frame_width, dst_frame_height), cvSize(caffe_input_size, caffe_input_size));
		}
		else
		{
			show_smoke_rect = adjust_smoke_rect;
		}
		t_rects.push_back(show_smoke_rect);
	}
	lio::MergeRects(t_rects, dst_frame_width, dst_frame_height);
	rects.swap(t_rects);
}
void VasDetectMaster::SetRegResultProcessFunction(signal_reg::slot_function_type fun)
{
	reg_signal_.connect(fun);
}
void VasDetectMaster::SetServiceCommandReplyProcessFunction(signal_service_command_reply::slot_function_type fun)
{
	service_command_reply_signal_.connect(fun);
}