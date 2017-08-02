#include "stdafx.h"
#include "VasSsdMaster.h"

/*-------------------------------Detecter-------------------------------------*/


SsdDetecter::SsdDetecter()
{
	detect_param_ = vas::SsdDetecterParameter();
	CheckParameter(detect_param_);
}
SsdDetecter::~SsdDetecter()
{

	raw_frame_.release();
	reg_frame_.release();
	show_frame_.release();
}

SsdDetecter::SsdDetecter(const char *file_name)
{
	bool success = vas::ReadProtoFromTextFile(file_name, &detect_param_);
	if (success)
	{
		service_id_ = detect_param_.service_id();
		CheckParameter(detect_param_);
	}
}

SsdDetecter::SsdDetecter(const vas::SsdDetecterParameter &param)
{
	detect_param_.CopyFrom(param);
	service_id_ = param.service_id();
	CheckParameter(detect_param_);
}

void SsdDetecter::CheckParameter(const vas::SsdDetecterParameter &param)
{
	CHECK(param.has_service_id()) << " No service id " << endl;
	CHECK_GE(param.reg_height(), 576) << " Reg height must be greater or equal to than 576!" << endl;
	CHECK_GE(param.reg_width(), 800) << " Reg height must be greater than or equal to 800!" << endl;
	CHECK_GE(param.reg_interval(), 20) << "Reg time interval must be greater than or equal to 20 " << endl;
}





/*-------------------------------DetectMaster-------------------------------------*/

VasSsdMaster::VasSsdMaster()
{

}


VasSsdMaster::~VasSsdMaster()
{


}


bool VasSsdMaster::AddDetecter(const char *proto_file_name)
{
	boost::mutex::scoped_lock lock(mutex_);
	vas::SsdDetecterParameter param;
	CHECK(vas::ReadProtoFromTextFile(proto_file_name, &param)) << "read " << proto_file_name << " failed " << endl;

	if (CheckIdUnique(param.service_id()))
	{
		lock.unlock();
		return false;
	}

	shared_ptr<SsdDetecter> ptr_detecter = shared_ptr<SsdDetecter>(new SsdDetecter(proto_file_name));
	detecter_list_.push_back(ptr_detecter);
	detect_thread_list_.push_back(boost::thread(&VasSsdMaster::VideoDetect, this, detecter_list_.back()));
	lio::BoostSleep(500);
	lock.unlock();
	return true;
}

bool VasSsdMaster::AddDetecter(const string &service_id)
{
	if (CheckIdUnique(service_id))
	{
		return false;
	}
	boost::mutex::scoped_lock lock(mutex_);
	shared_ptr<SsdDetecter> ptr_detecter = shared_ptr<SsdDetecter>(new SsdDetecter());
	ptr_detecter->detect_param_.set_service_id(service_id.c_str());
	ptr_detecter->service_id_ = service_id;

	detecter_list_.push_back(ptr_detecter);
	detect_thread_list_.push_back(boost::thread(&VasSsdMaster::VideoDetect, this, detecter_list_.back()));
	lio::BoostSleep(500);
	lock.unlock();
	return true;
}

bool VasSsdMaster::AddDetecter(const vas::SsdDetecterParameter &param)
{
	if (!CheckIdUnique(param.service_id()))
	{
		return false;
	}
	boost::mutex::scoped_lock lock(mutex_);
	shared_ptr<SsdDetecter> ptr_detecter = shared_ptr<SsdDetecter>(new SsdDetecter(param));
	detecter_list_.push_back(ptr_detecter);
	detect_thread_list_.push_back(boost::thread(&VasSsdMaster::VideoDetect, this, detecter_list_.back()));
	lio::BoostSleep(500);
	lock.unlock();
	return true;
}

bool VasSsdMaster::DeleteDetecter(const char *service_id)
{
	boost::mutex::scoped_lock lock(mutex_);
	vector<shared_ptr<SsdDetecter>>::iterator iter;
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
bool VasSsdMaster::DeleteDetecter(const string &service_id)
{
	return DeleteDetecter(service_id.c_str());
}

bool VasSsdMaster::SetShowRegFrame(const char* service_id, bool is_show)
{
	shared_ptr<SsdDetecter> ptr_detecter = GetDetecterPtr(service_id);
	if (ptr_detecter)
	{
		ptr_detecter->detect_param_.set_show_result_frame(is_show);
		return true;
	}
	else return false;
}


void VasSsdMaster::CreateDetecterContext(shared_ptr<SsdDetecter> ptr_detecter)
{
	CHECK(ptr_detecter->detect_param_.IsInitialized()) << ptr_detecter->service_id_ << " ssd_ detect_parameter is not inited!" << endl;

	if (ptr_detecter->detect_param_.save_video())
	{
		SetupVideoSaveFile(ptr_detecter);
		//ptr_detecter->video_writer_ = VideoWriter("VideoTest.avi", CV_FOURCC('M', 'J', 'P', 'G'), 25.0, Size(ptr_detecter->detect_param_.reg_width(), ptr_detecter->detect_param_.reg_height()));
	}


	ptr_detecter->detect_param_.set_detect_status(vas::DETECT_ON);

	if (ptr_detecter->detect_param_.show_result_frame())
	{
		namedWindow(ptr_detecter->service_id_, 1);
		ptr_detecter->has_window_ = true;
	}



}
void VasSsdMaster::SetupVideoSaveFile(shared_ptr<SsdDetecter> ptr_detecter)
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
void VasSsdMaster::DestroyDetecterContext(shared_ptr<SsdDetecter> ptr_detecter)
{
	ptr_detecter->raw_frame_.release();
	ptr_detecter->reg_frame_.release();
	ptr_detecter->show_frame_.release();

	if (ptr_detecter->detect_param_.save_video())
	{
		ptr_detecter->video_writer_.release();
	}
	ptr_detecter->detect_param_.set_detect_status(vas::DETECT_OFF);

	if (ptr_detecter->has_window_)
	{
		destroyWindow(ptr_detecter->service_id_);
	}
}
bool VasSsdMaster::UpdateFrame(shared_ptr<SsdDetecter> ptr_detecter, const Mat &frame, const int64_t &pts, const int64_t &No)
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

shared_ptr<SsdDetecter> VasSsdMaster::GetDetecterPtr(const char *service_id)
{
	boost::mutex::scoped_lock lock(mutex_);
	vector<shared_ptr<SsdDetecter>>::iterator iter;
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

shared_ptr<SsdDetecter> VasSsdMaster::GetDetecterPtr(const string &service_id)
{
	boost::mutex::scoped_lock lock(mutex_);
	vector<shared_ptr<SsdDetecter>>::iterator iter;
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

void VasSsdMaster::DetectFrame(shared_ptr<SsdDetecter> ptr_detecter, const Mat &frame, const int64_t &pts, const int64_t &No)
{

	CHECK(UpdateFrame(ptr_detecter, frame, pts, No));

	vector<DetectRect> detect_rects=ssd_detect_fun_(ptr_detecter->reg_frame_);

	vector<CvRect> utified_reg_rects;
	vector<float> utified_reg_scores;
	vector<string> utifiled_reg_labels;

	int reg_height, reg_width;
	reg_width = ptr_detecter->detect_param_.reg_width();
	reg_height = ptr_detecter->detect_param_.reg_height();
	for (size_t i = 0; i < detect_rects.size(); i++)
	{

		if (CheckRegType(detect_rects[i].label, ptr_detecter->detect_param_) && detect_rects[i].score> ptr_detecter->detect_param_.confidence_threshold())
		{
			CvRect rect;
			rect.x = detect_rects[i].x1*reg_width;
			rect.y = detect_rects[i].y1*reg_height;
			rect.width = detect_rects[i].x2*reg_width - detect_rects[i].x1*reg_width;
			rect.height = detect_rects[i].y2*reg_height - detect_rects[i].y1*reg_height;
			
			utified_reg_rects.push_back(rect);
			utified_reg_scores.push_back(detect_rects[i].score);
			utifiled_reg_labels.push_back(detect_rects[i].label);

			if (ptr_detecter->detect_param_.show_result_frame())
			{
				CvPoint P1, P2, PText;
				P1.x = rect.x;
				P1.y = rect.y;
				P2.x = rect.x + rect.width;
				P2.y = rect.y + rect.height;
				rectangle(ptr_detecter->show_frame_, P1, P2, CV_RGB(0, 255, 0), 1);
				lio::WtrieLabel(ptr_detecter->show_frame_, rect, detect_rects[i].label, 0, 255, 0);
			}
		}
	}

	if (utifiled_reg_labels.size()>0)
	{
		reg_signal_(ptr_detecter->service_id_.c_str(), ptr_detecter->show_frame_, utified_reg_rects, reg_width, reg_height, utified_reg_scores, utifiled_reg_labels);
	}
}

bool VasSsdMaster::CheckRegType(const string &label, const vas::SsdDetecterParameter &param)
{
	for (size_t i = 0; i < param.reg_type().size(); i++)
	{
		if (label.compare(param.reg_type(i)) == 0 && label.compare("smoke") != 0)
			return true;
	}
	return false;
}

void VasSsdMaster::SetRegResultProcessFunction(signal_reg::slot_function_type fun)
{
	reg_signal_.connect(fun);
}
void VasSsdMaster::SetSsdDetectFunction(std::function<vector<DetectRect>(const Mat&)> set_fun)
{
	ssd_detect_fun_ = set_fun;
}
void VasSsdMaster::SetGetFrameFunction(std::function<bool(const char *service_id, Mat &frame, int64_t &pts, int64_t &No)> set_fun)
{
	get_frame_fun_ = set_fun;
}

void VasSsdMaster::VideoDetect(shared_ptr<SsdDetecter> ptr_detecter)
{

	Mat frame;
	int64_t pts;
	int64_t No;


	while (true)
	{
		if (ptr_detecter->detect_param_.detect_status() == vas::DETECT_OFF)
		{
			break;
		}
		CreateDetecterContext(ptr_detecter);
		LOG(INFO) << ptr_detecter->service_id_ << " create detect process success!" << endl;
		while (ptr_detecter->detect_param_.detect_status() != vas::DETECT_OFF)
		{
			if (ptr_detecter->detect_param_.detect_status() == vas::DETECT_PAUSE)
			{
				lio::BoostSleep(500);
				continue;
			}
			bool got_frame = get_frame_fun_(ptr_detecter->service_id_.c_str(), frame, pts, No);
			if (got_frame)
			{
				//time_t c1, c2;
				//c1 = clock();
				DetectFrame(ptr_detecter, frame, pts, No);
				//c2 = clock();
				//LOG(INFO) << " detect needs " << c2 - c1 << endl;
				if (ptr_detecter->detect_param_.show_result_frame())
				{
					if (!ptr_detecter->has_window_)
					{
						ptr_detecter->has_window_ = true;
						namedWindow(ptr_detecter->service_id_, 1);
					}
					imshow(ptr_detecter->service_id_, ptr_detecter->show_frame_);
					waitKey(1);
				}
				else
				{
					if (ptr_detecter->has_window_)
					{
						destroyWindow(ptr_detecter->service_id_);
						ptr_detecter->has_window_ = false;
					}
				}

			}
			else
			{
				lio::BoostSleep(10);
			}

			lio::BoostSleep(ptr_detecter->detect_param_.reg_interval());
		}
		DestroyDetecterContext(ptr_detecter);
		lio::BoostSleep(1000);
	}

}

bool VasSsdMaster::CheckIdUnique(const char* service_id)
{
	vector<shared_ptr<SsdDetecter>>::iterator iter;
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

bool VasSsdMaster::CheckIdUnique(const string &service_id)
{
	vector<shared_ptr<SsdDetecter>>::iterator iter;
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