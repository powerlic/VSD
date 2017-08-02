#include "stdafx.h"
#include "SsdService.h"


/*--------------------------------------SsdInstance--------------------------------------------*/
SsdInstance::SsdInstance(const char* proto_file_name)
{
	CHECK(vas::ReadProtoFromTextFile(proto_file_name, &service_param_));
	service_id_ = service_param_.service_id();
	CheckParameter(service_param_);
}

SsdInstance::SsdInstance(const vas::SsdServiceParameter &param)
{
	service_param_.CopyFrom(param);
	service_id_ = service_param_.service_id();
	CheckParameter(service_param_);
}
SsdInstance::SsdInstance()
{
	service_param_ = vas::SsdServiceParameter();
	service_param_.mutable_decode_parameter();
	service_param_.mutable_detect_parameter();
}
SsdInstance::~SsdInstance()
{
	service_param_.Clear();
}

void SsdInstance::CheckParameter(const vas::SsdServiceParameter &param)
{
	CHECK(param.has_service_id()) << " service_id not set!" << endl;
}



/*------------------------SsdService-------------------------------------------*/

SsdService::SsdService()
{
	ptr_decode_master_ = shared_ptr<VasDecodeMaster>(new VasDecodeMaster);
	ptr_detect_master_ = shared_ptr<VasSsdMaster>(new VasSsdMaster);
	ptr_detect_master_->SetSsdDetectFunction(std::bind(&SsdService::Detect, this, std::placeholders::_1));
	ptr_detect_master_->SetGetFrameFunction(std::bind(&VasDecodeMaster::GetDecodedFrame, ptr_decode_master_, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));
	save_reg_frame_queue_.clear();
	save_reg_frame_thread_ = boost::thread(&SsdService::SaveRegFrameFunction, this);
	InitSsdModel("..\\proto\\ssd_model.prototxt");
	instance_list_.clear();

}

SsdService::SsdService(const char* ssd_model_proto_file_name)
{
	ptr_decode_master_ = shared_ptr<VasDecodeMaster>(new VasDecodeMaster);
	ptr_detect_master_ = shared_ptr<VasSsdMaster>(new VasSsdMaster);
	ptr_detect_master_->SetSsdDetectFunction(std::bind(&SsdService::Detect, this, std::placeholders::_1));
	ptr_detect_master_->SetGetFrameFunction(std::bind(&VasDecodeMaster::GetDecodedFrame, ptr_decode_master_, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));
	save_reg_frame_queue_.clear();
	save_reg_frame_thread_ = boost::thread(&SsdService::SaveRegFrameFunction, this);
	InitSsdModel(ssd_model_proto_file_name);
	instance_list_.clear();
}
SsdService::SsdService(const vas::SsdModelPath &ssd_model_path)
{
	ptr_decode_master_ = shared_ptr<VasDecodeMaster>(new VasDecodeMaster);
	ptr_detect_master_ = shared_ptr<VasSsdMaster>(new VasSsdMaster);
	ptr_detect_master_->SetSsdDetectFunction(std::bind(&SsdService::Detect, this, std::placeholders::_1));
	ptr_detect_master_->SetGetFrameFunction(std::bind(&VasDecodeMaster::GetDecodedFrame, ptr_decode_master_, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));
	save_reg_frame_queue_.clear();
	save_reg_frame_thread_ = boost::thread(&SsdService::SaveRegFrameFunction, this);
	InitSsdModel(ssd_model_path);
	instance_list_.clear();
}
SsdService::SsdService(const string& model_file, const string& trained_file, const string& mean_file, const string&mean_value, const string& label_file)
{
	ptr_decode_master_ = shared_ptr<VasDecodeMaster>(new VasDecodeMaster);
	ptr_detect_master_ = shared_ptr<VasSsdMaster>(new VasSsdMaster);
	ptr_detect_master_->SetSsdDetectFunction(std::bind(&SsdService::Detect, this, std::placeholders::_1));
	ptr_detect_master_->SetGetFrameFunction(std::bind(&VasDecodeMaster::GetDecodedFrame, ptr_decode_master_, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));
	save_reg_frame_queue_.clear();
	save_reg_frame_thread_ = boost::thread(&SsdService::SaveRegFrameFunction, this);
	vas::SsdModelPath model_path;
	model_path.set_deploy_file_path(model_file);
	model_path.set_trained_model_path(trained_file);
	model_path.set_label_file_path(label_file);
	model_path.set_mean_file_path(mean_file);
	model_path.set_mean_value(mean_value);
	InitSsdModel(model_path);
	instance_list_.clear();

}
SsdService::~SsdService()
{
	instance_list_.clear();
	ptr_detect_master_.reset();
	ptr_decode_master_.reset();
	save_reg_frame_queue_.clear();
	save_reg_frame_thread_.interrupt();
}

vector<DetectRect> SsdService::Detect(const Mat &frame)
{
	boost::mutex::scoped_lock lock(caffe_mutex_);
	vector<DetectRect> detect_rects = ptr_ssd_->Detect(frame);
	return detect_rects;
}

void SsdService::SaveRegFrameFunction()
{
	while (true)
	{
		while (save_reg_frame_queue_.size()>0)
		{
			SaveRegMat reg_res = save_reg_frame_queue_.pop();

			string name_str = lio::GetDateTimeString();
			string file_path;
			file_path = "./regres/" + reg_res.label;
			fstream _file;
			_file.open(file_path, ios::in);
			if (!_file)
			{
				_mkdir(file_path.c_str());
			}

			name_str = "./regres/" + reg_res.label + "/" + String(reg_res.service_id) + "_" + name_str + ".jpg";
			imwrite(name_str, reg_res.reg_frame);
			boost::this_thread::sleep(boost::get_system_time() + boost::posix_time::milliseconds(50));
		}

		boost::this_thread::sleep(boost::get_system_time() + boost::posix_time::milliseconds(500));
	}

}

void SsdService::InitSsdModel(const char *proto_file_name)
{
	CHECK(vas::ReadProtoFromTextFile(proto_file_name, &ssd_model_path_));
	ptr_ssd_ = shared_ptr<SsdP>(CreateSsdP(ssd_model_path_.deploy_file_path(), 
											ssd_model_path_.trained_model_path(), 
											ssd_model_path_.mean_file_path(), 
											ssd_model_path_.mean_value(), 
											ssd_model_path_.label_file_path()));
	CHECK(ptr_ssd_);
}
void SsdService::InitSsdModel(const vas::SsdModelPath &ssd_model_path)
{
	ssd_model_path_.CopyFrom(ssd_model_path);
	ptr_ssd_ = shared_ptr<SsdP>(CreateSsdP(ssd_model_path_.deploy_file_path(),
		ssd_model_path_.trained_model_path(),
		ssd_model_path_.mean_file_path(),
		ssd_model_path_.mean_value(),
		ssd_model_path_.label_file_path()));
	CHECK(ptr_ssd_);
}

bool SsdService::CheckIdUnique(const char* service_id)
{
	vector<shared_ptr<SsdInstance>>::iterator iter;
	for (iter = instance_list_.begin(); iter != instance_list_.end(); iter++)
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
bool SsdService::CheckIdUnique(const string &service_id)
{
	return CheckIdUnique(service_id.c_str());
}
bool SsdService::AddService(const char *proto_file_name)
{
	shared_ptr<SsdInstance> ptr_instance = shared_ptr<SsdInstance>(new SsdInstance(proto_file_name));

	bool id_unique = CheckIdUnique(ptr_instance->service_id_);
	bool ret_1 = ptr_decode_master_->AddVideoStream(ptr_instance->service_param_.decode_parameter());
	bool ret_2 = ptr_detect_master_->AddDetecter(ptr_instance->service_param_.detect_parameter());

	if (ret_1&&ret_2)
	{
		instance_list_.push_back(ptr_instance);
		return true;
	}
	else if (ret_1&& !ret_2)
	{
		ptr_decode_master_->DeleteVideoStream(ptr_instance->service_id_);
		return false;
	}
	else if (ret_2&& !ret_1)
	{
		ptr_detect_master_->DeleteDetecter(ptr_instance->service_id_);
		return false;
	}

}
bool SsdService::AddService(const vas::DecodeParameter &decode_param, const vas::SsdDetecterParameter &detect_param)
{
	vas::SsdServiceParameter param;
	CHECK_EQ(decode_param.service_id(), detect_param.service_id()) << " Not same service_id!";
	param.mutable_decode_parameter()->CopyFrom(decode_param);
	param.mutable_detect_parameter()->CopyFrom(detect_param);
	param.set_service_id(decode_param.service_id());
	return AddService(param);
}

bool SsdService::AddService(const vas::SsdServiceParameter &param)
{
	shared_ptr<SsdInstance> ptr_instance = shared_ptr<SsdInstance>(new SsdInstance(param));

	boost::mutex::scoped_lock lock(mutex_);
	if (!CheckIdUnique(ptr_instance->service_id_))
	{
		lock.unlock();
		return false;
	}

	bool ret_1 = ptr_decode_master_->AddVideoStream(ptr_instance->service_param_.decode_parameter());
	bool ret_2 = ptr_detect_master_->AddDetecter(ptr_instance->service_param_.detect_parameter());

	if (ret_1&& ret_2)
	{
		instance_list_.push_back(ptr_instance);
		lock.unlock();
		return true;
	}
	else if (ret_1 && !ret_2)
	{
		ptr_decode_master_->DeleteVideoStream(ptr_instance->service_id_);
		lock.unlock();
		return false;
	}
	else if (!ret_1 && ret_2)
	{
		ptr_detect_master_->DeleteDetecter(ptr_instance->service_id_);
		lock.unlock();
		return false;
	}

}

bool SsdService::AddService(const char *service_id, const char *url, const int url_type,
	const int decode_mode, const uint32_t reg_width, const uint32_t reg_height,const vector<string> &reg_types)
{
	vas::SsdServiceParameter service_paramter;
	CHECK(vas::ReadProtoFromTextFile("..\\proto\\ssd_service_default_settting.prototxt", &service_paramter));
	service_paramter.set_service_id(service_id);
	service_paramter.mutable_detect_parameter()->set_service_id(service_id);
	service_paramter.mutable_decode_parameter()->set_service_id(service_id);
	service_paramter.mutable_decode_parameter()->set_url(url);

	if (url_type == 0)
	{
		service_paramter.mutable_decode_parameter()->set_video_source(vas::RTSP_STREAM);
	}
	else if (url_type == 1)
	{
		service_paramter.mutable_decode_parameter()->set_video_source(vas::VIDEO_FILE);
	}

	SetRegTypes(reg_types, service_paramter);
	AddService(service_paramter);
	return true;
}

void SsdService::SetRegTypes(const vector<string> &reg_types, vas::SsdServiceParameter &service_parameter)
{
	for (size_t i = 0; i < reg_types.size(); i++)
	{
		service_parameter.mutable_detect_parameter()->add_reg_type(reg_types[i]);
	}
}
void SsdService::StartServicesFromProtoFile(const char *proto_file_name)
{
	vas::SsdServiceList service_list;
	CHECK(vas::ReadProtoFromTextFile(proto_file_name, &service_list));
	for (size_t i = 0; i < service_list.service_paramter_size(); i++)
	{
		const vas::SsdServiceParameter &service_paramter = service_list.service_paramter(i);
		AddService(service_paramter);
	}
}
void SsdService::StartServicesFromProtoFile(const vas::SsdServiceList &service_list)
{
	for (size_t i = 0; i < service_list.service_paramter_size(); i++)
	{
		const vas::SsdServiceParameter &service_paramter = service_list.service_paramter(i);
		AddService(service_paramter);
	}
}
bool SsdService::SetShowRegFrame(const char* service_id, bool is_show)
{
	return ptr_detect_master_->SetShowRegFrame(service_id, is_show);
}

bool SsdService::DeleteService(const char* service_id)
{
	bool ret_1 = ptr_decode_master_->DeleteVideoStream(service_id);
	bool ret_2 = ptr_detect_master_->DeleteDetecter(service_id);

	vector<shared_ptr<SsdInstance>>::iterator iter;
	boost::mutex::scoped_lock lock(mutex_);
	for (iter = instance_list_.begin(); iter != instance_list_.end(); iter++)
	{
		if ((*iter)->service_id_.compare(service_id) == 0)
		{
			instance_list_.erase(iter);
			break;
		}
	}
	lock.unlock();
	if (ret_1 || ret_2)
	{
		return true;
	}
	else
	{
		return false;
	}

}

bool SsdService::DeleteService(const string &service_id)
{
	return DeleteService(service_id.c_str());
}

void SsdService::DeleteService()
{
	boost::mutex::scoped_lock lock(mutex_);
	if (instance_list_.size()>0)
	{
		vector<shared_ptr<SsdInstance>>::iterator iter;
		for (iter = instance_list_.begin(); iter != instance_list_.end(); iter++)
		{
			string t_service_id = (*iter)->service_id_;
			bool ret_1 = ptr_decode_master_->DeleteVideoStream(t_service_id);
			bool ret_2 = ptr_detect_master_->DeleteDetecter(t_service_id);
		}
		instance_list_.clear();
	}
	//cv::destroyAllWindows();
	lock.unlock();
}

bool SsdService::GetServiceParameter(const char* service_id, vas::SsdServiceParameter &service_parameter)
{
	shared_ptr<SsdInstance> ptr_instance = GetServiceInstance(service_id);
	if (ptr_instance)
	{
		service_parameter.CopyFrom(ptr_instance->service_param_);
		return true;
	}
	else return false;
}
bool SsdService::GetServiceStreamStatus(const char* service_id, vas::StreamStatus &stream_status)
{
	shared_ptr<SsdInstance> ptr_instance = GetServiceInstance(service_id);
	if (ptr_instance)
	{
		stream_status = ptr_instance->service_param_.decode_parameter().stream_status();
		return true;
	}
	else return false;
}

bool SsdService::GetServiceParameter(const string &service_id, vas::SsdServiceParameter &service_parameter)
{
	return GetServiceParameter(service_id.c_str(), service_parameter);
}

shared_ptr<SsdInstance> SsdService::GetServiceInstance(const char*service_id)
{
	boost::mutex::scoped_lock lock(mutex_);
	vector<shared_ptr<SsdInstance>>::iterator iter;
	for (iter = instance_list_.begin(); iter != instance_list_.end(); iter++)
	{
		string t_service_id = (*iter)->service_id_;
		if (t_service_id.compare(String(service_id)) == 0)
		{
			lock.unlock();
			return *iter;
		}
	}
	lock.unlock();
	LOG(INFO) << " No service named " << service_id << endl;
	return NULL;
}

bool SsdService::SetRegResultProcessFunction(signal_reg::slot_function_type fun)
{
	if (ptr_detect_master_)
	{
		ptr_detect_master_->SetRegResultProcessFunction(fun);
		return true;
	}
	else
	{
		LOG(INFO) << "ptr_detect_master_ is not inited " << endl;
		return false;
	}
}

bool SsdService::SetStreamStatusErrorFunction(signal_t::slot_function_type fun)
{
	if (ptr_decode_master_)
	{
		ptr_decode_master_->SetStreamStatusErrorFunction(fun);
		return true;
	}
	else
	{
		LOG(INFO) << "ptr_decode_master_ is not inited " << endl;
		return false;
	}
}

void SsdService::Reset()
{
	DeleteService();
	instance_list_.clear();
	ptr_decode_master_.reset();
	ptr_detect_master_.reset();
	save_reg_frame_thread_.interrupt();

}

void SsdService::GetServices(vector<string> &service_list)
{
	service_list.clear();
	for (size_t i = 0; i < instance_list_.size(); i++)
	{
		service_list.push_back(instance_list_[i]->service_id_);
	}
}

bool SsdService::QuerySerive(const char *service_id, vas::ServiceStatus &status)
{
	shared_ptr<SsdInstance> ptr_instance = GetServiceInstance(service_id);
	if (!ptr_instance)
	{
		status = vas::SERVICE_UNKNOWN;
		return false;
	}
	vas::SsdServiceParameter service_param_;
	bool ret = GetServiceParameter(service_id, service_param_);
	status = service_param_.service_status();
	return true;
}