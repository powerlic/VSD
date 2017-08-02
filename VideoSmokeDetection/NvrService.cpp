#include "stdafx.h"
#include "NvrService.h"


/*------------------------------NvrServiceInstance-----------------------------*/
NvrServiceInstance::NvrServiceInstance(const char* proto_file_name)
{
	CHECK(vas::ReadProtoFromTextFile(proto_file_name, &service_param_));
	service_id_ = service_param_.service_id();
	CheckParameter(service_param_);
}
NvrServiceInstance::NvrServiceInstance(const vas::NvrServiceParameter &param)
{
	service_param_.CopyFrom(param);
	service_id_ = service_param_.service_id();
	CheckParameter(service_param_);
}
NvrServiceInstance::~NvrServiceInstance()
{
	service_param_.Clear();
}
void NvrServiceInstance::CheckParameter(const vas::NvrServiceParameter &param)
{

}

/*---------------------------NvrService----------------------------------------*/
NvrService::NvrService()
{
}
NvrService::NvrService(const char* caffe_model_proto_file_name, const char*nvr_access_proto_file)
{
	ptr_nvr_master_ = shared_ptr<NvrMaster>(new NvrMaster(nvr_access_proto_file));
	ptr_detect_master_ = shared_ptr<VasDetectMaster>(new VasDetectMaster);
	ptr_detect_master_->SetCaffeRegFunction(std::bind(&NvrService::Predict, this, std::placeholders::_1, std::placeholders::_2));
	ptr_detect_master_->SetGetFrameFunction(std::bind(&NvrMaster::GetDecodedFrame, ptr_nvr_master_, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));
	InitCaffeModel(caffe_model_proto_file_name);
	instance_list_.clear();

}
NvrService::NvrService(const vas::CaffeModelPath &caffe_path, const vas::NvrParameter nvr_param)
{
	ptr_nvr_master_ = shared_ptr<NvrMaster>(new NvrMaster(nvr_param));
	ptr_detect_master_ = shared_ptr<VasDetectMaster>(new VasDetectMaster);
	ptr_detect_master_->SetCaffeRegFunction(std::bind(&NvrService::Predict, this, std::placeholders::_1, std::placeholders::_2));
	ptr_detect_master_->SetGetFrameFunction(std::bind(&NvrMaster::GetDecodedFrame, ptr_nvr_master_, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));
	InitCaffeModel(caffe_path);
	instance_list_.clear();
}
NvrService::~NvrService()
{


}

bool NvrService::AddService(const char *service_id, const uint32_t channel_no, const uint32_t reg_width, const uint32_t reg_height,
							const vector<string> &reg_types, const float smoke_sensitivity, const float fire_sensitivity)
{
	vas::NvrServiceParameter nvr_service_paramter;
	CHECK(vas::ReadProtoFromTextFile("..\\proto\\nvr_service_default_settting.prototxt", &nvr_service_paramter));
	nvr_service_paramter.set_service_id(service_id);
	nvr_service_paramter.mutable_nvr_channel()->set_service_id(service_id);
	nvr_service_paramter.mutable_nvr_channel()->set_cam_status(vas::CamStatus::CAM_ONLINE);
	nvr_service_paramter.mutable_nvr_channel()->set_channel_no(channel_no);
	nvr_service_paramter.mutable_nvr_channel()->set_dst_height(reg_height);
	nvr_service_paramter.mutable_nvr_channel()->set_dst_width(reg_width);

	nvr_service_paramter.mutable_detect_parameter()->set_service_id(service_id);

	SetRegTypes(reg_types, nvr_service_paramter);

	nvr_service_paramter.mutable_detect_parameter()->set_smoke_detect_sensitity(smoke_sensitivity);
	nvr_service_paramter.mutable_detect_parameter()->set_fire_detect_sensitity(fire_sensitivity);

	return AddService(nvr_service_paramter);
	
}

bool NvrService::AddService(const vas::NvrServiceParameter &param)
{
	shared_ptr<NvrServiceInstance> ptr_instance = shared_ptr<NvrServiceInstance>(new NvrServiceInstance(param));
	boost::mutex::scoped_lock lock(mutex_);
	if (!CheckIdUnique(ptr_instance->service_id_))
	{
		lock.unlock();
		return false;
	}
	bool ret_1 = ptr_nvr_master_->AddNvrVideoStream(ptr_instance->service_param_.nvr_channel());
	bool ret_2 = ptr_detect_master_->AddDetecter(ptr_instance->service_param_.detect_parameter());

	if (ret_1&& ret_2)
	{
		instance_list_.push_back(ptr_instance);
		lock.unlock();
		return true;
	}
	else if (ret_1 && !ret_2)
	{
		ptr_nvr_master_->DeleteNvrVideoStream(ptr_instance->service_id_);
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

bool NvrService::AddService(const char* proto_file_name)
{
	vas::NvrServiceParameter nvr_service_paramter;
	CHECK(vas::ReadProtoFromTextFile(proto_file_name, &nvr_service_paramter));
	return AddService(nvr_service_paramter);
}
bool NvrService::AddService(const vas::NvrChannel &nvr_channel_info, const vas::DetectParameter &detect_param)
{
	vas::NvrServiceParameter nvr_service_paramter;
	CHECK_EQ(nvr_channel_info.service_id(), detect_param.service_id()) << " Not same service_id!";
	nvr_service_paramter.mutable_nvr_channel()->CopyFrom(nvr_channel_info);
	nvr_service_paramter.mutable_detect_parameter()->CopyFrom(detect_param);
	nvr_service_paramter.set_service_id(nvr_service_paramter.service_id());
	return AddService(nvr_service_paramter);
}
void NvrService::StartServicesFromProtoFile(const char *proto_file_name)
{
	vas::NvrServiceList service_list;
	CHECK(vas::ReadProtoFromTextFile(proto_file_name, &service_list));
	for (size_t i = 0; i < service_list.service_parameter().size(); i++)
	{
		vas::NvrServiceParameter service_paramter = service_list.service_parameter(i);
		/*int r = rand();
		char n[10];
		sprintf(n,"%d",r);
		service_paramter.set_service_id(n);
		service_paramter.mutable_decode_parameter()->set_service_id(n);
		service_paramter.mutable_detect_parameter()->set_service_id(n);*/
		AddService(service_paramter);
	}

}
void NvrService::StartServicesFromProtoFile(const vas::NvrServiceList &nvr_service_list)
{
	for (size_t i = 0; i < nvr_service_list.service_parameter().size(); i++)
	{
		vas::NvrServiceParameter service_paramter = nvr_service_list.service_parameter(i);
		AddService(service_paramter);
	}
}

bool NvrService::DeleteService(const char* service_id)
{
	bool ret_1 = ptr_nvr_master_->DeleteNvrVideoStream(service_id);
	bool ret_2 = ptr_detect_master_->DeleteDetecter(service_id);

	vector<shared_ptr<NvrServiceInstance>>::iterator iter;
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
	if(ret_1 || ret_2)
	{
		//service_command_reply_signal_(service_id, vas::SERVICE_STOP_SUCESS, "Service stop succcess!");
		return true;
	}
	else
	{
		//service_command_reply_signal_(service_id, vas::SERVICE_STOP_FALIED, "Service stop failed!");
		return false;
	}
}
bool NvrService::DeleteService(const string &service_id)
{
	return DeleteService(service_id.c_str());
}
void NvrService::DeleteService()
{
	boost::mutex::scoped_lock lock(mutex_);
	if (instance_list_.size()>0)
	{
		vector<shared_ptr<NvrServiceInstance>>::iterator iter;
		for (iter = instance_list_.begin(); iter != instance_list_.end(); iter++)
		{
			string t_service_id = (*iter)->service_id_;
			bool ret_1 = ptr_nvr_master_->DeleteNvrVideoStream(t_service_id);
			bool ret_2 = ptr_detect_master_->DeleteDetecter(t_service_id);
		}
		instance_list_.clear();
	}
	//cv::destroyAllWindows();
	lock.unlock();
}

bool NvrService::GetRegResult(const char* service_id, Mat &show_frame, int64_t &pts)
{
	bool ret = ptr_detect_master_->GetRegResult(service_id, show_frame, pts);
	return ret;
}
bool NvrService::GetServiceParameter(const char*service_id, vas::NvrServiceParameter &service_parameter)
{
	shared_ptr<NvrServiceInstance> ptr_instance = GetServiceInstance(service_id);
	if (ptr_instance)
	{
		service_parameter.CopyFrom(ptr_instance->service_param_);
		return true;
	}
	else return false;
}
bool NvrService::GetServiceParameter(const string& service_id, vas::NvrServiceParameter &service_parameter)
{
	return GetServiceParameter(service_id.c_str(), service_parameter);
}
bool NvrService::SetRegResultProcessFunction(signal_reg::slot_function_type fun)
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
bool NvrService::SetStreamStatusErrorFunction(cam_signal_t::slot_function_type fun)
{
	if (ptr_nvr_master_)
	{
		ptr_nvr_master_->SetStreamStatusErrorFunction(fun);
		return true;
	}
	else
	{
		LOG(INFO) << "ptr_decode_master_ is not inited " << endl;
		return false;
	}
}

//bool NvrService::SetServiceCommandReplyProcessFunction(signal_service_command_reply::slot_function_type fun)
//{
	//if (ptr_detect_master_)
	//{
	//	ptr_detect_master_->SetServiceCommandReplyProcessFunction(fun);
	//}
	//else
	//{
	//	LOG(INFO) << "ptr_detect_master_ is not inited " << endl;
	//	return false;
	//}
	//if (ptr_nvr_master_)
	//{
	//	ptr_nvr_master_->SetServiceCommandReplyProcessFunction(fun);
	//}
	//else
	//{
	//	LOG(INFO) << "ptr_decode_master_ is not inited " << endl;
	//	return false;
	//}
	//service_command_reply_signal_.connect(fun);
//	return true;
//}

void NvrService::Reset()
{
	DeleteService();
	instance_list_.clear();
	ptr_nvr_master_.reset();
	ptr_detect_master_.reset();
	//ptr_classifier_.reset();
}



/*----------------------------------------------------------------------------------*/
vector<Prediction> NvrService::Predict(const Mat &image_mat, const int &N)
{
	boost::mutex::scoped_lock lock(caffe_mutex_);
	std::vector<Prediction> predictions;
	CHECK(ptr_classifier_) << " classifier is error" << endl;
	predictions = ptr_classifier_->Classify(image_mat, N);
	return predictions;
}

void NvrService::InitCaffeModel(const char *proto_file_name)
{
	CHECK(vas::ReadProtoFromTextFile(proto_file_name, &caffe_model_path_));
	ptr_classifier_ = shared_ptr<Classifier>(CreateClassifier(	caffe_model_path_.deploy_file_path(),
																caffe_model_path_.trained_model_path(),
																caffe_model_path_.mean_file_path(),
																caffe_model_path_.label_file_path(), caffe_model_path_.use_gpu()) );

	CHECK(ptr_classifier_);
}
void NvrService::InitCaffeModel(const vas::CaffeModelPath &caffe_path)
{
	caffe_model_path_.CopyFrom(caffe_path);
	ptr_classifier_ = shared_ptr<Classifier>(CreateClassifier(caffe_model_path_.deploy_file_path(),
		caffe_model_path_.trained_model_path(),
		caffe_model_path_.mean_file_path(),
		caffe_model_path_.label_file_path(), caffe_model_path_.use_gpu()));
	CHECK(ptr_classifier_);
}

bool NvrService::CheckIdUnique(const char* service_id)
{
	vector<shared_ptr<NvrServiceInstance>>::iterator iter;
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
bool NvrService::CheckIdUnique(const string &service_id)
{
	return CheckIdUnique(service_id.c_str());
}
void NvrService::SetRegTypes(const vector<string> &reg_types, vas::NvrServiceParameter &service_parameter)
{
	bool smoke_include = false;
	bool fire_include = false;
	vector<string> filtered_reg_types;
	for (size_t i = 0; i < reg_types.size(); i++)
	{
		if (reg_types[i].compare("smoke") == 0) smoke_include = true;
		else if (reg_types[i].compare("fire") == 0)
		{
			fire_include = true;
		}
		else filtered_reg_types.push_back(reg_types[i]);
	}
	service_parameter.mutable_detect_parameter()->mutable_reg_type()->Clear();
	if (smoke_include)
	{
		service_parameter.mutable_detect_parameter()->add_reg_type("smoke");
	}
	if (fire_include)
	{
		service_parameter.mutable_detect_parameter()->add_reg_type("fire");
	}
	for (size_t i = 0; i < filtered_reg_types.size(); i++)
	{
		service_parameter.mutable_detect_parameter()->add_reg_type(filtered_reg_types[i]);
	}

}
shared_ptr<NvrServiceInstance> NvrService::GetServiceInstance(const string &service_id)
{
	return GetServiceInstance(service_id.c_str());
}
shared_ptr<NvrServiceInstance> NvrService::GetServiceInstance(const char*service_id)
{
	boost::mutex::scoped_lock lock(mutex_);
	vector<shared_ptr<NvrServiceInstance>>::iterator iter;
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