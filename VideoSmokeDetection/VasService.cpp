#define _DLL_VASP
#include "stdafx.h"
#include"VasP.h"
#include "VasService.h"

/*--------------------------------------VasInstance--------------------------------------------*/
VasInstance::VasInstance(const char* proto_file_name)
{
	CHECK(vas::ReadProtoFromTextFile(proto_file_name, &service_param_));
	service_id_ = service_param_.service_id();
	CheckParameter(service_param_);
}

VasInstance::VasInstance(const vas::ServiceParameter &param)
{
	service_param_.CopyFrom(param);
	service_id_ = service_param_.service_id();
	CheckParameter(service_param_);
}
VasInstance::VasInstance()
{
	service_param_ = vas::ServiceParameter();
	service_param_.mutable_decode_parameter();
	service_param_.mutable_detect_parameter();
}
VasInstance::~VasInstance()
{
	service_param_.Clear();
}

void VasInstance::CheckParameter(const vas::ServiceParameter &param)
{
	CHECK(param.has_service_id()) << " service_id not set!" << endl;
}


/*------------------------VasService-------------------------------------------*/
VasService::VasService()
{
	ptr_decode_master_ = shared_ptr<VasDecodeMaster>(new VasDecodeMaster);
	ptr_detect_master_ = shared_ptr<VasDetectMaster>(new VasDetectMaster);
	ptr_detect_master_->SetCaffeRegFunction(std::bind(&VasService::Predict, this, std::placeholders::_1, std::placeholders::_2));
	ptr_detect_master_->SetGetFrameFunction(std::bind(&VasDecodeMaster::GetDecodedFrame, ptr_decode_master_, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));
	save_reg_frame_queue_.clear();
	save_reg_frame_thread_ = boost::thread(&VasService::SaveRegFrameFunction, this);
	InitCaffeModel("..\\proto\\caffe_model.prototxt");
	instance_list_.clear();
}
VasService::VasService(const char* caffe_model_proto_file_name)
{
	ptr_decode_master_ = shared_ptr<VasDecodeMaster>(new VasDecodeMaster);
	ptr_detect_master_ = shared_ptr<VasDetectMaster>(new VasDetectMaster);
	ptr_detect_master_->SetCaffeRegFunction(std::bind(&VasService::Predict, this, std::placeholders::_1, std::placeholders::_2));
	ptr_detect_master_->SetGetFrameFunction(std::bind(&VasDecodeMaster::GetDecodedFrame, ptr_decode_master_, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));
	save_reg_frame_queue_.clear();
	save_reg_frame_thread_ = boost::thread(&VasService::SaveRegFrameFunction, this);
	InitCaffeModel(caffe_model_proto_file_name);
	instance_list_.clear();
}
VasService::VasService(const vas::CaffeModelPath &caffe_path)
{
	ptr_decode_master_ = shared_ptr<VasDecodeMaster>(new VasDecodeMaster);
	ptr_detect_master_ = shared_ptr<VasDetectMaster>(new VasDetectMaster);
	ptr_detect_master_->SetCaffeRegFunction(std::bind(&VasService::Predict, this, std::placeholders::_1, std::placeholders::_2));
	ptr_detect_master_->SetGetFrameFunction(std::bind(&VasDecodeMaster::GetDecodedFrame, ptr_decode_master_, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));
	save_reg_frame_queue_.clear();
	save_reg_frame_thread_ = boost::thread(&VasService::SaveRegFrameFunction, this);
	InitCaffeModel(caffe_path);
	instance_list_.clear();
}

VasService::~VasService()
{
	instance_list_.clear();
	ptr_classifier_->Release();
	ptr_detect_master_.reset();
	ptr_decode_master_.reset();
	save_reg_frame_thread_.interrupt();
}

vector<Prediction> VasService::Predict(const Mat &image_mat, const int &N)
{
	boost::mutex::scoped_lock lock(caffe_mutex_);
	std::vector<Prediction> predictions;
	predictions = ptr_classifier_->Classify(image_mat, N);
	return predictions;
}

void VasService::SaveRegFrameFunction()
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

void VasService::InitCaffeModel(const char *proto_file_name)
{
	CHECK(vas::ReadProtoFromTextFile(proto_file_name, &caffe_model_path_));
	ptr_classifier_ = shared_ptr<Classifier>(CreateClassifier(caffe_model_path_.deploy_file_path(), 
															  caffe_model_path_.trained_model_path(),
															  caffe_model_path_.mean_file_path(), 
															  caffe_model_path_.label_file_path(), caffe_model_path_.use_gpu()));
	CHECK(ptr_classifier_);
}

void VasService::InitCaffeModel(const vas::CaffeModelPath &caffe_path)
{
	caffe_model_path_.CopyFrom(caffe_path);
	ptr_classifier_ = shared_ptr<Classifier>(CreateClassifier(caffe_model_path_.deploy_file_path(),
		caffe_model_path_.trained_model_path(),
		caffe_model_path_.mean_file_path(),
		caffe_model_path_.label_file_path(), caffe_model_path_.use_gpu()));
	CHECK(ptr_classifier_);
}
bool VasService::CheckIdUnique(const char* service_id)
{
	vector<shared_ptr<VasInstance>>::iterator iter;
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
bool VasService::CheckIdUnique(const string &service_id)
{
	return CheckIdUnique(service_id.c_str());
}

bool VasService::GetRegResult(const char* service_id, Mat &show_frame, int64_t &pts)
{
	bool ret = ptr_detect_master_->GetRegResult(service_id, show_frame, pts);
	return ret;
}

bool VasService::AddService(const char *proto_file_name)
{
	shared_ptr<VasInstance> ptr_instance = shared_ptr<VasInstance>(new VasInstance(proto_file_name));

	bool id_unique = CheckIdUnique(ptr_instance->service_id_);
	/*if (id_unique);
	{
		return false;
	}*/

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
bool VasService::AddService(const vas::DecodeParameter &decode_param, const vas::DetectParameter &detect_param)
{
	vas::ServiceParameter param;
	CHECK_EQ(decode_param.service_id(), detect_param.service_id())<<" Not same service_id!";
	param.mutable_decode_parameter()->CopyFrom(decode_param);
	param.mutable_detect_parameter()->CopyFrom(detect_param);
	param.set_service_id(decode_param.service_id());
	return AddService(param);
}

bool VasService::AddService(const vas::ServiceParameter &param)
{
	shared_ptr<VasInstance> ptr_instance = shared_ptr<VasInstance>(new VasInstance(param));

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
	else if (ret_1 && !ret_2 )
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

bool VasService::AddService(const char *service_id, const char *url, const int url_type,
				            const int decode_mode, const uint32_t reg_width, const uint32_t reg_height,
							const vector<string> &reg_types, const float smoke_sensitivity, const float fire_sensitivity)
{

	vas::ServiceParameter service_paramter;
	CHECK(vas::ReadProtoFromTextFile("..\\proto\\service_default_settting.prototxt", &service_paramter));
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
	service_paramter.mutable_detect_parameter()->set_smoke_detect_sensitity(smoke_sensitivity);
	service_paramter.mutable_detect_parameter()->set_fire_detect_sensitity(fire_sensitivity);
	AddService(service_paramter);
	return true;
}

void VasService::SetRegTypes(const vector<string> &reg_types, vas::ServiceParameter &service_parameter)
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

void VasService::StartServicesFromProtoFile(const char *proto_file_name)
{
	vas::ServiceList service_list;
	CHECK(vas::ReadProtoFromTextFile(proto_file_name, &service_list));
	for (size_t i = 0; i < service_list.service_paramter_size(); i++)
	{
		const vas::ServiceParameter &service_paramter = service_list.service_paramter(i);
		/*int r = rand();
		char n[10];
		sprintf(n,"%d",r);
		service_paramter.set_service_id(n);
		service_paramter.mutable_decode_parameter()->set_service_id(n);
		service_paramter.mutable_detect_parameter()->set_service_id(n);*/
		AddService(service_paramter);
	}
}
void VasService::StartServicesFromProtoFile(const vas::ServiceList &service_list)
{
	for (size_t i = 0; i < service_list.service_paramter_size(); i++)
	{
		vas::ServiceParameter service_paramter = service_list.service_paramter(i);
		AddService(service_paramter);
	}
}
bool VasService::SetShowRegFrame(const char* service_id, bool is_show)
{
	return ptr_detect_master_->SetShowRegFrame(service_id, is_show);
}

bool VasService::DeleteService(const char* service_id)
{
	bool ret_1 = ptr_decode_master_->DeleteVideoStream(service_id);
	bool ret_2 = ptr_detect_master_->DeleteDetecter(service_id);

	vector<shared_ptr<VasInstance>>::iterator iter;
	boost::mutex::scoped_lock lock(mutex_);
	for (iter = instance_list_.begin(); iter != instance_list_.end(); iter++)
	{
		if ((*iter)->service_id_.compare(service_id)==0)
		{
			instance_list_.erase(iter);
			break;
		}
	}
	lock.unlock();
	if (ret_1  || ret_2)
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
bool VasService::DeleteService(const string &service_id)
{
	return DeleteService(service_id.c_str());
}
void VasService::DeleteService()
{
	boost::mutex::scoped_lock lock(mutex_);
	if (instance_list_.size()>0)
	{
		vector<shared_ptr<VasInstance>>::iterator iter;
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
bool VasService::GetServiceParameter(const char* service_id, vas::ServiceParameter &service_parameter)
{
	shared_ptr<VasInstance> ptr_instance = GetServiceInstance(service_id);
	if (ptr_instance)
	{
		service_parameter.CopyFrom(ptr_instance->service_param_);
		return true;
	}
	else return false;
}
bool VasService::GetServiceStreamStatus(const char* service_id, vas::StreamStatus &stream_status)
{
	shared_ptr<VasInstance> ptr_instance = GetServiceInstance(service_id);
	if (ptr_instance)
	{
		stream_status = ptr_instance->service_param_.decode_parameter().stream_status();
		return true;
	}
	else return false;
}

bool VasService::GetServiceParameter(const string &service_id, vas::ServiceParameter &service_parameter)
{
	return GetServiceParameter(service_id.c_str(), service_parameter);
}
shared_ptr<VasInstance> VasService::GetServiceInstance(const char*service_id)
{
	boost::mutex::scoped_lock lock(mutex_);
	vector<shared_ptr<VasInstance>>::iterator iter;
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

bool VasService::SetRegResultProcessFunction(signal_reg::slot_function_type fun)
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

bool VasService::SetStreamStatusErrorFunction(signal_t::slot_function_type fun)
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

//bool VasService::SetServiceCommandReplyProcessFunction(signal_service_command_reply::slot_function_type fun)
//{
	/*if (ptr_detect_master_)
	{
		ptr_detect_master_->SetServiceCommandReplyProcessFunction(fun);
	}
	else
	{
		LOG(INFO) << "ptr_detect_master_ is not inited " << endl;
		return false;
	}
	if (ptr_decode_master_)
	{
		ptr_decode_master_->SetServiceCommandReplyProcessFunction(fun);
	}
	else
	{
		LOG(INFO) << "ptr_decode_master_ is not inited " << endl;
		return false;
	}
	service_command_reply_signal_.connect(fun);*/
	//return true;
//}
void VasService::Reset()
{
	DeleteService();
	instance_list_.clear();
	ptr_decode_master_.reset();
	ptr_detect_master_.reset();
	ptr_classifier_.reset();
	save_reg_frame_thread_.interrupt();
	
}
void VasService::GetServices(vector<string> &service_list)
{
	service_list.clear();
	for (size_t i = 0; i < instance_list_.size(); i++)
	{
		service_list.push_back(instance_list_[i]->service_id_);
	}
}
bool VasService::QuerySerive(const char *service_id, vas::ServiceStatus &status)
{
	shared_ptr<VasInstance> ptr_instance = GetServiceInstance(service_id);
	if (!ptr_instance)
	{
		status = vas::SERVICE_UNKNOWN;
		return false;
	}
	vas::ServiceParameter service_param_;
	bool ret=GetServiceParameter(service_id, service_param_);
	status = service_param_.service_status();
	return true;
}