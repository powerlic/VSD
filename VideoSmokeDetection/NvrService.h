#pragma once
#include"NvrMaster.h"
#include"VasDetectMaster.h"
#include"VasP.h"

class NvrServiceInstance
{
public:
	NvrServiceInstance(const char* proto_file_name);
	NvrServiceInstance(const vas::NvrServiceParameter &param);
	NvrServiceInstance();
	~NvrServiceInstance();

	string service_id_;
	vas::NvrServiceParameter service_param_;

	void Reset();
	void CheckParameter(const vas::NvrServiceParameter &param);
};


class NvrService :public VasNvrP
{
public:
	NvrService();
	NvrService(const char* caffe_model_proto_file_name,const char*nvr_access_proto_file);
	//NvrService(const string& model_file, const string& trained_file, const string& mean_file, const string& label_file, bool caffe_use_gpu,
	//		   const string&nvr_ip,const string& nvr_user,const string &nvr_pwd, const int64_t port);
	NvrService(const vas::CaffeModelPath &caffe_path, const vas::NvrParameter nvr_param);
	virtual ~NvrService();



	/*---------------------Interface--------------------------*/
	/*Add a service immedately
	*@param service_id[in]: unique index for a service
	*@param cam_id: camera id in nvr [0-16]
	*@warning: GPU decode mode is not support!!!!
	*@param reg_width[in]: assigned width for reg_frame width
	*@param reg_height[in]: assigned height for reg_frame height
	*@param reg_types[in]: ret_type list
	*@param sensitivity [0,1],  sensitivity for smoke detection, 1 means the threshold is very low and is easy to raise fault smoke alarm!!
	*@return  0, if init success, -1, if init parameter failed
	*@warning: return 0 not means start a process successful! When a service starts successful or failed, the Callback function will notice it!
	*/
	bool AddService(const char *service_id, const uint32_t channel_no,const uint32_t reg_width, const uint32_t reg_height,
		            const vector<string> &reg_types, const float smoke_sensitivity = 0.2, const float fire_sensitivity = 0.2);


	bool AddService(const vas::NvrServiceParameter &param);
	bool AddService(const char* proto_file_name);
	bool AddService(const vas::NvrChannel &nvr_channel_info, const vas::DetectParameter &detect_param);

	void StartServicesFromProtoFile(const char *proto_file_name);
	void StartServicesFromProtoFile(const vas::NvrServiceList &nvr_service_list);

	/*Delete a service by service_id
	*/
	bool DeleteService(const char* service_id);
	bool DeleteService(const string& service_id);

	void DeleteService();


	/*void PauseService(const char* service_id);
	void ResumeService(const char* service_id);*/

	bool GetRegResult(const char* service_id, Mat &show_frame, int64_t &pts);

	bool GetServiceParameter(const char*service_id, vas::NvrServiceParameter &service_parameter);
	bool GetServiceParameter(const string& service_id, vas::NvrServiceParameter &service_parameter);


	/*
	Set Reg result process function
	Fun should be as void(const char* service_id, const vector<RectF> &reg_rects, const vector<float> &scores, const vector<string> &labels)
	and use boost::bind(&SomeClass::fun, this, _1,_2,_3,_4) to set.
	It can not be a complicated process function, otherwise it will block the reg thread! In most cases, reg results can be insert into your own message list to notify other app
	*/
	bool SetRegResultProcessFunction(signal_reg::slot_function_type fun);


	/* Set a stream status process function for decode process
	Fun should be as void(const char *, vas::StreamStatus)
	and use boost::bind(&SomeClass::fun, this, _1,_2) to set
	*/
	bool SetStreamStatusErrorFunction(cam_signal_t::slot_function_type fun);



	/*
	Set a service command reply function for decode process
	Fun should be as void(const char* service_id, const vas::ServiceCommandReply &reply, const string &info)
	and use boost::bind(&SomeClass::fun, this, _1,_2,_3) to set
	*/
	bool SetServiceCommandReplyProcessFunction(signal_service_command_reply::slot_function_type fun);


	/*
	Reset
	*/
	void Reset();

private:
	shared_ptr<NvrMaster> ptr_nvr_master_;
	shared_ptr<VasDetectMaster> ptr_detect_master_;

	boost::mutex mutex_;
	vector<shared_ptr<NvrServiceInstance>> instance_list_;

	boost::mutex caffe_mutex_;
	void InitCaffeModel(const char *proto_file_name);
	void InitCaffeModel(const vas::CaffeModelPath &caffe_path);

	vas::CaffeModelPath caffe_model_path_;
	shared_ptr<Classifier> ptr_classifier_;
	vector<Prediction> Predict(const Mat &image_mat, const int &N = 5);

	shared_ptr<NvrServiceInstance> GetServiceInstance(const string &service_id);
	shared_ptr<NvrServiceInstance> GetServiceInstance(const char*service_id);

	bool CheckIdUnique(const char* service_id);
	bool CheckIdUnique(const string &service_id);

	signal_service_command_reply service_command_reply_signal_;

	void SetRegTypes(const vector<string> &reg_types, vas::NvrServiceParameter &service_parameter);

};

