#pragma once
#include"VasP.h"
#include"VasDecodeMaster.h"
#include"VasSsdMaster.h"


class SsdInstance
{
public:
	SsdInstance(const char* proto_file_name);
	SsdInstance(const vas::SsdServiceParameter &param);
	SsdInstance();
	~SsdInstance();

	string service_id_;
	vas::SsdServiceParameter service_param_;

	void Reset();
	void CheckParameter(const vas::SsdServiceParameter &param);
};



class SsdService : public VasSsdP
{
	struct SaveRegMat
	{
		Mat reg_frame;
		string label;
		string service_id;
	};

public:
	SsdService();
	SsdService(const char* ssd_model_proto_file_name);
	SsdService(const string& model_file, const string& trained_file, const string& mean_file, const string&mean_value, const string& label_file);
	SsdService(const vas::SsdModelPath &caffe_path);
	virtual ~SsdService();


	bool AddService(const char *service_id, const char *url, const int url_type,
		const int decode_mode, const uint32_t reg_width, const uint32_t reg_height,
		const vector<string> &reg_types);

	bool AddService(const vas::DecodeParameter &decode_param, const vas::SsdDetecterParameter &detect_param);
	bool AddService(const char* proto_file_name);
	bool AddService(const vas::SsdServiceParameter &param);

	void StartServicesFromProtoFile(const char *proto_file_name);
	void StartServicesFromProtoFile(const vas::SsdServiceList &service_list);

	bool SetShowRegFrame(const char* service_id, bool is_show);

	/*Delete a service by service_id
	*/
	bool DeleteService(const char* service_id);
	bool DeleteService(const string& service_id);

	void DeleteService();

	//bool GetRegResult(const char* service_id, Mat &show_frame, int64_t &pts);

	bool GetServiceParameter(const char*service_id, vas::SsdServiceParameter &service_parameter);
	bool GetServiceParameter(const string& service_id, vas::SsdServiceParameter &service_parameter);


	bool GetServiceStreamStatus(const char* service_id, vas::StreamStatus &stream_status);

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
	bool SetStreamStatusErrorFunction(signal_t::slot_function_type fun);


	/*
	Reset
	*/
	void Reset();

	void GetServices(vector<string> &service_list);


	bool QuerySerive(const char *service_id, vas::ServiceStatus &status);


private:
	shared_ptr<VasDecodeMaster> ptr_decode_master_;
	shared_ptr<VasSsdMaster> ptr_detect_master_;

	vector<shared_ptr<SsdInstance>> instance_list_;


	boost::mutex caffe_mutex_;
	boost::mutex mutex_;

	void InitSsdModel(const char *proto_file_name);
	void InitSsdModel(const vas::SsdModelPath &ssd_model_path);
	vas::SsdModelPath ssd_model_path_;
	shared_ptr<SsdP> ptr_ssd_;
	vector<DetectRect> Detect(const Mat &frame);



	shared_ptr<SsdInstance> GetServiceInstance(const string &service_id);
	shared_ptr<SsdInstance> GetServiceInstance(const char*service_id);

	void SaveRegFrameFunction();
	concurrent_queue<SaveRegMat> save_reg_frame_queue_;
	boost::thread save_reg_frame_thread_;

	bool CheckIdUnique(const char* service_id);
	bool CheckIdUnique(const string &service_id);

	void SetUpSaveFile();

	//signal_service_command_reply service_command_reply_signal_;

	void SetRegTypes(const vector<string> &reg_types, vas::SsdServiceParameter &service_parameter);

};



