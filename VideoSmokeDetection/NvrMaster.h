#pragma once
#include"stdafx.h"
#include"VasVideoSource.h"
#include"vasIO.h"
#include"VasProto.prototxt.pb.h"
#include"LicImageOperation.h"
#include "HCNetSDK.h"
#include "plaympeg4.h"

#pragma comment(lib, "HCNetSDK.lib")
#pragma comment(lib, "HCCore.lib")
#pragma comment(lib, "PlayCtrl.lib")

using namespace std;
using namespace cv;

#define MAX_CAM_NUM 32


class NvrVideoStream
{
public:
	NvrVideoStream(const char *proto_file_name);
	NvrVideoStream(const vas::NvrChannel &param);
	~NvrVideoStream();


	string service_id_;
	uint16_t connect_times_ = 0;
	vas::NvrChannel channel_param_;


	concurrent_queue<DecodedFrame> frame_queue_;
	uint64_t last_ms_ = 0;
	uint64_t decode_count_ = 0;

	DWORD cam_id_=-1;
	NET_DVR_PREVIEWINFO preview_info_;
	LONG play_handle_;

	void CheckParameter(const vas::NvrChannel &param);

	bool is_send_start_failed_msg_ = false;
	bool is_display_ = false;
};


class NvrMaster
{
public:
	NvrMaster(const char*nvr_param_proto_file);
	NvrMaster(const string &ip, const string &user_name,const string &password, const int64_t &port);
	NvrMaster(const vas::NvrParameter&param);
	virtual ~NvrMaster();

	bool AddNvrVideoStream(const char* service_id,const int32_t channel_no, const int convert_mode, const int dst_width, const int dst_height);
	bool AddNvrVideoStream(const char*proto_file_name);
	bool AddNvrVideoStream(const vas::NvrChannel &channel_param);

	bool DeleteNvrVideoStream(const char* service_id);
	bool DeleteNvrVideoStream(const string &service_id);

	/*Delete all video streams
	*@return true if get the frame, else return false
	*/
	//void DeleteVideoStream();

	/*bool GetDecodedFrame(const string &service_id, Mat &frame, int64_t &pts, int64_t &No);*/
	bool GetDecodedFrame(const char *service_id, Mat &frame, int64_t &pts, int64_t &No);

	/*Pause all decode threads
	*/
	//void Pause();
	/*Pause a decode thread
	*/
	//bool Pause(const char* service_id);
	//bool Pause(const string &service_id);


	/*Resume all decode threads after pause
	*/
	//void Resume();

	/*Resume a decode thread after pause
	*/
	//bool Resume(const char* service_id);
	//bool Resume(const string &service_id);

	/*
	Service Command Reply Proccess Function
	*/
	void SetServiceCommandReplyProcessFunction(signal_service_command_reply::slot_function_type fun);


	/*Register a stream status process function for decode process
	Fun should be as void(const char *, vas::StreamStatus)
	and use boost::bind(&SomeClass::fun, this, _1,_2) to set
	*/
	void SetStreamStatusErrorFunction(cam_signal_t::slot_function_type fun);


private:
	signal_service_command_reply service_command_reply_signal_;
	cam_signal_t sig_;
	connection_t sig_connect_;
	connection_t service_command_reply_connetct_;

	bool login_ = false;
	LONG nvr_user_id_;
	boost::thread frame_process_thread_;
	void FrameProcessFun();
	void Login();
	void Logout();


	boost::thread status_check_thread_;
	void CheckFun();
	int check_interval_time_;

	void CheckNvrParam(const vas::NvrParameter &param);
	

	NET_DVR_DEVICEINFO_V30 nvr_device_info_;
	vas::NvrParameter nvr_param_;
	vector<shared_ptr<NvrVideoStream>> nvr_stream_list_;
	bool CheckIdUnique(const string &service_id);

	inline shared_ptr<NvrVideoStream> GetNvrVideoStreamPtr(const char* service_id);
	inline shared_ptr<NvrVideoStream> GetNvrVideoStreamPtr(const string&service_id);

	char err_buf_[1024];
	boost::mutex mutex_;
};

