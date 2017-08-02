#pragma once

#include<vector> 
#include"VasProto.prototxt.pb.h"
#include"cvpreload.h"

using namespace std;
using namespace cv;


#ifdef _DLL_VASP
#define DLL_VASP_API __declspec(dllexport)
#else
#define DLL_VASP_API __declspec(dllimport)
#endif


typedef boost::signal<void(const char* service_id, Mat &reg_frame, const vector<CvRect> &reg_rects, const uint32_t reg_width, const uint32_t reg_height, const vector<float> &scores, const vector<string> &labels)>  signal_reg;
//typedef boost::signal<void(const char* service_id, const vas::ServiceCommandReply &reply, const string &info)>  signal_service_command_reply;
typedef boost::signal<void(const char* service_id, vas::StreamStatus status)>  signal_t;
typedef boost::signal<void(const char* service_id, vas::CamStatus status)>  cam_signal_t;


class VasP
{
public:

	/*Add a service immedately
	*@param service_id[in]: unique index for a service
	*@param url[in]: the rstp addr or video file path
	*@param url_type[in]: 0, rtsp stream; 1, video file path
	*@param decode_mode[in]: 0, cpu; 1, gpu
	*@warning: GPU decode mode is not support!!!!
	*@param reg_width[in]: assigned width for reg_frame width
	*@param reg_height[in]: assigned height for reg_frame height
	*@param reg_types[in]: ret_type list
	*@param sensitivity [0,1],  sensitivity for smoke or fire detection, set 1 means the threshold is very low and is easy to raise fault smoke alarm!!
	*@return  0, if init success, -1, if init parameter failed
	*@warning: return 0 not means start a process successful! When a service starts successful or failed, the Callback function will notice it!
	*@warning: should set the default service parameter in the proto dir
	*/
	virtual bool AddService(const char *service_id, const char *url, const int url_type,
		const int decode_mode, const uint32_t reg_width, const uint32_t reg_height,
		const vector<string> &reg_types, const float smoke_sensitivity = 0.2, const float fire_sensitivity=0.2) = 0;
	/*virtual bool AddService(const string &service_id, const char *url, const int url_type,
		const int decode_mode, const uint32_t reg_width, const uint32_t reg_height,
		const vector<string> &reg_types)=0;*/

	/*@Brief: Refer to the service.prototxt to start the service
	*/
	virtual bool AddService(const vas::DecodeParameter &decode_param, const vas::DetectParameter &detect_param)=0;
	virtual bool AddService(const char* proto_file_name)=0;
	virtual bool AddService(const vas::ServiceParameter &param)=0;

	/*@Brief: Start multi services from the proto file
	*/
	virtual void StartServicesFromProtoFile(const char *proto_file_name)=0;
	virtual void StartServicesFromProtoFile(const vas::ServiceList &service_list)=0;

	/*Delete a service by service_id
	*/
	virtual bool DeleteService(const char* service_id)=0;
	virtual bool DeleteService(const string& service_id)=0;

	/*@Brief: Delete all services
	*/
	virtual void DeleteService()=0;

	/*Get reg result frame
	*/
	virtual bool GetRegResult(const char* service_id, Mat &show_frame, int64_t &pts) = 0;


	/*Get service parameter*/
	virtual bool GetServiceParameter(const char* service_id, vas::ServiceParameter &service_parameter)=0;
	virtual bool GetServiceParameter(const string &service_id, vas::ServiceParameter &service_parameter)=0;

	/*
	 Get stream status
	*/
	virtual bool GetServiceStreamStatus(const char* service_id, vas::StreamStatus &stream_status)=0;

	virtual bool SetShowRegFrame(const char* service_id, bool is_show)=0;

	/*
	Set Reg result process function
	Fun should be as void(const char* service_id, const vector<RectF> &reg_rects, const vector<float> &scores, const vector<string> &labels)
	and use boost::bind(&SomeClass::fun, this, _1,_2,_3,_4) to set.
	It can not be a complicated process function, otherwise it will block the reg thread! In most cases, reg results can be insert into your own message list to notify other app
	*/
	virtual bool SetRegResultProcessFunction(signal_reg::slot_function_type fun)=0;


	/* Set a stream status process function for decode process
	Fun should be as void(const char *, vas::StreamStatus)
	and use boost::bind(&SomeClass::fun, this, _1,_2) to set
	*/
	virtual bool SetStreamStatusErrorFunction(signal_t::slot_function_type fun) = 0;


	/*
	Set a service command reply function for decode process
	Fun should be as void(const char* service_id, const vas::ServiceCommandReply &reply, const string &info)
	and use boost::bind(&SomeClass::fun, this, _1,_2,_3) to set
	warning: despareted
	*/
	//virtual bool SetServiceCommandReplyProcessFunction(signal_service_command_reply::slot_function_type fun) = 0;

	/*
	Reset
	*/
	virtual void Reset()=0;


	virtual void GetServices(vector<string> &service_list)=0;

	virtual bool QuerySerive(const char *service_id, vas::ServiceStatus &status)=0;
};

class VasSsdP
{
public:
	/*Add a service immedately
	*@param service_id[in]: unique index for a service
	*@param url[in]: the rstp addr or video file path
	*@param url_type[in]: 0, rtsp stream; 1, video file path
	*@param decode_mode[in]: 0, cpu; 1, gpu
	*@warning: GPU decode mode is not support!!!!
	*@param reg_width[in]: assigned width for reg_frame width
	*@param reg_height[in]: assigned height for reg_frame height
	*@param reg_types[in]: ret_type list
	*@return  0, if init success, -1, if init parameter failed
	*@warning: return 0 not means start a process successful! When a service starts successful or failed, the Callback function will notice it!
	*@warning: should set the default service parameter in the proto dir
	*/
	virtual bool AddService(const char *service_id, const char *url, const int url_type,
		                    const int decode_mode, const uint32_t reg_width, const uint32_t reg_height,
		                    const vector<string> &reg_types) = 0;
	/*virtual bool AddService(const string &service_id, const char *url, const int url_type,
	const int decode_mode, const uint32_t reg_width, const uint32_t reg_height,
	const vector<string> &reg_types)=0;*/

	/*@Brief: Refer to the service.prototxt to start the service
	*/
	virtual bool AddService(const vas::DecodeParameter &decode_param, const vas::SsdDetecterParameter &detect_param) = 0;
	virtual bool AddService(const char* proto_file_name) = 0;
	virtual bool AddService(const vas::SsdServiceParameter &param) = 0;

	/*@Brief: Start multi services from the proto file
	*/
	virtual void StartServicesFromProtoFile(const char *proto_file_name) = 0;
	virtual void StartServicesFromProtoFile(const vas::SsdServiceList &service_list) = 0;

	/*Delete a service by service_id
	*/
	virtual bool DeleteService(const char* service_id) = 0;
	virtual bool DeleteService(const string& service_id) = 0;

	/*@Brief: Delete all services
	*/
	virtual void DeleteService() = 0;

	/*Get reg result frame
	*/
	//virtual bool GetRegResult(const char* service_id, Mat &show_frame, int64_t &pts) = 0;


	/*Get service parameter*/
	virtual bool GetServiceParameter(const char* service_id, vas::SsdServiceParameter &service_parameter) = 0;
	virtual bool GetServiceParameter(const string &service_id, vas::SsdServiceParameter &service_parameter) = 0;

	/*
	Get stream status
	*/
	virtual bool GetServiceStreamStatus(const char* service_id, vas::StreamStatus &stream_status) = 0;

	virtual bool SetShowRegFrame(const char* service_id, bool is_show) = 0;

	/*
	Set Reg result process function
	Fun should be as void(const char* service_id, const vector<RectF> &reg_rects, const vector<float> &scores, const vector<string> &labels)
	and use boost::bind(&SomeClass::fun, this, _1,_2,_3,_4) to set.
	It can not be a complicated process function, otherwise it will block the reg thread! In most cases, reg results can be insert into your own message list to notify other app
	*/
	virtual bool SetRegResultProcessFunction(signal_reg::slot_function_type fun) = 0;


	/* Set a stream status process function for decode process
	Fun should be as void(const char *, vas::StreamStatus)
	and use boost::bind(&SomeClass::fun, this, _1,_2) to set
	*/
	virtual bool SetStreamStatusErrorFunction(signal_t::slot_function_type fun) = 0;


	/*
	Set a service command reply function for decode process
	Fun should be as void(const char* service_id, const vas::ServiceCommandReply &reply, const string &info)
	and use boost::bind(&SomeClass::fun, this, _1,_2,_3) to set
	warning: despareted
	*/
	//virtual bool SetServiceCommandReplyProcessFunction(signal_service_command_reply::slot_function_type fun) = 0;

	/*
	Reset
	*/
	virtual void Reset() = 0;
	virtual void GetServices(vector<string> &service_list) = 0;

	virtual bool QuerySerive(const char *service_id, vas::ServiceStatus &status) = 0;
};



class VasNvrP
{
public:

	/*Add a service immedately
	*@param service_id[in]: unique index for a service
	*@param url[in]: the rstp addr or video file path
	*@param url_type[in]: 0, rtsp stream; 1, video file path
	*@param decode_mode[in]: 0, cpu; 1, gpu
	*@warning: GPU decode mode is not support!!!!
	*@param reg_width[in]: assigned width for reg_frame width
	*@param reg_height[in]: assigned height for reg_frame height
	*@param reg_types[in]: ret_type list
	*@param sensitivity [0,1],  sensitivity for smoke or fire detection, set 1 means the threshold is very low and is easy to raise fault smoke alarm!!
	*@return  0, if init success, -1, if init parameter failed
	*@warning: return 0 not means start a process successful! When a service starts successful or failed, the Callback function will notice it!
	*@warning: should set the default service parameter in the proto dir
	*/
	virtual bool AddService(const char *service_id, const uint32_t channel_no, const uint32_t reg_width, const uint32_t reg_height,
		const vector<string> &reg_types, const float smoke_sensitivity = 0.2, const float fire_sensitivity = 0.2) = 0;
	/*virtual bool AddService(const string &service_id, const char *url, const int url_type,
	const int decode_mode, const uint32_t reg_width, const uint32_t reg_height,
	const vector<string> &reg_types)=0;*/

	/*@Brief: Refer to the service.prototxt to start the service
	*/
	virtual bool AddService(const vas::NvrChannel &decode_param, const vas::DetectParameter &detect_param) = 0;
	virtual bool AddService(const char* proto_file_name) = 0;
	virtual bool AddService(const vas::NvrServiceParameter &param) = 0;

	/*@Brief: Start multi services from the proto file
	*/
	virtual void StartServicesFromProtoFile(const char *proto_file_name) = 0;
	virtual void StartServicesFromProtoFile(const vas::NvrServiceList &service_list) = 0;

	/*Delete a service by service_id
	*/
	virtual bool DeleteService(const char* service_id) = 0;
	virtual bool DeleteService(const string& service_id) = 0;

	/*@Brief: Delete all services
	*/
	virtual void DeleteService() = 0;

	/*Get reg result frame
	*/
	virtual bool GetRegResult(const char* service_id, Mat &show_frame, int64_t &pts) = 0;


	/*Get service parameter*/
	virtual bool GetServiceParameter(const char* service_id, vas::NvrServiceParameter &service_parameter) = 0;
	virtual bool GetServiceParameter(const string &service_id, vas::NvrServiceParameter &service_parameter) = 0;

	/*
	Set Reg result process function
	Fun should be as void(const char* service_id, const vector<RectF> &reg_rects, const vector<float> &scores, const vector<string> &labels)
	and use boost::bind(&SomeClass::fun, this, _1,_2,_3,_4) to set.
	It can not be a complicated process function, otherwise it will block the reg thread! In most cases, reg results can be insert into your own message list to notify other app
	*/
	virtual bool SetRegResultProcessFunction(signal_reg::slot_function_type fun) = 0;


	/* Set a stream status process function for decode process
	Fun should be as void(const char *, vas::StreamStatus)
	and use boost::bind(&SomeClass::fun, this, _1,_2) to set
	*/
	virtual bool SetStreamStatusErrorFunction(cam_signal_t::slot_function_type fun) = 0;


	/*
	Set a service command reply function for decode process
	Fun should be as void(const char* service_id, const vas::ServiceCommandReply &reply, const string &info)
	and use boost::bind(&SomeClass::fun, this, _1,_2,_3) to set
	*/
	//virtual bool SetServiceCommandReplyProcessFunction(signal_service_command_reply::slot_function_type fun) = 0;

	/*
	Reset
	*/
	virtual void Reset() = 0;
};




extern "C" DLL_VASP_API VasP* _stdcall CreateVasP(const char* caffe_model_proto_file_name);
extern "C" DLL_VASP_API VasNvrP* _stdcall CreateVasNvrP(const char* caffe_model_proto_file_name, const char*nvr_access_proto_file);
extern "C" DLL_VASP_API VasSsdP* _stdcall CreateVasSsdP(const char* ssd_model_proto_file_name);