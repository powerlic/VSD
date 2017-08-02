#pragma once
#include"VasP.h"
#include"VasClientP.h"
#include"VasIO.h"
#include"cvpreload.h"
#include "concurrent_queue.h"

class VasUnit
{
public:
	VasUnit(const char* caffe_model_proto_file, const char *service_proto_file, const char*com_set_proto_file);
	virtual ~VasUnit();
	
	void Start();

private:
	shared_ptr<VasP> ptr_vas_;
	shared_ptr<VasClientP> ptr_client_;
	
	vas::VasClientParameter client_param_;
	vas::ServiceList service_list_;

	string GetServiceStreamAddress(const char *service_id);

	void RegResultProcFun(const char* service_id,Mat&reg_frame, const vector<CvRect> &reg_rects,
							 const uint32_t dst_width, const uint32_t dst_height,
		                      const vector<float> &scores, const vector<string> &labels);

	void ServiceCommandReplyProcFun(const char* service_id, const vas::ServiceCommandReply &reply, const string &info);


	void StreamErrProcFun(const char *service_id, vas::StreamStatus status);


	void ClientLinkStatusProcFun(const vas::EnAppState &state);

	void ReceiveMsgProcFun(const vas::VasReturn &ret);

	int64 GetTime();
	void SetStartHeartbeatMsg();


	//boost::thread client_supervisor_thread_;
	//void ClientSupervisorFun();

};

