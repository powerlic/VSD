#pragma once
#include"VasClientP.h"
#include"VasCom.h"
#include"VasIO.h"
#include"VasMsgTrans.h"



class VasClient:public VasClientP
{
public:
	VasClient(const char *proto_file);
	virtual ~VasClient();


	bool Start();
	void Stop();

	//bool CanReconnect();

	void SetHeartBeatMsg(const vas::VasHeartbeat &heartbeat_msg);
	void SetHeartBeatInterval(int interval_sec);
	//void StartHeartBeat();
	//void StopHeartBeat();

	void SendFeedbackInfo(const vas::VasFeedback &feedback_msg);

	void SendDetectResult(const vas::VasResult &result_msg);
	vas::EnAppState GetClientState();

	void SetReceiveMsgProcFun(signal_msg_return::slot_function_type fun);
	void SetStreamStatusProcFunction(signal_link_state::slot_function_type fun);

	void Reset();

private:
	vas::VasClientParameter client_param_;
	void CheckParam(const vas::VasClientParameter& client_param);

	shared_ptr<VasCom> ptr_com_;
	string heart_msg_;

	signal_msg_return msg_ret_signal_;

	void ReceiveMsg(const string &msg);
};



