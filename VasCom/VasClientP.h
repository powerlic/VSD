#pragma once
#include"stdafx.h"
#include<vector> 
#include"VasProto.prototxt.pb.h"


#ifdef _DLL_VasClientP
#define DLL_VasClientP_API __declspec(dllexport)
#else
#define DLL_VasClientP_API __declspec(dllimport)
#endif


typedef boost::signal<void(const vas::VasReturn&)>  signal_msg_return;
typedef boost::signal<void(const vas::EnAppState&)>  signal_link_state;

class VasClientP
{
public:
	virtual bool Start() = 0;
	virtual void Stop() = 0;

	//virtual bool CanReconnect()=0;

	virtual void SetHeartBeatMsg(const vas::VasHeartbeat &heartbeat_msg)=0;
	virtual void SetHeartBeatInterval(int interval_sec)=0;
	//virtual void StartHeartBeat()=0;
	//virtual void StopHeartBeat()=0;

	virtual void SendDetectResult(const vas::VasResult &result_msg)=0;
	virtual void SendFeedbackInfo(const vas::VasFeedback &feedback_msg) = 0;
	
	virtual vas::EnAppState GetClientState()=0;


	virtual void SetReceiveMsgProcFun(signal_msg_return::slot_function_type fun) = 0;
	virtual void SetStreamStatusProcFunction(signal_link_state::slot_function_type fun) = 0;
	
	virtual void Reset() = 0;
};

extern "C" DLL_VasClientP_API VasClientP* _stdcall CreateVasClientP(const char* com_set_proto_file);