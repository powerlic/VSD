#include "stdafx.h"
#include "VasClient.h"


VasClient::VasClient(const char*proto_file)
{
	CHECK(vas::ReadProtoFromTextFile(proto_file, &client_param_));
	CheckParam(client_param_);
	ptr_com_ = shared_ptr<VasCom>(new VasCom(client_param_.com_param()));
	ptr_com_->SetReceiveProcessFun(boost::bind(&VasClient::ReceiveMsg, this, _1));
}

VasClient::~VasClient()
{
	client_param_.Clear();
}

void VasClient::CheckParam(const vas::VasClientParameter& client_param)
{

}
bool VasClient::Start()
{
	return ptr_com_->Start();
}
void VasClient::Stop()
{
	ptr_com_->Stop();
}

//bool VasClient::CanReconnect()
//{
//	return ptr_com_->CanReconnect();
//}
void VasClient::SetHeartBeatMsg(const vas::VasHeartbeat &heartbeat_msg)
{
	string content_msg = VasMsgTrans::Get()->TransMsg(&heartbeat_msg);
	BYTE msg[1024 * 4];
	int length;
	VasMsgTrans::Get()->TransSendMsg(&heartbeat_msg, FRAME_TYPE_HEARTBEAT, msg, length);
	ptr_com_->SetHeartBeatMessage((char*)msg,length);
}
void VasClient::SetHeartBeatInterval(int interval_sec)
{
	ptr_com_->SetHeartBeatInterval(interval_sec);
}
//void VasClient::StartHeartBeat()
//{	
//	if (GetClientState()==vas::ST_STARTED)
//	{
//		ptr_com_->StartHeartBeat();
//	}
//}
//void VasClient::StopHeartBeat()
//{
//	ptr_com_->StopHeartBeat();
//
//}
void VasClient::SendDetectResult(const vas::VasResult &result_msg)
{
	BYTE msg[1024 * 4];
	int length;
	VasMsgTrans::Get()->TransSendMsg(&result_msg, FRAME_TYPE_REG_RESULT, msg, length);
	ptr_com_->Send(msg,length);
}
void VasClient::SendFeedbackInfo(const vas::VasFeedback &feedback_msg)
{
	BYTE msg[1024 * 4];
	int length;
	VasMsgTrans::Get()->TransSendMsg(&feedback_msg, FRAME_TYPE_FEEDBACK, msg,length);
	ptr_com_->Send(msg, length);
}
vas::EnAppState VasClient::GetClientState()
{
	return ptr_com_->GetAppState();
}
void VasClient::SetReceiveMsgProcFun(signal_msg_return::slot_function_type fun)
{
	msg_ret_signal_.connect(fun);

}
void VasClient::SetStreamStatusProcFunction(signal_link_state::slot_function_type fun)
{
	ptr_com_->SetLinkStateProcessFun(fun);
}
void VasClient::Reset()
{
	ptr_com_.reset();
}
void VasClient::ReceiveMsg(const string &msg)
{
	vas::VasReturn ret;
	VasMsgTrans::Get()->TransReceiveMsg(msg, ret);
	msg_ret_signal_(ret);
}