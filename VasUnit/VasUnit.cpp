// VasUnit.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "VasUnit.h"


VasUnit::VasUnit(const char* caffe_model_proto_file, const char *service_proto_file, const char*com_set_proto_file)
{
	CHECK(vas::ReadProtoFromTextFile(service_proto_file, &service_list_));
	CHECK(vas::ReadProtoFromTextFile(com_set_proto_file, &client_param_));
	ptr_client_ = shared_ptr<VasClientP>(CreateVasClientP(com_set_proto_file));
	ptr_vas_ = shared_ptr<VasP>(CreateVasP(caffe_model_proto_file));


	CHECK(ptr_client_) << " Client Init failed " << endl;
	CHECK(ptr_vas_) << " Vas Init failed " << endl;

	ptr_client_->SetReceiveMsgProcFun(boost::bind(&VasUnit::ReceiveMsgProcFun, this, _1));
	ptr_client_->SetStreamStatusProcFunction(boost::bind(&VasUnit::ClientLinkStatusProcFun, this, _1));

	ptr_vas_->SetRegResultProcessFunction(boost::bind(&VasUnit::RegResultProcFun, this, _1, _2, _3, _4, _5, _6,_7));
//	ptr_vas_->SetServiceCommandReplyProcessFunction(boost::bind(&VasUnit::ServiceCommandReplyProcFun, this, _1, _2, _3));
	ptr_vas_->SetStreamStatusErrorFunction(boost::bind(&VasUnit::StreamErrProcFun, this, _1, _2));

}


VasUnit::~VasUnit()
{
	ptr_vas_->Reset();
	ptr_client_->Reset();
	service_list_.Clear();
}
void VasUnit::SetStartHeartbeatMsg()
{
	if (ptr_vas_)
	{
		const vas::VasComputeNode &node_param = client_param_.node_param();
		string service_id_str = "";
		if (service_list_.service_paramter().size()>0)
		{
			service_id_str = service_list_.service_paramter(0).service_id();
			for (size_t i = 1; i < service_list_.service_paramter().size(); i++)
			{
				service_id_str += "," + service_list_.service_paramter(i).service_id();
			}
		}
		vas::VasHeartbeat heratbeat_msg;
		heratbeat_msg.set_psi_name("vasHeartbeat");
		heratbeat_msg.mutable_params()->set_service_list(service_id_str);
		heratbeat_msg.mutable_params()->set_node_id(node_param.node_id());
		heratbeat_msg.mutable_params()->set_service_list(service_id_str);
		heratbeat_msg.mutable_params()->set_timestamp(GetTime());
		ptr_client_->SetHeartBeatMsg(heratbeat_msg);
	}
}

void VasUnit::Start()
{
	ptr_vas_->StartServicesFromProtoFile(service_list_);
	SetStartHeartbeatMsg();
	ptr_client_->Start();
}
string VasUnit::GetServiceStreamAddress(const char *service_id)
{
	for (size_t i = 0; i < service_list_.service_paramter().size(); i++)
	{

		if (service_list_.service_paramter(i).service_id().compare(service_id)==0)
		{
			return service_list_.service_paramter(i).decode_parameter().url();
		}
	}
	return string("No match rtsp");
}


void VasUnit::RegResultProcFun(const char* service_id, Mat&reg_frame, const vector<CvRect> &reg_rects,
	const uint32_t dst_width, const uint32_t dst_height,
	const vector<float> &scores, const vector<string> &labels)
{
	vas::VasResult reg_msg;
	reg_msg.set_psi_name("vasResult");

	const vas::VasComputeNode &node_param = client_param_.node_param();

	reg_msg.mutable_params()->set_node_id(node_param.node_id());
	reg_msg.mutable_params()->set_service_id(service_id);

	for (size_t i = 0; i < reg_rects.size(); i++)
	{
		vas::RegRes *reg_res;
		reg_res = reg_msg.mutable_params()->mutable_result_rects()->Add();
		reg_res->set_reg_type(labels[i]);
		reg_res->set_score(scores[i]);
		reg_res->set_x(reg_rects[i].x);
		reg_res->set_x(reg_rects[i].y);
		reg_res->set_width(reg_rects[i].width);
		reg_res->set_height(reg_rects[i].height);
	}
	reg_msg.mutable_params()->set_timestamp(GetTime());

	ptr_client_->SendDetectResult(reg_msg);
}
void VasUnit::ServiceCommandReplyProcFun(const char* service_id, const vas::ServiceCommandReply &reply, const string &info)
{
	const vas::VasComputeNode &node_param = client_param_.node_param();
	vas::VasFeedback feed_back_msg;
	feed_back_msg.set_psi_name("vasFeedback");
	feed_back_msg.mutable_params()->set_address_type("rtsp");
	feed_back_msg.mutable_params()->set_address(GetServiceStreamAddress(service_id));
	feed_back_msg.mutable_params()->set_service_id(service_id);
	feed_back_msg.mutable_params()->set_node_id(node_param.node_id());
	feed_back_msg.mutable_params()->set_device_channel("UNKNOWN");
	feed_back_msg.mutable_params()->set_device_id(node_param.node_name());
	feed_back_msg.mutable_params()->set_timestamp(GetTime());
	feed_back_msg.mutable_params()->set_feedback_code(1);
	feed_back_msg.mutable_params()->set_feedback_info("service started");

	/*switch (reply)
	{	
		case vas::SERVICE_START_SUCCESS:
			feed_back_msg.mutable_params()->set_feedback_code(1);
			feed_back_msg.mutable_params()->set_feedback_info("service started");
			break;
		case vas::SERVICE_STOP_SUCESS:
			feed_back_msg.mutable_params()->set_feedback_code(2);
			feed_back_msg.mutable_params()->set_feedback_info("service stop");
			break;
		default:
			break;
	}*/
	//ptr_client_->SendFeedbackInfo(feed_back_msg);

}
void VasUnit::StreamErrProcFun(const char *service_id, vas::StreamStatus status)
{
	const vas::VasComputeNode &node_param = client_param_.node_param();

	vas::VasFeedback feed_back_msg;
	feed_back_msg.set_psi_name("vasFeedback");
	feed_back_msg.mutable_params()->set_address_type("rtsp");
	feed_back_msg.mutable_params()->set_address(service_id);
	feed_back_msg.mutable_params()->set_device_channel("No");
	feed_back_msg.mutable_params()->set_device_id("CAM");
	feed_back_msg.mutable_params()->set_timestamp(GetTime());

	switch (status)
	{
		case vas::STREAM_UNKNOWN:
			break;
		case vas::STREAM_CONNECTING:
			break;
		case vas::STREAM_NORMAL:
			//feed_back_msg.mutable_params()->set_feedback_code(1);
			//feed_back_msg.mutable_params()->set_feedback_info("service started");
			break;
		case vas::STREAM_FINISH:
			break;
		case vas::STREAM_STOP:
			break;
		case vas::STREAM_PAUSE:
			break;
		case vas::STREAM_NETWORK_FAULT:
			feed_back_msg.mutable_params()->set_feedback_code(7);
			feed_back_msg.mutable_params()->set_feedback_info("stream error");
			ptr_client_->SendFeedbackInfo(feed_back_msg);
			break;
		case vas::FILE_FAULT:
			break;
		case vas::STREAM_CPU_DECODE_FAULT:
			break;
		case vas::STREAM_GPU_DECODE_FAULT:
			break;
		default:
			break;
	}
	

}

void VasUnit::ClientLinkStatusProcFun(const vas::EnAppState &state)
{
	switch (state)
	{
		case vas::ST_STARTED:
			break;
		case  vas::ST_STOPPED:
			//LOG(ERROR) << "Disconnect to the Server " << endl;
			ptr_client_->Stop();
			/*LOG(INFO) << "Reconnect to the Server:" << client_param_.com_param().ip() << " port:" << client_param_.com_param().port() << endl;
			if (!client_supervisor_thread_.joinable())
			{
				client_supervisor_thread_ = boost::thread(boost::bind(&VasUnit::ClientSupervisorFun, this));
			}*/
			break;
		default:
			break;
	}
}
//void VasUnit::ClientSupervisorFun()
//{
//	while (ptr_client_->GetClientState()==vas::ST_STOPPED)
//	{
//		while (!ptr_client_->CanReconnect())
//		{
//			LOG(INFO) << "Wait for client clear"<< endl;
//			boost::thread::sleep(boost::get_system_time() + boost::posix_time::seconds(5));
//		}
//		while (!ptr_client_->Start())
//		{
//			boost::thread::sleep(boost::get_system_time() + boost::posix_time::seconds(5));
//		}
//	}
//	ptr_client_->StartHeartBeat();
//}

void VasUnit::ReceiveMsgProcFun(const vas::VasReturn &ret)
{
	//LOG(INFO) << ret.ret_msg() << endl;
}
int64 VasUnit::GetTime()
{
	SYSTEMTIME st;
	GetLocalTime(&st);
	tm temptm = { st.wSecond, st.wMinute, st.wHour, st.wDay, st.wMonth - 1, st.wYear - 1900, st.wDayOfWeek, 0, 0 };
	time_t ts = mktime(&temptm) * 1000 + st.wMilliseconds;
	return ts;
}