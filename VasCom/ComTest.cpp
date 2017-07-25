#include "stdafx.h"
#include "ComTest.h"

void TestHeartbeat()
{
	shared_ptr<VasCom> ptr_com = shared_ptr<VasCom>(new VasCom("../proto/com.prototxt"));
	ptr_com->Start();
	vas::VasHeartbeat heartbeat_msg;

	heartbeat_msg.set_psi_name("vasHeartbeat");
	heartbeat_msg.mutable_params()->set_app_key("xdfadsfasdfasdf");
	heartbeat_msg.mutable_params()->set_node_id(1);
	heartbeat_msg.mutable_params()->set_service_list("service_1,service_2");
	heartbeat_msg.mutable_params()->set_timestamp(time(NULL));
	string send_msg = VasMsgTrans::Get()->TransSendMsg(&heartbeat_msg,FRAME_TYPE_HEARTBEAT);

	ptr_com->Send(send_msg);

	while (true)
	{
		boost::thread::sleep(boost::get_system_time() + boost::posix_time::seconds(500));
	}

}
void TestSendRegResult()
{
	shared_ptr<VasCom> ptr_com = shared_ptr<VasCom>(new VasCom("../proto/com.prototxt"));
	ptr_com->Start();
	vas::VasResult reg_msg;

	reg_msg.set_psi_name("vasResult");
	//reg_msg.mutable_params()->set_app_key("xdfadsfasdfasdf");
	reg_msg.mutable_params()->set_node_id(1);
	reg_msg.mutable_params()->set_service_id("service_1");

	vas::RegRes *reg_res_1;
	reg_res_1=reg_msg.mutable_params()->mutable_result_rects()->Add();
	reg_res_1->set_reg_type("smoke");
	reg_res_1->set_score("0.5");
	reg_res_1->set_x(0); 
	reg_res_1->set_x(0);
	reg_res_1->set_width(10);
	reg_res_1->set_height(100);

	reg_msg.mutable_params()->set_timestamp(1);

	string send_msg = VasMsgTrans::Get()->TransSendMsg(&reg_msg, FRAME_TYPE_REG_RESULT);
	ptr_com->Send(send_msg);

	while (true)
	{
		boost::thread::sleep(boost::get_system_time() + boost::posix_time::seconds(500));
	}
}
void TestFeedback()
{
	shared_ptr<VasCom> ptr_com = shared_ptr<VasCom>(new VasCom("../proto/com.prototxt"));
	ptr_com->Start();

	vas::VasFeedback feed_back_msg;

	feed_back_msg.set_psi_name("vasResult");
	//reg_msg.mutable_params()->set_app_key("xdfadsfasdfasdf");
	feed_back_msg.mutable_params()->set_node_id(1);
	feed_back_msg.mutable_params()->set_service_id("service_1");
	feed_back_msg.mutable_params()->set_address_type("rtsp");
	feed_back_msg.mutable_params()->set_feedback_code(1);
	feed_back_msg.mutable_params()->set_feedback_info("ok");
	feed_back_msg.mutable_params()->set_address("rtsp_addr");
	feed_back_msg.mutable_params()->set_timestamp(time(NULL));


	string send_msg = VasMsgTrans::Get()->TransSendMsg(&feed_back_msg, FRAME_TYPE_FEEDBACK);
	ptr_com->Send(send_msg);

	while (true)
	{
		boost::thread::sleep(boost::get_system_time() + boost::posix_time::seconds(500));
	}
}

void TestVasResultReturn()
{
	shared_ptr<VasCom> ptr_com = shared_ptr<VasCom>(new VasCom("../proto/com.prototxt"));
	ptr_com->Start();
	function<void(const string&)> fun = bind(&MsgProcFun, std::placeholders::_1);
	ptr_com->SetReceiveProcessFun(fun);
	while (true)
	{
		boost::thread::sleep(boost::get_system_time() + boost::posix_time::seconds(500));
	}
}
void MsgProcFun(const string &msg)
{
	LOG(INFO) << msg << endl;
	string msg_content = msg.substr(FRAME_HEADER_LEN);
	//VasMsgTrans::Get()->TransSendMsg(&feed_back_msg, FRAME_TYPE_FEEDBACK);
	vas::VasReturn ret_msg;
	VasMsgTrans::Get()->TransReceiveMsg(msg_content, ret_msg);
	LOG(INFO)<< ret_msg.ret_code() << endl;
	LOG(INFO) << ret_msg.ret_msg() << endl;
}