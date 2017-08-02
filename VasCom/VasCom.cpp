#include "stdafx.h"
#include "VasCom.h"
#include "VasIO.h"


#define FRAME_HEADER_LEN 26

VasCom::VasCom(const char *ptoto_file) :client_(this)
{
	CHECK(vas::ReadProtoFromTextFile(ptoto_file, &com_param_));
	CheckParam(com_param_);

}
VasCom::VasCom(const vas::VasComParameter &com_param) : client_(this)
{
	com_param_.CopyFrom(com_param);
	CheckParam(com_param_);
}


VasCom::VasCom() : client_(this)
{
}



VasCom::~VasCom()
{

}

void VasCom::CheckParam(const vas::VasComParameter &com_param_)
{

}

void VasCom::SetAppState(vas::EnAppState state)
{
	com_param_.set_app_state(state);
}
const vas::EnAppState&  VasCom::GetAppState()
{
	return com_param_.app_state();
}

bool VasCom::Start()
{
	string str = com_param_.ip();
	bAsyncConn_ = false;
	force_stop_ = false;
	if (client_.Start(s2ws(str).c_str(), com_param_.port(), false))
	{
		SetAppState(vas::ST_STARTED);
		heart_beat_thread_ = boost::thread(&VasCom::Heartbeat, this);
		return true;
	}
	else
	{
		LOG(INFO) << "Can not connect to the server " << com_param_.ip() << endl;
		heart_beat_thread_ = boost::thread(&VasCom::Heartbeat, this);
		return false;
	}
}
bool VasCom::StartHeartBeat()
{
	if (com_param_.app_state() == vas::ST_STARTED)
	{
		if (heart_beat_msg_)
		{
			if (!heart_beat_thread_.joinable())
			{
				heart_beat_thread_ = boost::thread(&VasCom::Heartbeat, this);
				LOG(INFO) << "Start heart beat: " << heart_beat_msg_ << endl;
				return true;
			}
		}
		else LOG(ERROR) << "Not set the heart beat msg!" << endl;
	}
	else LOG(ERROR) << "Can not connect to the server." << endl;
	return false;
}
bool VasCom::StopHeartBeat()
{
	if (heart_beat_thread_.joinable())
	{
		heart_beat_thread_.interrupt();
		return true;
	}
	return false;
}

void VasCom::Heartbeat()
{
	try
	{
		while (!force_stop_)
		{
			while (com_param_.app_state() == vas::ST_STARTED)
			{
				if (heart_beat_set_)
				{
					Send(heart_beat_msg_, heart_beat_len_);
				}
				else
				{
					LOG(ERROR) << " Heart beat messge not set!" << endl;
				}
				boost::thread::sleep(boost::get_system_time() + boost::posix_time::seconds(com_param_.heartbeat_interval()));
			}
			
			if (force_stop_)
			{
				break;
			}
			while (!CanReconnect())
			{
				LOG(INFO) << " Wait for client clear" << endl;
				boost::thread::sleep(boost::get_system_time() + boost::posix_time::seconds(5));
			}
			string str = com_param_.ip();
			while (!client_.Start(s2ws(str).c_str(), com_param_.port(), false))
			{

				LOG(INFO) << "On Reconnect to the server " << endl;
				boost::thread::sleep(boost::get_system_time() + boost::posix_time::seconds(5));

			}
			LOG(INFO) << "Reconnect to the server success!" << endl;
			SetAppState(vas::ST_STARTED);
		}
		
	}
	catch (boost::thread_interrupted&)//捕获线程中断异常  
	{
		LOG(INFO) << "Stop connect" << endl;
	}
}
void VasCom::Stop()
{
	force_stop_ = true;
	SetAppState(vas::ST_STOPPED);
	StopHeartBeat();
	client_.Stop();
}
bool VasCom::Send(const string &msg)
{
	BYTE *frameBuffer = new BYTE[4096];
	memset(frameBuffer, 0, sizeof(BYTE) * 4096);
	size_t Len = msg.length();
	if (Len>4096)
	{
		delete frameBuffer;
		return false;
	}
	memcpy(frameBuffer, msg.c_str(), Len);
	bool ret = Send(frameBuffer, Len);
	delete frameBuffer;
	return ret;

}

void VasCom::SetLinkStateProcessFun(signal_link_state::slot_function_type fun)
{
	link_signal_.connect(fun);
}
void VasCom::SetReceiveProcessFun(recevie_msg_signal::slot_function_type fun)
{
	recevie_msg_signal_.connect(fun);
}
void VasCom::SetHeartBeatMessage(const char* heart_beat_msg,int length)
{
	memcpy(heart_beat_msg_, heart_beat_msg, length);
	heart_beat_len_ = length;
	heart_beat_set_ = true;
}
void VasCom::SetHeartBeatInterval(int interval_sec)
{
	com_param_.set_heartbeat_interval(interval_sec);
}
bool VasCom::Send(const BYTE *pBuffer, int iLen)
{
	send_mutex_.try_lock();
	if (client_.Send(pBuffer, iLen))
	{
		send_mutex_.unlock();
		return true;
	}
	else
	{
		send_mutex_.unlock();
		return false;
	}
}

EnHandleResult VasCom::OnSend(ITcpClient* pSender, CONNID dwConnID, const BYTE* pData, int iLength)
{
	//TCHAR szAddress[40];
	//int iAddressLen = sizeof(szAddress) / sizeof(TCHAR);
	//USHORT usPort;
	//pSender->GetLocalAddress(szAddress, iAddressLen, usPort);

	on_send_mutex_.try_lock();
	char msg[4*1024] = {0};
	memcpy(msg, pData + FRAME_HEADER_LEN, iLength);
	msg[iLength - FRAME_HEADER_LEN] = '\0';
	LOG(INFO) << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Send msg:" << msg << endl;
	SetAppState(vas::ST_STARTED);
	on_send_mutex_.unlock();
	return HR_OK;
}

EnHandleResult VasCom::OnReceive(ITcpClient* pSender, CONNID dwConnID, const BYTE* pData, int iLength)
{
	on_receive_mutex_.try_lock();
	char msg[4 * 1024];
	memset(msg, 0, sizeof(char) * 4 * 1024);
	if (iLength<FRAME_HEADER_LEN)
	{
		return HR_OK;
	}
	memcpy(msg, pData + FRAME_HEADER_LEN, iLength - FRAME_HEADER_LEN );
	msg[iLength - FRAME_HEADER_LEN] = '\0';
	LOG(INFO) << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Receive msg: "<< string(msg) << endl;
	if (iLength - FRAME_HEADER_LEN>0)
	{
		recevie_msg_signal_(string(msg));
	}
	on_receive_mutex_.unlock();
	return HR_OK;
}

bool VasCom::CanReconnect()
{
	return client_.GetState() == SS_STOPPED;
}
EnHandleResult VasCom::OnClose(ITcpClient* pSender, CONNID dwConnID, EnSocketOperation enOperation, int iErrorCode)
{
	LOG(INFO) << "Close connect, id:" << dwConnID << endl;
	SetAppState(vas::ST_STOPPED);
	link_signal_(vas::ST_STOPPED);
	return HR_OK;
}

EnHandleResult VasCom::OnConnect(ITcpClient* pSender, CONNID dwConnID)
{
	TCHAR szAddress[40];
	int iAddressLen = sizeof(szAddress) / sizeof(TCHAR);
	USHORT usPort;
	pSender->GetLocalAddress(szAddress, iAddressLen, usPort);
	LOG(INFO) << " connect " << dwConnID << " ip: " << wct2s(szAddress) << ", user port: " << (int)usPort << endl;
	SetAppState(vas::ST_STARTED);
	link_signal_(vas::ST_STARTED);
	return HR_OK;
}
wstring VasCom::s2ws(string s)
{
	int len;
	int slength = (int)s.length() + 1;
	len = MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, 0, 0);
	wchar_t* buf = new wchar_t[len];
	MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, buf, len);
	std::wstring r(buf);
	delete[] buf;
	return r;
}
 string VasCom::wct2s(wchar_t *wchar)
{
	wchar_t * wText = wchar;
	DWORD dwNum = WideCharToMultiByte(CP_OEMCP, NULL, wText, -1, NULL, 0, NULL, FALSE);// WideCharToMultiByte的运用
	char *psText;  // psText为char*的临时数组，作为赋值给std::string的中间变量
	psText = new char[dwNum];
	WideCharToMultiByte(CP_OEMCP, NULL, wText, -1, psText, dwNum, NULL, FALSE);// WideCharToMultiByte的再次运用
	string szDst = psText;// std::string赋值
	delete[]psText;// psText的清除
	return szDst;
}