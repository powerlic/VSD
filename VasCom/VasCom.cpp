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
	//SetAppState(vas::ST_STARTED);
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
	cout << s2ws(str).c_str() << endl;
	if (client_.Start(s2ws(str).c_str(), com_param_.port(), bAsyncConn_))
	{
		SetAppState(vas::ST_STARTED);
		return true;
	}
	else
	{
		LOG(INFO) << "Can not connect to the server " << com_param_.ip() << endl;
		SetAppState(vas::ST_STOPPED);
	}
}
bool VasCom::StartHeartBeat()
{
	if (com_param_.app_state() == vas::ST_STARTED)
	{
		if (!heart_beat_msg_.empty())
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
		while (com_param_.app_state() == vas::ST_STARTED)
		{
			LOG(INFO) << "Heart beat:" << heart_beat_msg_ << endl;
			Send(heart_beat_msg_);
			boost::thread::sleep(boost::get_system_time() + boost::posix_time::seconds(heart_beat_interval_sec_));
		}
	}
	catch (boost::thread_interrupted&)//捕获线程中断异常  
	{
		LOG(INFO) << "Heart beat stop" << endl;
	}
}
void VasCom::Stop()
{
	SetAppState(vas::ST_STOPPED);
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
void VasCom::SetReceiveProcessFun(recevie_msg_signal::slot_function_type fun)
{
	recevie_msg_signal_.connect(fun);
}
void VasCom::SetHeartBeatMessage(const string &heart_beat_msg)
{
	heart_beat_msg_ = heart_beat_msg;
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
	TCHAR szAddress[40];
	int iAddressLen = sizeof(szAddress) / sizeof(TCHAR);
	USHORT usPort;
	pSender->GetLocalAddress(szAddress, iAddressLen, usPort);
	char *msg = new char[iLength + 1];
	memcpy(msg, pData, iLength);
	msg[iLength] = '\0';
	LOG(INFO) << " Send msg:" << msg << endl;
	delete msg;
	SetAppState(vas::ST_STARTED);
	return HR_OK;
}

EnHandleResult VasCom::OnReceive(ITcpClient* pSender, CONNID dwConnID, const BYTE* pData, int iLength)
{
	char *msg_ = new char[iLength + 1];
	memcpy(msg_, pData, iLength);
	msg_[iLength] = '\0';
	string msg = msg_;
	LOG(INFO) << " Receive msg:" << msg << endl;
	if (iLength>0)
	{
		recevie_msg_signal_(msg);
	}
	delete msg_;
	return HR_OK;
}

EnHandleResult VasCom::OnClose(ITcpClient* pSender, CONNID dwConnID, EnSocketOperation enOperation, int iErrorCode)
{
	SetAppState(vas::ST_STOPPED);
	LOG(INFO) << "Close connect, id:" << dwConnID << endl;
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