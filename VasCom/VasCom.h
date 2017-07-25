#pragma once
#include"stdafx.h"
#include "TcpClient.h"
#include "VasProto.prototxt.pb.h"








class VasCom : public CTcpClientListener
{
	typedef boost::signals::connection  connection_t;
	typedef boost::signal<void(const string &receive_msg)>  recevie_msg_signal;

public:
	
	VasCom(const char *ptoto_file);
	VasCom(const vas::VasComParameter &com_param);
	VasCom(const char *ip, const int port,const int heart_beat_interval);
	virtual ~VasCom();

	bool Start();
	void Stop();
	bool Send(const BYTE *pBuffer, int iLen);

	bool StartHeartBeat();
	bool StopHeartBeat();

	bool Send(const string &msg);
	void SetAppState(vas::EnAppState state);
	const vas::EnAppState&  GetAppState();
	//callback
	void SetReceiveProcessFun(recevie_msg_signal::slot_function_type fun);
	void SetHeartBeatMessage(const string &heart_beat_msg);

private:
	VasCom();
	CTcpClient client_;
	vas::VasComParameter com_param_;
	void CheckParam(const vas::VasComParameter &com_param);
	bool bAsyncConn_ = false;

	string heart_beat_msg_;
	int heart_beat_interval_sec_ = 1;
	void Heartbeat();
	boost::thread heart_beat_thread_;

	virtual EnHandleResult OnSend(ITcpClient* pSender, CONNID dwConnID, const BYTE* pData, int iLength);
	virtual EnHandleResult OnReceive(ITcpClient* pSender, CONNID dwConnID, const BYTE* pData, int iLength);
	virtual EnHandleResult OnClose(ITcpClient* pSender, CONNID dwConnID, EnSocketOperation enOperation, int iErrorCode);
	virtual EnHandleResult OnConnect(ITcpClient* pSender, CONNID dwConnID);


	boost::mutex send_mutex_;
	recevie_msg_signal recevie_msg_signal_;
	wstring s2ws(string s);
	string wct2s(wchar_t *wchar);
};

