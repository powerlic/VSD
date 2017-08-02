#pragma once
#include"stdafx.h"
#include "TcpClient.h"
#include "VasProto.prototxt.pb.h"


class VasCom : public CTcpClientListener
{
	typedef boost::signals::connection  connection_t;
	typedef boost::signal<void(const string &receive_msg)>  recevie_msg_signal;
	typedef boost::signal<void(const vas::EnAppState&)>  signal_link_state;

public:
	VasCom(const char *ptoto_file);
	VasCom(const vas::VasComParameter &com_param);
	VasCom(const char *ip, const int port,const int heart_beat_interval);
	virtual ~VasCom();

	bool Start();
	void Stop();
	bool Send(const BYTE *pBuffer, int iLen);

	bool CanReconnect();

	

	bool Send(const string &msg);
	
	const vas::EnAppState&  GetAppState();
	//callback
	void SetReceiveProcessFun(recevie_msg_signal::slot_function_type fun);
	void SetLinkStateProcessFun(signal_link_state::slot_function_type fun);


	void SetHeartBeatMessage(const char* heart_beat_msg, int length);
	void SetHeartBeatInterval(int interval_sec);

private:
	VasCom();
	CTcpClient client_;
	vas::VasComParameter com_param_;
	void CheckParam(const vas::VasComParameter &com_param);
	bool bAsyncConn_ = false;

	bool StartHeartBeat();
	bool StopHeartBeat();


	void SetAppState(vas::EnAppState state);
	BYTE heart_beat_msg_[4 * 1024];
	int heart_beat_len_;
	bool heart_beat_set_ = false;
	void Heartbeat();
	boost::thread heart_beat_thread_;

	virtual EnHandleResult OnSend(ITcpClient* pSender, CONNID dwConnID, const BYTE* pData, int iLength);
	virtual EnHandleResult OnReceive(ITcpClient* pSender, CONNID dwConnID, const BYTE* pData, int iLength);
	virtual EnHandleResult OnClose(ITcpClient* pSender, CONNID dwConnID, EnSocketOperation enOperation, int iErrorCode);
	virtual EnHandleResult OnConnect(ITcpClient* pSender, CONNID dwConnID);


	boost::mutex send_mutex_;
	boost::mutex on_send_mutex_;
	boost::mutex on_receive_mutex_;

	recevie_msg_signal recevie_msg_signal_;
	connection_t receive_signal_connetct_;

	signal_link_state link_signal_;
	connection_t link_signal_connetct_;



	wstring s2ws(string s);
	string wct2s(wchar_t *wchar);

	bool force_stop_ = false;
};

