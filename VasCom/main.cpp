// VasCom.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include"VasCom.h"
#include "json2pb.h"
#include"VasIO.h"
#include"ComTest.h"
void MsgProcFun(const string &msg)
{
	LOG(INFO) <<"Outside Recevie"<< msg << endl;
}

void testJson2Proto()
{
	vas::BgParameter bg_param;
	vas::ReadProtoFromTextFile("../proto/bg.prototxt", &bg_param);
	string str = pb2json(bg_param);
	cout << str << endl;
}


int _tmain1(int argc, _TCHAR* argv[])
{

	testJson2Proto();

	shared_ptr<VasCom> ptr_com = shared_ptr<VasCom>(new VasCom("../proto/com.prototxt"));
	ptr_com->Start();
	ptr_com->SetHeartBeatMessage("Heart beat msg");
	ptr_com->StartHeartBeat();
	std::function<void(const string&)> fun = std::bind(&MsgProcFun, std::placeholders::_1);
	ptr_com->SetReceiveProcessFun(fun);
	system("pause");

	return 0;
}

int main()
{
	//TestHeartbeat();
	TestSendRegResult();
	system("pause");
	return 0;
}
