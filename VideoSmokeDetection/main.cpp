// VideoSmokeDetection.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"
#include"VasIO.h"
#include"Test.h"


int main(int argc, char *argv[])
{
	google::InitGoogleLogging(argv[0]);
	FLAGS_alsologtostderr = true;  //������־��Ϣ������־�ļ�֮���Ƿ�ȥ��׼���
	FLAGS_colorlogtostderr = true;  //���ü�¼����׼�������ɫ��Ϣ������ն�֧�֣�
	FLAGS_log_dir = ".\\Logs";
	FLAGS_colorlogtostderr = true;
	google::SetLogDestination(google::GLOG_INFO, ".\\Logs\\INFO_");
	google::SetStderrLogging(google::GLOG_INFO);
	google::SetLogDestination(google::GLOG_ERROR, ".\\Logs\\ERROR_");
	google::SetStderrLogging(google::GLOG_ERROR);
	//google::SetLogFilenameExtension("txt");
	//


	//TestBgVibeReadFromTxtFile();
	//TestBgGuassianReadFromTxtFile();
	//TestBgVibe();
	//TestFilter();
	//TestDetectMaster();
	//TestDvrDetect();
	//TestDecodeMaster();
	//TestBgGuassian();
	//TestVas();


	TestVasServices();
	//TestMergeRects();

	google::ShutdownGoogleLogging();
	return 0;
}


