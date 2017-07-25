// VideoSmokeDetection.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include"VasIO.h"
#include"Test.h"


int main(int argc, char *argv[])
{
	google::InitGoogleLogging(argv[0]);
	FLAGS_alsologtostderr = true;  //设置日志消息除了日志文件之外是否去标准输出
	FLAGS_colorlogtostderr = true;  //设置记录到标准输出的颜色消息（如果终端支持）
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


