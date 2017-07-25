// VsdTest.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include"VsdMaster.h"
#include <gflags\gflags.h>



void TestVas(const char* caffe_model_path,const char* serivce_list_path);
void TestNvrService(const char* caffe_model_path, const char *nvr_param_path, const char* serivce_list_path);
void TestVibeBoost();



DEFINE_string(ServiceType, " ", "Nvr, Rtsp");
DEFINE_string(CaffeModelPath, "../proto/caffe_model_0523.prototxt", "protofile for load caffe model file");
DEFINE_string(NvrParamPath, "../proto/nvr.prototxt", "protofile for load nvr");
DEFINE_string(ServiceFilePath, "../proto/rtsp_service_list.prototxt", "nvr, rtsp service list");


void CheckLogFilePath()
{
	string file_path;
	file_path = "./Logs";
	fstream reg_file;
	reg_file.open(file_path, ios::in);
	if (!reg_file)
	{
		_mkdir(file_path.c_str());
	}
}

int main_1(int argc, char *argv[])
{
	std::string usage("");
	usage = std::string(argv[0]) + string("\n --ServiceType=\"Nvr\" \n --NvrParamPath==\"../proto/nvr.prototxt\" \n --CaffeModelPath= \"../proto/caffe_model_0523.prototxt\" \n --ServiceFilePath=\"../proto/nvr_service_list.prototxt \" \n");
	usage += std::string(argv[0]) + " \n --ServiceType=\"Rtsp\" \n --CaffeModelPath= \"../proto/caffe_model_0523.prototxt\" \n --ServiceFilePath=\"../proto/rtsp_service_list.prototxt\" \n ";
	google::SetUsageMessage(usage);
	CheckLogFilePath();
	google::InitGoogleLogging(argv[0]);
	FLAGS_alsologtostderr = true;  //设置日志消息除了日志文件之外是否去标准输出
	FLAGS_colorlogtostderr = true;  //设置记录到标准输出的颜色消息（如果终端支持）
	FLAGS_log_dir = ".\\Logs";
	FLAGS_colorlogtostderr = true;
	google::SetLogDestination(google::GLOG_INFO, ".\\Logs\\INFO_");
	google::SetStderrLogging(google::GLOG_INFO);
	google::SetLogDestination(google::GLOG_ERROR, ".\\Logs\\ERROR_");
	google::SetStderrLogging(google::GLOG_ERROR);

	if (argc<3)
	{
		gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_imageset");
	}


	google::ParseCommandLineFlags(&argc, &argv, false);
	

	if (FLAGS_ServiceType.empty() || FLAGS_CaffeModelPath.empty() || FLAGS_ServiceFilePath.empty())
	{
		google::ShowUsageWithFlagsRestrict(argv[0], "How to use VideoSmokeDetection.");
		return 1;
	}



	const string &source_type = FLAGS_ServiceType;
	const string &caffe_model_path = FLAGS_CaffeModelPath;
	string nvr_param;
	if (source_type.compare("Nvr")==0)
	{
		if (FLAGS_NvrParamPath.empty())
		{
			google::ShowUsageWithFlagsRestrict(argv[0], "How to use VideoSmokeDetection.");
			return 1;
		}
		nvr_param = FLAGS_NvrParamPath;
	}

	const string &service_list_path = FLAGS_ServiceFilePath;

	
	if (source_type.compare("Nvr") == 0)
	{
		TestNvrService(caffe_model_path.c_str(), nvr_param.c_str(), service_list_path.c_str());
	}
	else if (source_type.compare("Rtsp") == 0)
	{
		TestVas(caffe_model_path.c_str(), service_list_path.c_str());
	}

	 //TestVas();
	while (true)
	{
		Sleep(1000);
	}


	google::ShutdownGoogleLogging();

	return 0;

}

void TestVas(const char* caffe_model_path, const char* serivce_list_path)
{
	shared_ptr<VsdMaster> ptr_demo = shared_ptr<VsdMaster>(new VsdMaster(caffe_model_path));

	//const char *service_id = "test_inner";
	//int url_type = 1;//mp4 video_file
	//const char *url = "E:\\Video\\208\\IMG_7964.mp4";
	//int decode_mode = 0;//cpu decode
	//uint32_t reg_width = 800;
	//uint32_t reg_height = 576;
	//vector<string> reg_types;
	//reg_types.push_back("smoke");
	//float smoke_detect_sensitity = 0.8;//灵敏度越高，越容易误报
	//float fire_detect_sensitity = 0.2;
	//ptr_demo->ptr_vas->AddService(service_id, url, url_type, decode_mode, reg_width, reg_height, reg_types, smoke_detect_sensitity, fire_detect_sensitity);

	ptr_demo->ptr_vas->StartServicesFromProtoFile(serivce_list_path);

	while (true)
	{
		Sleep(1000);
	}
}

void TestNvrService(const char* caffe_model_path, const char *nvr_param_path, const char* serivce_list_path)
{
	shared_ptr<VsdNvrMaster> ptr_demo = shared_ptr<VsdNvrMaster>(new VsdNvrMaster(caffe_model_path, nvr_param_path));
	//const char *service_id_1 = "QiangJi_1";
	//const char *service_id_2 = "BanQiuJi_2";
	//int channel_no_1 = 0;
	//int channel_no_2 = 1;
	//int decode_mode = 0;//cpu decode
	//uint32_t reg_width = 800;
	//uint32_t reg_height = 576;
	//vector<string> reg_types;
	//reg_types.push_back("smoke");
	//reg_types.push_back("car");
	//float smoke_detect_sensitity = 0.8;//灵敏度越高，越容易误报
	//float fire_detect_sensitity = 0.2;
	//ptr_demo->ptr_vas_nvr_->AddService(service_id_1, channel_no_1, reg_width, reg_height, reg_types, smoke_detect_sensitity, fire_detect_sensitity);
	//ptr_demo->ptr_vas_nvr_->AddService(service_id_2, channel_no_2, reg_width, reg_height, reg_types, smoke_detect_sensitity, fire_detect_sensitity);

	ptr_demo->ptr_vas_nvr_->StartServicesFromProtoFile(serivce_list_path);


	while (true)
	{
		Sleep(1000);
	}
}
int main()
{
	TestVibeBoost();
}
void TestVibeBoost()
{
	const char* caffe_model_path = "..\\proto\\caffe_model_0523.prototxt";
	const char* service_path = "..\\proto\\service_video_file_vibe_boost.prototxt";
	shared_ptr<VsdMaster> ptr_demo = shared_ptr<VsdMaster>(new VsdMaster(caffe_model_path));

	ptr_demo->ptr_vas->StartServicesFromProtoFile(service_path);
	while (true)
	{
		Sleep(1000);
	}
}