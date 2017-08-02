#pragma once
#include"stdafx.h"
#include"VasP.h"

using namespace std;
using namespace cv;

class VsdMaster
{
public:
	VsdMaster(const char*caffe_model_path_proto_file);
	~VsdMaster();

	shared_ptr<VasP> ptr_vas;
private:

	/*Implement your callback functiones to replace them!
	*/
	void RegResultProcessFun(const char* service_id, Mat &frame, const vector<CvRect> &reg_rects,
		const uint32_t dst_width, const uint32_t dst_height,
		const vector<float> &scores, const vector<string> &labels);

	void CheckSmokeRegFun(const char* service_id, const vector<CvRect> &reg_rects,
		const uint32_t dst_width, const uint32_t dst_height,
		const vector<float> &scores, const vector<string> &labels);

	void ServiceCommandReplyProcessFun(const char* service_id, const vas::ServiceCommandReply &reply, const string &info);
	void StreamErrProcessFun(const char *service_id, vas::StreamStatus status);

	bool find_smoke_ = false;
};


class VsdNvrMaster
{
public:
	VsdNvrMaster(const char*caffe_model_path_proto_file, const char*nvr_access_proto_file);
	~VsdNvrMaster();
	shared_ptr<VasNvrP> ptr_vas_nvr_;

private:
	void RegResultProcessFun(const char* service_id, Mat &frame, const vector<CvRect> &reg_rects,
							 const uint32_t dst_width, const uint32_t dst_height,
		                     const vector<float> &scores, const vector<string> &labels);
	void ServiceCommandReplyProcessFun(const char* service_id, const vas::ServiceCommandReply &reply, const string &info);
	void StreamErrProcessFun(const char *service_id, vas::CamStatus status);

	bool find_smoke_ = false;
};

class VsdSsdMaster
{
public:
	VsdSsdMaster(const char*ssd_model_path_proto_file);
	~VsdSsdMaster();
	shared_ptr<VasSsdP> ptr_vas_ssd_;

private:
	void RegResultProcessFun(const char* service_id, Mat &frame, const vector<CvRect> &reg_rects,
		const uint32_t dst_width, const uint32_t dst_height,
		const vector<float> &scores, const vector<string> &labels);
	void ServiceCommandReplyProcessFun(const char* service_id, const vas::ServiceCommandReply &reply, const string &info);
	void StreamErrProcessFun(const char *service_id, vas::StreamStatus status);
};

